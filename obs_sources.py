"""Phase 1：ISD-Lite / IGRA2 解析、QC、台站筛选和本地缓存。

数据下载由 fetch_obs.py 负责；本模块只读取 cache 下的原始文件，
把选定台站解析为紧凑的 npz，供 Phase 2/3 的 verify_obs.py 使用。

准备缓存：

    python -u obs_sources.py --prepare \
      --cache-dir cache \
      --init-times 2022-01-02T00:00:00,... \
      --lead-hours 240 \
      --isd-target 600 --igra-target 200

自检：

    python -u obs_sources.py --self-test --cache-dir cache
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from config import parse_utc

ISD_DIR = "isd_lite/2022"
IGRA_DIR = "igra2"
PARSED_DIR = "obs_parsed"
TARGET_LAT = np.linspace(90.0, -89.75, 720)


@dataclass(frozen=True)
class Station:
    sid: str
    name: str
    lat: float
    lon: float
    elev: float
    country: str = ""
    state: str = ""


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def hour_of(t: datetime) -> int:
    return int(t.replace(tzinfo=timezone.utc).timestamp() // 3600)


def hours_for(init_times: list[datetime], lead_hours: int) -> np.ndarray:
    hours = set()
    for init in init_times:
        for lead in range(6, lead_hours + 1, 6):
            hours.add(hour_of(init + timedelta(hours=lead)))
    return np.array(sorted(hours), dtype=np.int64)


def _bin_key(lat: float, lon: float) -> tuple[int, int]:
    return int((lat + 90.0) // 30.0), int((lon + 180.0) // 60.0)


def _round_robin(groups: dict, target: int, per_bin: int) -> list:
    selected = []
    for _ in range(per_bin):
        for key in sorted(groups):
            if len(selected) >= target:
                return selected
            if groups[key]:
                selected.append(groups[key].pop(0))
    while len(selected) < target:
        added = False
        for key in sorted(groups):
            if len(selected) >= target:
                break
            if groups[key]:
                selected.append(groups[key].pop(0))
                added = True
        if not added:
            break
    return selected


# ---------------------------------------------------------------------------
# 台站元数据
# ---------------------------------------------------------------------------


def load_isd_stations(cache_dir: Path) -> list[Station]:
    path = cache_dir / "isd-history.csv"
    stations = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            try:
                lat = float(row["LAT"])
                lon = float(row["LON"])
                elev = float(row["ELEV(M)"])
            except (TypeError, ValueError):
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            if abs(lat) < 1e-6 and abs(lon) < 1e-6:
                continue
            if elev <= -999:
                continue
            if row.get("WBAN") == "99999":
                continue
            begin = row.get("BEGIN", "")
            end = row.get("END", "")
            if not begin or not end or end < "20221231" or begin > "20220101":
                continue
            stations.append(
                Station(
                    sid=f"{row['USAF']}-{row['WBAN']}",
                    name=row.get("STATION NAME", "").strip(),
                    lat=lat,
                    lon=lon,
                    elev=elev,
                    country=row.get("CTRY", "").strip(),
                    state=row.get("STATE", "").strip(),
                )
            )
    return stations


def load_igra_stations(cache_dir: Path) -> list[Station]:
    path = cache_dir / "igra2-station-list.txt"
    stations = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                sid = line[0:11].strip()
                lat = float(line[11:20])
                lon = float(line[20:30])
                elev = float(line[30:38])
                name = line[38:69].strip()
                first = int(line[72:76])
                last = int(line[77:81])
                count = int(line[82:88])
            except (ValueError, IndexError):
                continue
            if not sid or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            if elev <= -999 or count <= 100:
                continue
            if last < 2022 or first > 2022:
                continue
            stations.append(
                Station(sid=sid, name=name, lat=lat, lon=lon, elev=elev)
            )
    return stations


# ---------------------------------------------------------------------------
# ISD-Lite 解析
# ---------------------------------------------------------------------------


def parse_isd_lite(path: Path) -> dict[str, np.ndarray] | None:
    """解析单个 ISD-Lite 2022 文件为按时间排序的数组。"""
    hours: list[int] = []
    t2: list[float] = []
    msl: list[float] = []
    u10: list[float] = []
    v10: list[float] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 12:
                continue
            try:
                year, month, day, hour = (int(x) for x in parts[:4])
                temp, _dew, slp, wdir, wspd, _sky, _p01, _p06 = (
                    int(x) for x in parts[4:12]
                )
            except ValueError:
                continue
            if year != 2022:
                continue
            try:
                t = datetime(year, month, day, hour, tzinfo=timezone.utc)
            except ValueError:
                continue
            hours.append(hour_of(t))
            t2.append(temp / 10.0 + 273.15 if temp != -9999 and -900 <= temp <= 600 else np.nan)
            msl.append(slp / 10.0 if slp != -9999 and 8700 <= slp <= 10850 else np.nan)
            uv = (np.nan, np.nan)
            if wspd != -9999 and 0 <= wspd <= 750:
                speed = wspd / 10.0
                if wdir != -9999 and 0 <= wdir <= 360:
                    rad = math.radians(wdir)
                    uv = (-speed * math.sin(rad), -speed * math.cos(rad))
                elif speed == 0:
                    uv = (0.0, 0.0)
            u10.append(uv[0])
            v10.append(uv[1])
    if not hours:
        return None
    order = np.argsort(hours, kind="stable")
    return {
        "time": np.asarray(hours, dtype=np.int64)[order],
        "2t": np.asarray(t2, dtype=np.float32)[order],
        "msl": np.asarray(msl, dtype=np.float32)[order],
        "10u": np.asarray(u10, dtype=np.float32)[order],
        "10v": np.asarray(v10, dtype=np.float32)[order],
    }


def value_at(records: dict[str, np.ndarray], hour: int, key: str,
             tolerance: float = 0.5) -> float:
    if key not in records:
        return np.nan
    times = records["time"]
    value = records[key]
    if times.size == 0:
        return np.nan
    idx = int(np.searchsorted(times, hour))
    candidates = []
    if idx < times.size:
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    best = min(candidates, key=lambda i: abs(int(times[i]) - hour))
    if abs(int(times[best]) - hour) > tolerance:
        return np.nan
    return float(value[best])


def isd_coverage(records: dict[str, np.ndarray], target_hours: np.ndarray) -> dict:
    case = {k: 0 for k in ("2t", "msl", "10u")}
    for h in target_hours:
        for key in case:
            value = value_at(records, int(h), key)
            if np.isfinite(value):
                case[key] += 1
    year = {k: 0 for k in ("2t", "msl", "10u")}
    times = records["time"]
    for key in year:
        values = records[key]
        mask = np.isfinite(values) & (times % 6 == 0)
        year[key] = int(mask.sum())
    return {"case": case, "year": year}


# ---------------------------------------------------------------------------
# IGRA2 解析
# ---------------------------------------------------------------------------


def parse_igra_2022(path: Path, target_hours: set[int] | None = None) -> dict[str, np.ndarray] | None:
    """解析 IGRA2 data-por zip 中 2022 年 00/12 UTC 的 850/500 hPa 要素。"""
    hours: list[int] = []
    t850: list[float] = []
    z500: list[float] = []
    u850: list[float] = []
    v850: list[float] = []
    u500: list[float] = []
    v500: list[float] = []

    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            current: int | None = None
            current_idx: int | None = None
            for line in text:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith("#"):
                    try:
                        year = int(line[13:17])
                        month = int(line[18:20])
                        day = int(line[21:23])
                        hour = int(line[24:26])
                    except (ValueError, IndexError):
                        current = None
                        current_idx = None
                        continue
                    if year > 2022:
                        break
                    if year == 2022 and hour in (0, 12):
                        h = hour_of(datetime(year, month, day, hour, tzinfo=timezone.utc))
                        if target_hours is None or h in target_hours:
                            hours.append(h)
                            t850.append(np.nan)
                            z500.append(np.nan)
                            u850.append(np.nan)
                            v850.append(np.nan)
                            u500.append(np.nan)
                            v500.append(np.nan)
                            current = h
                            current_idx = len(hours) - 1
                        else:
                            current = None
                            current_idx = None
                    else:
                        current = None
                        current_idx = None
                    continue
                if current is None or current_idx is None:
                    continue
                try:
                    pressure = int(line[9:15])
                    height = int(line[16:21])
                    temp = int(line[22:27])
                    wdir = int(line[34:39])
                    wspd = int(line[40:45])
                except (ValueError, IndexError):
                    continue
                if abs(pressure - 85000) <= 500 and -1000 <= temp <= 600:
                    t850[current_idx] = temp / 10.0 + 273.15
                    if wdir != -9999 and wspd != -9999 and 0 <= wdir <= 360:
                        speed = wspd / 10.0
                        rad = math.radians(wdir)
                        u850[current_idx] = -speed * math.sin(rad)
                        v850[current_idx] = -speed * math.cos(rad)
                # IGRA2 的位势高度缺测码是 -8888（不是 -9999）；500 hPa 高度应为正。
                if abs(pressure - 50000) <= 500 and height > 0:
                    z500[current_idx] = float(height)
                    if wdir != -9999 and wspd != -9999 and 0 <= wdir <= 360:
                        speed = wspd / 10.0
                        rad = math.radians(wdir)
                        u500[current_idx] = -speed * math.sin(rad)
                        v500[current_idx] = -speed * math.cos(rad)

    if not hours:
        return None
    order = np.argsort(hours, kind="stable")
    out = {
        "time": np.asarray(hours, dtype=np.int64)[order],
        "t850": np.asarray(t850, dtype=np.float32)[order],
        "z500": np.asarray(z500, dtype=np.float32)[order],
        "u850": np.asarray(u850, dtype=np.float32)[order],
        "v850": np.asarray(v850, dtype=np.float32)[order],
        "u500": np.asarray(u500, dtype=np.float32)[order],
        "v500": np.asarray(v500, dtype=np.float32)[order],
    }
    return out


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------


def _save(path: Path, records: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez(tmp, **records)
    os.replace(tmp, path)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def prepare_isd_cache(
    cache_dir: Path,
    init_times: list[datetime],
    lead_hours: int,
    target: int = 600,
    max_per_bin: int = 25,
    workers: int = 8,
) -> list[dict]:
    stations = load_isd_stations(cache_dir)
    raw_dir = cache_dir / ISD_DIR
    parsed_dir = cache_dir / PARSED_DIR / "isd"
    target_hours = hours_for(init_times, lead_hours)
    log(f"ISD 缓存：{len(stations)} 个站，目标 {len(target_hours)} 个时次，解析中 ...")

    def work(s: Station):
        path = raw_dir / f"{s.sid}-2022.gz"
        if not path.exists():
            return None
        records = parse_isd_lite(path)
        if records is None:
            return None
        cov = isd_coverage(records, target_hours)
        return s, records, cov

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(work, s): s for s in stations}
        for done, fut in enumerate(as_completed(futures), start=1):
            item = fut.result()
            if item is not None:
                results.append(item)
            if done % 400 == 0 or done == len(futures):
                log(f"  ISD {done}/{len(futures)}")

    # 覆盖要求：2t/msl 至少 20% 的 case 时次；10u 不强制。
    min_case = max(20, int(0.2 * len(target_hours)))
    good = [
        item for item in results
        if item[2]["case"]["2t"] >= min_case and item[2]["case"]["msl"] >= min_case
    ]
    good.sort(
        key=lambda x: (x[2]["case"]["10u"] > 0, x[2]["case"]["2t"] + x[2]["case"]["msl"]),
        reverse=True,
    )
    # 国家配额：避免美国站（ISD 中数量最多）主导全球统计。
    capped = []
    country_counts: dict[str, int] = defaultdict(int)
    for item in good:
        country = item[0].country or "XX"
        cap = 120 if country == "US" else 25
        if country_counts[country] < cap:
            capped.append(item)
            country_counts[country] += 1
    good = capped
    log(f"  国家配额后候选 {len(good)}（US {country_counts.get('US', 0)}）")

    groups: dict[tuple[int, int], list] = defaultdict(list)
    for item in good:
        groups[_bin_key(item[0].lat, item[0].lon)].append(item)
    selected = _round_robin(groups, min(target, len(good)), max_per_bin)

    info = []
    for s, records, cov in selected:
        _save(parsed_dir / f"{s.sid}.npz", records)
        info.append(
            {
                "sid": s.sid, "name": s.name, "lat": s.lat, "lon": s.lon,
                "elev": s.elev, "country": s.country, "state": s.state,
                "case_2t": cov["case"]["2t"], "case_msl": cov["case"]["msl"],
                "case_10u": cov["case"]["10u"],
                "year_2t": cov["year"]["2t"], "year_msl": cov["year"]["msl"],
                "year_10u": cov["year"]["10u"],
            }
        )
    with open(cache_dir / PARSED_DIR / "stations_isd.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["sid", "name", "lat", "lon", "elev", "country", "state",
                  "case_2t", "case_msl", "case_10u", "year_2t", "year_msl", "year_10u"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(info)
    log(f"ISD 缓存完成：通过覆盖筛选 {len(good)}，选中 {len(selected)}")
    return info


def prepare_igra_cache(
    cache_dir: Path,
    init_times: list[datetime],
    lead_hours: int,
    target: int = 200,
    max_per_bin: int = 10,
    workers: int = 4,
) -> list[dict]:
    stations = load_igra_stations(cache_dir)
    raw_dir = cache_dir / IGRA_DIR
    parsed_dir = cache_dir / PARSED_DIR / "igra"
    all_hours = hours_for(init_times, lead_hours)
    target_hours = {int(h) for h in all_hours if datetime.fromtimestamp(int(h) * 3600, tz=timezone.utc).hour in (0, 12)}
    log(f"IGRA2 缓存：{len(stations)} 个站，目标 {len(target_hours)} 个 00/12 时次，解析中 ...")

    def work(s: Station):
        path = raw_dir / f"{s.sid}-data.txt.zip"
        if not path.exists():
            return None
        records = parse_igra_2022(path, target_hours)
        if records is None:
            return None
        n_t850 = int(np.isfinite(records["t850"]).sum())
        n_z500 = int(np.isfinite(records["z500"]).sum())
        return s, records, n_t850, n_z500

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(work, s): s for s in stations}
        for done, fut in enumerate(as_completed(futures), start=1):
            item = fut.result()
            if item is not None:
                results.append(item)
            if done % 50 == 0 or done == len(futures):
                log(f"  IGRA2 {done}/{len(futures)}")

    good = [x for x in results if x[2] >= 10 and x[3] >= 10]
    good.sort(key=lambda x: x[2] + x[3], reverse=True)
    groups: dict[tuple[int, int], list] = defaultdict(list)
    for item in good:
        groups[_bin_key(item[0].lat, item[0].lon)].append(item)
    selected = _round_robin(groups, min(target, len(good)), max_per_bin)

    info = []
    for s, records, n_t850, n_z500 in selected:
        _save(parsed_dir / f"{s.sid}.npz", records)
        info.append(
            {"sid": s.sid, "name": s.name, "lat": s.lat, "lon": s.lon,
             "elev": s.elev, "n_t850": n_t850, "n_z500": n_z500}
        )
    with open(cache_dir / PARSED_DIR / "stations_igra.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["sid", "name", "lat", "lon", "elev", "n_t850", "n_z500"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(info)
    log(f"IGRA2 缓存完成：通过覆盖筛选 {len(good)}，选中 {len(selected)}")
    return info


def load_isd_cache(cache_dir: Path) -> tuple[list[Station], dict[str, dict[str, np.ndarray]]]:
    path = cache_dir / PARSED_DIR / "stations_isd.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    stations = [
        Station(r["sid"], r["name"], float(r["lat"]), float(r["lon"]), float(r["elev"]),
                r.get("country", ""), r.get("state", ""))
        for r in rows
    ]
    data = {
        s.sid: _load(cache_dir / PARSED_DIR / "isd" / f"{s.sid}.npz")
        for s in stations
    }
    return stations, data


def load_igra_cache(cache_dir: Path) -> tuple[list[Station], dict[str, dict[str, np.ndarray]]]:
    path = cache_dir / PARSED_DIR / "stations_igra.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    stations = [
        Station(r["sid"], r["name"], float(r["lat"]), float(r["lon"]), float(r["elev"]))
        for r in rows
    ]
    data = {
        s.sid: _load(cache_dir / PARSED_DIR / "igra" / f"{s.sid}.npz")
        for s in stations
    }
    return stations, data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def self_test(cache_dir: Path) -> None:
    isd_files = sorted((cache_dir / ISD_DIR).glob("*.gz"))
    igra_files = sorted((cache_dir / IGRA_DIR).glob("*.zip"))
    if not isd_files or not igra_files:
        raise SystemExit("缺少测试用 ISD/IGRA2 文件")
    isd = None
    for path in isd_files:
        rec = parse_isd_lite(path)
        if rec is not None and np.isfinite(rec["2t"]).any():
            isd = rec
            break
    assert isd is not None and isd["time"].size > 0
    assert np.all(np.diff(isd["time"]) >= 0)
    assert np.nanmax(isd["2t"]) < 350 and np.nanmin(isd["2t"]) > 150

    igra = None
    for path in igra_files:
        rec = parse_igra_2022(path, None)
        if rec is not None and np.isfinite(rec["t850"]).any() and np.isfinite(rec["z500"]).any():
            igra = rec
            break
    assert igra is not None and igra["time"].size > 0
    assert np.nanmax(igra["t850"]) < 330 and np.nanmin(igra["t850"]) > 180
    assert np.nanmax(igra["z500"]) < 6200 and np.nanmin(igra["z500"]) > 4500
    print("self-test OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--init-times", default=",".join([
        "2022-01-02T00:00:00", "2022-01-02T12:00:00",
        "2022-01-16T00:00:00", "2022-01-16T12:00:00",
        "2022-04-01T00:00:00", "2022-04-01T12:00:00",
        "2022-04-15T00:00:00", "2022-04-15T12:00:00",
        "2022-07-01T00:00:00", "2022-07-01T12:00:00",
        "2022-07-15T00:00:00", "2022-07-15T12:00:00",
        "2022-10-01T00:00:00", "2022-10-01T12:00:00",
        "2022-10-15T00:00:00", "2022-10-15T12:00:00",
    ]))
    parser.add_argument("--lead-hours", type=int, default=240)
    parser.add_argument("--isd-target", type=int, default=600)
    parser.add_argument("--igra-target", type=int, default=200)
    parser.add_argument("--isd-workers", type=int, default=8)
    parser.add_argument("--igra-workers", type=int, default=4)
    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)

    if args.self_test:
        self_test(cache_dir)
        return
    if not args.prepare:
        parser.error("需要 --prepare 或 --self-test")
    init_times = [parse_utc(x.strip()) for x in args.init_times.split(",") if x.strip()]
    isd_info = prepare_isd_cache(
        cache_dir, init_times, args.lead_hours,
        target=args.isd_target, workers=args.isd_workers,
    )
    igra_info = prepare_igra_cache(
        cache_dir, init_times, args.lead_hours,
        target=args.igra_target, workers=args.igra_workers,
    )
    summary = {
        "init_times": [t.isoformat() for t in init_times],
        "lead_hours": args.lead_hours,
        "isd_selected": len(isd_info),
        "igra_selected": len(igra_info),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(cache_dir / PARSED_DIR / "prepare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"完成: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
