"""Phase 0 试点：用 ISD-Lite 地面站和 IGRA2 探空检验 Aurora 的观测预报能力。

这是 Phase 0 的自包含脚本，不是最终的 Phase 1 数据管线。它做四件事：

1. 下载并解析 ISD-Lite 2022 地面站（2t/msl/10u）；
2. 下载并解析 IGRA2 2022 探空（t850/z500）；
3. 跑一个 2022-01-02T00 起报的 Aurora（HRES T0 + fine-tuned LoRA）4 步预报；
4. 把模式、ERA5、persistence 匹配到台站，输出 RMSE/bias/MAE 和对照图。

运行示例：

    python pilot_obs.py \
      --init-time 2022-01-02T00:00:00 --lead-hours 24 \
      --checkpoint-path /path/to/aurora-0.25-finetuned.ckpt \
      --isd-stations 60 --igra-stations 10 \
      --out results/obs_pilot
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
import random
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aurora_utils import build_batch, choose_device, load_model, load_static, run_rollout
from config import LEVELS, forecast_times, parse_utc
from sources import HREST0Source

G = 9.80665
ISD_BASE = "https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data/2022"
ISD_HISTORY = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
IGRA_LIST = "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt"
IGRA_BASE = "https://www.ncei.noaa.gov/pub/data/igra/data/data-por"

VARIABLES_SURF = ["2t", "msl", "10u"]
VARIABLES_UPPER = ["t850", "z500"]


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def fetch(url: str, path: Path, timeout: int = 180, retries: int = 3) -> Path | None:
    """下载到 path；已存在则复用。404 返回 None，其他错误重试。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "aurora-obs-pilot/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
            return path
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last_exc = exc
        except Exception as exc:  # noqa: BLE001 - 网络错误统一重试
            last_exc = exc
        time.sleep(2 * (attempt + 1))
    log(f"  下载失败 {url}: {type(last_exc).__name__}: {last_exc}")
    return None


def _bin_key(lat: float, lon: float) -> tuple[int, int]:
    return int((lat + 90.0) // 30.0), int((lon + 180.0) // 60.0)


def _round_robin(groups: dict, target: int, per_bin: int = 6) -> list:
    """按 bin 轮转取样，避免站点集中在欧洲/北美。"""
    selected = []
    for _ in range(per_bin):
        for key in sorted(groups):
            if len(selected) >= target:
                return selected
            if groups[key]:
                selected.append(groups[key].pop(0))
    return selected


# ---------------------------------------------------------------------------
# ISD-Lite 地面站
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Station:
    sid: str
    name: str
    lat: float
    lon: float
    elev: float
    country: str = ""
    state: str = ""
    icao: str = ""
    begin: str = ""
    end: str = ""


@dataclass
class StationCoverage:
    station: Station
    case: dict[str, int] = field(default_factory=lambda: {v: 0 for v in VARIABLES_SURF})
    year: dict[str, int] = field(default_factory=lambda: {v: 0 for v in VARIABLES_SURF})

    @property
    def score(self) -> float:
        case_min = min(self.case.values())
        return case_min * 100000 + sum(self.case.values()) * 1000 + sum(self.year.values())


def load_isd_stations(cache_dir: Path) -> list[Station]:
    path = fetch(ISD_HISTORY, cache_dir / "isd-history.csv", timeout=120)
    if path is None:
        raise RuntimeError("无法下载 ISD 台站表")
    stations: list[Station] = []
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
            begin = row.get("BEGIN", "")
            end = row.get("END", "")
            if not begin or not end or end < "20221231" or begin > "20220101":
                continue
            if row.get("WBAN") == "99999":
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
                    icao=row.get("ICAO", "").strip(),
                    begin=begin,
                    end=end,
                )
            )
    return stations


def choose_candidates(stations: list[Station], target: int, seed: int = 0) -> list[Station]:
    rng = random.Random(seed)
    groups: dict[tuple[int, int], list[Station]] = defaultdict(list)
    for station in stations:
        groups[_bin_key(station.lat, station.lon)].append(station)
    for group in groups.values():
        rng.shuffle(group)
    # 每个 bin 先取最多 3 个，再轮转补足
    selected: list[Station] = []
    seen: set[str] = set()
    for _ in range(3):
        for key in sorted(groups):
            if len(selected) >= target:
                return selected
            if groups[key]:
                station = groups[key].pop(0)
                if station.sid not in seen:
                    selected.append(station)
                    seen.add(station.sid)
    # 补足：从剩余站中轮转
    while len(selected) < target:
        added = False
        for key in sorted(groups):
            if len(selected) >= target:
                break
            if groups[key]:
                station = groups[key].pop(0)
                if station.sid not in seen:
                    selected.append(station)
                    seen.add(station.sid)
                    added = True
        if not added:
            break
    return selected


def parse_isd_lite(path: Path) -> list[dict]:
    """解析 ISD-Lite 2022 年文件，返回逐小时记录。

    ISD-Lite 字段：year month day hour temp dewpoint slp wdir wspd sky p01 p06
    温度/露点单位为 0.1°C，SLP 为 0.1 hPa，风速为 0.1 m/s；-9999 表示缺测。
    """
    records: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 12:
                continue
            try:
                year, month, day, hour = (int(x) for x in parts[:4])
                temp, dew, slp, wdir, wspd, _sky, _p01, _p06 = (
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
            rec: dict = {"time": t}
            if temp != -9999 and -900 <= temp <= 600:
                rec["2t"] = temp / 10.0 + 273.15
            if slp != -9999 and 8700 <= slp <= 10850:
                rec["msl"] = slp / 10.0
            if wspd != -9999 and 0 <= wspd <= 750:
                speed = wspd / 10.0
                if wdir != -9999 and 0 <= wdir <= 360:
                    rad = math.radians(wdir)
                    rec["10u"] = -speed * math.sin(rad)
                    rec["10v"] = -speed * math.cos(rad)
                elif speed == 0:
                    rec["10u"] = 0.0
                    rec["10v"] = 0.0
            records.append(rec)
    return records


def isd_coverage(records: list[dict], case_times: list[datetime]) -> StationCoverage | None:
    by_time = {r["time"]: r for r in records}
    cov = StationCoverage(station=Station("", "", 0, 0, 0))  # placeholder, replaced by caller
    for t in case_times:
        rec = by_time.get(t)
        if rec:
            for var in VARIABLES_SURF:
                if var in rec:
                    cov.case[var] += 1
    for rec in records:
        if rec["time"].hour in (0, 6, 12, 18):
            for var in VARIABLES_SURF:
                if var in rec:
                    cov.year[var] += 1
    return cov


def prepare_isd(
    cache_dir: Path, init: datetime, lead_hours: int, target: int, candidates: int
) -> tuple[list[tuple[Station, list[dict]]], list[dict]]:
    log("加载 ISD 台站表 ...")
    stations = load_isd_stations(cache_dir)
    log(f"  2022 年可用候选站: {len(stations)}")
    chosen = choose_candidates(stations, candidates)
    log(f"  下载 {len(chosen)} 个候选站的 ISD-Lite 文件 ...")

    case_times = [init + timedelta(hours=h) for h in range(6, lead_hours + 1, 6)]

    def work(station: Station):
        url = f"{ISD_BASE}/{station.sid}-2022.gz"
        path = fetch(url, cache_dir / "isd_lite" / "2022" / f"{station.sid}-2022.gz")
        if path is None:
            return None
        records = parse_isd_lite(path)
        if not records:
            return None
        cov = isd_coverage(records, case_times)
        assert cov is not None
        cov.station = station
        return station, records, cov

    results = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(work, s): s for s in chosen}
        for done, fut in enumerate(as_completed(futures), start=1):
            try:
                item = fut.result()
            except Exception as exc:  # noqa: BLE001
                log(f"  解析失败 {futures[fut].sid}: {type(exc).__name__}: {exc}")
                item = None
            if item is not None:
                results.append(item)
            if done % 20 == 0 or done == len(futures):
                log(f"    ISD {done}/{len(futures)}")

    # 至少要有 2 个 case 时次；年度覆盖用于排除长期缺测站
    good = [
        item
        for item in results
        if item[2].case["2t"] >= 2
        and item[2].year["2t"] >= 200
        and item[2].case["msl"] >= 2
        and item[2].year["msl"] >= 200
    ]
    # 优先有 10u 的站，但不强制（部分站只报温度/气压）
    good.sort(key=lambda x: (x[2].case["10u"] > 0, x[2].score), reverse=True)
    groups: dict[tuple[int, int], list] = defaultdict(list)
    for item in good:
        s = item[0]
        groups[_bin_key(s.lat, s.lon)].append(item)
    selected = _round_robin(groups, target, per_bin=6)
    selected_stations = [(s, recs) for s, recs, _ in selected]
    info = [
        {
            "sid": s.sid,
            "name": s.name,
            "lat": s.lat,
            "lon": s.lon,
            "elev": s.elev,
            "country": s.country,
            "state": s.state,
            "case_2t": c.case["2t"],
            "case_msl": c.case["msl"],
            "case_10u": c.case["10u"],
            "year_2t": c.year["2t"],
            "year_msl": c.year["msl"],
            "year_10u": c.year["10u"],
        }
        for s, _, c in selected
    ]
    log(f"  选中 ISD 站: {len(selected)}（候选通过覆盖筛选: {len(good)}）")
    return selected_stations, info


# ---------------------------------------------------------------------------
# IGRA2 探空
# ---------------------------------------------------------------------------


def load_igra_stations(cache_dir: Path) -> list[Station]:
    path = fetch(IGRA_LIST, cache_dir / "igra2-station-list.txt", timeout=120)
    if path is None:
        raise RuntimeError("无法下载 IGRA2 台站表")
    stations: list[Station] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                sid = line[0:11].strip()
                # IGRA2 台站表的纬度/经度/海拔字段含符号位：
                # 纬度 [11:20]，经度 [20:30]，海拔 [30:38]。
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
                Station(
                    sid=sid,
                    name=name,
                    lat=lat,
                    lon=lon,
                    elev=elev,
                    begin=str(first),
                    end=str(last),
                )
            )
    return stations


def choose_igra_candidates(stations: list[Station], target: int) -> list[Station]:
    groups: dict[tuple[int, int], list[Station]] = defaultdict(list)
    for s in stations:
        groups[_bin_key(s.lat, s.lon)].append(s)
    for group in groups.values():
        group.sort(key=lambda s: s.end, reverse=True)
    selected = _round_robin(groups, target, per_bin=3)
    return selected


def parse_igra_2022(path: Path, target_times: set[datetime]) -> dict[datetime, dict]:
    """从 IGRA2 的 data-por zip 中提取指定时次的 850/500 hPa 要素。"""
    out: dict[datetime, dict] = {t: {} for t in target_times}
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            current: datetime | None = None
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
                        current = datetime(year, month, day, hour, tzinfo=timezone.utc)
                    except (ValueError, IndexError):
                        current = None
                    if current is not None and current.year > 2022:
                        break
                    continue
                if current is None or current not in target_times:
                    continue
                try:
                    pressure = int(line[9:15])
                    height = int(line[16:21])
                    temp = int(line[22:27])
                    wdir = int(line[34:39])
                    wspd = int(line[40:45])
                except (ValueError, IndexError):
                    continue
                # IGRA2 数值：气压 Pa、位势高度 m、温度 0.1°C、风速 0.1 m/s
                if abs(pressure - 85000) <= 500 and -1000 <= temp <= 600:
                    out[current]["t850"] = temp / 10.0 + 273.15
                    if wdir != -9999 and wspd != -9999 and 0 <= wdir <= 360:
                        speed = wspd / 10.0
                        rad = math.radians(wdir)
                        out[current]["u850"] = -speed * math.sin(rad)
                        out[current]["v850"] = -speed * math.cos(rad)
                # IGRA2 的位势高度缺测码是 -8888（不是 -9999）。
                if abs(pressure - 50000) <= 500 and height > 0:
                    out[current]["z500"] = float(height)
                    if wdir != -9999 and wspd != -9999 and 0 <= wdir <= 360:
                        speed = wspd / 10.0
                        rad = math.radians(wdir)
                        out[current]["u500"] = -speed * math.sin(rad)
                        out[current]["v500"] = -speed * math.cos(rad)
    return out


def prepare_igra(
    cache_dir: Path, target_times: set[datetime], target: int
) -> tuple[list[tuple[Station, dict[datetime, dict]]], list[dict]]:
    log("加载 IGRA2 台站表 ...")
    stations = load_igra_stations(cache_dir)
    log(f"  2022 年可用探空站: {len(stations)}")
    candidates = choose_igra_candidates(stations, max(target * 2, 12))
    log(f"  下载 {len(candidates)} 个候选站的 IGRA2 数据 ...")

    selected: list[tuple[Station, dict[datetime, dict], int, int]] = []
    # 每批下载 4 个；多收一些候选，再按“两个时次都有 t850/z500”排序，
    # 避免只选到单时次的站。
    for start in range(0, len(candidates), 4):
        batch = candidates[start : start + 4]
        paths: dict[str, Path | None] = {}

        def download_one(s: Station):
            url = f"{IGRA_BASE}/{s.sid}-data.txt.zip"
            return s.sid, fetch(url, cache_dir / "igra2" / f"{s.sid}-data.txt.zip", timeout=300)

        with ThreadPoolExecutor(max_workers=4) as ex:
            for sid, path in ex.map(download_one, batch):
                paths[sid] = path

        for s in batch:
            path = paths.get(s.sid)
            if path is None:
                continue
            try:
                data = parse_igra_2022(path, target_times)
            except Exception as exc:  # noqa: BLE001
                log(f"  解析失败 {s.sid}: {type(exc).__name__}: {exc}")
                continue
            n_t850 = sum(1 for d in data.values() if "t850" in d)
            n_z500 = sum(1 for d in data.values() if "z500" in d)
            if n_t850 + n_z500 == 0:
                continue
            selected.append((s, data, n_t850, n_z500))

    # 优先：两个时次都有 t850 和 z500 > 总有效时次 > 单变量
    selected.sort(
        key=lambda x: (
            x[2] >= len(target_times) and x[3] >= len(target_times),
            x[2] + x[3],
            x[2],
            x[3],
        ),
        reverse=True,
    )
    final = selected[:target]
    info = [
        {
            "sid": s.sid,
            "name": s.name,
            "lat": s.lat,
            "lon": s.lon,
            "elev": s.elev,
            "n_t850": n_t850,
            "n_z500": n_z500,
        }
        for s, _, n_t850, n_z500 in final
    ]
    log(f"  选中 IGRA2 站: {len(final)}（候选有数据: {len(selected)}）")
    return [(s, data) for s, data, _, _ in final], info


# ---------------------------------------------------------------------------
# 模型与 ERA5
# ---------------------------------------------------------------------------


def run_aurora(
    init: datetime, leads: list[int], checkpoint_path: str
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    device = choose_device("auto")
    log(f"加载模型（{device}）...")
    model = load_model(device, checkpoint_path=checkpoint_path, use_lora=True)
    log("加载 HRES T0 初始场 ...")
    # 4 个并发比 10 个更稳：GCS 对大 chunk 的并发范围请求会限速。
    source = HREST0Source(workers=4)
    single, press = source.load_ic(init)
    log(f"跑 {len(leads)} 步 rollout ...")
    preds = run_rollout(model, build_batch(single, press, init), len(leads))

    lat = source.lat[:720]
    lon = source.lon
    fields: dict[int, dict[str, np.ndarray]] = {}
    for lead, pred in zip(leads, preds):
        z = pred.atmos_vars["z"].numpy().squeeze()
        t = pred.atmos_vars["t"].numpy().squeeze()
        fields[lead] = {
            "2t": pred.surf_vars["2t"].numpy().squeeze(),
            "msl": pred.surf_vars["msl"].numpy().squeeze() * 1e-2,
            "10u": pred.surf_vars["10u"].numpy().squeeze(),
            "z500": z[LEVELS.index(500)] / G,
            "t850": t[LEVELS.index(850)],
        }

    persistence = {
        "2t": single["t2m"].values[-1][:720],
        "msl": single["msl"].values[-1][:720] * 1e-2,
        "10u": single["u10"].values[-1][:720],
        "z500": press["z"].values[-1, LEVELS.index(500)][:720] / G,
        "t850": press["t"].values[-1, LEVELS.index(850)][:720],
    }
    static_z = load_static()["z"].numpy()[:720] / G
    return fields, persistence, lat, lon, static_z


def load_era5_truth(init: datetime, leads: list[int], cache_dir: Path) -> dict[int, dict[str, np.ndarray]]:
    """优先使用已有的 240h truth 缓存，避免重新下载。"""
    path = cache_dir / f"wb2truth_{init:%Y%m%dT%H%M}_240h.npz"
    if not path.exists():
        # 回退到 WB2Source；只取需要的时效
        from sources import WB2Source

        log("未找到 240h truth 缓存，使用 WB2Source 下载 ...")
        truth = WB2Source(str(cache_dir), workers=8).load_truth(init, max(leads))
        return {
            lead: {
                "2t": truth["2t"][lead // 6 - 1],
                "msl": truth["msl"][lead // 6 - 1],
                "10u": truth["10u"][lead // 6 - 1],
                "z500": truth["z500"][lead // 6 - 1],
                "t850": truth["t850"][lead // 6 - 1],
            }
            for lead in leads
        }
    data = np.load(path)
    out = {}
    for lead in leads:
        idx = lead // 6 - 1  # 缓存从 +6h 开始
        out[lead] = {
            "2t": data["truth_2t"][idx],
            "msl": data["truth_msl"][idx],
            "10u": data["truth_10u"][idx],
            "z500": data["truth_z500"][idx],
            "t850": data["truth_t850"][idx],
        }
    return out


# ---------------------------------------------------------------------------
# 台站匹配与指标
# ---------------------------------------------------------------------------


def nearest_index(lat: np.ndarray, lon: np.ndarray, slat: float, slon: float) -> tuple[int, int]:
    i = int(np.argmin(np.abs(lat - slat)))
    lon360 = slon % 360.0
    j = int(np.argmin(np.abs(((lon - lon360 + 180.0) % 360.0) - 180.0)))
    return i, j


def metrics(obs: list[float], pred: list[float]) -> dict:
    obs_arr = np.asarray(obs, dtype=np.float64)
    pred_arr = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(obs_arr) & np.isfinite(pred_arr)
    obs_arr, pred_arr = obs_arr[mask], pred_arr[mask]
    n = int(obs_arr.size)
    if n == 0:
        return {"n": 0, "bias": None, "mae": None, "rmse": None, "corr": None}
    diff = pred_arr - obs_arr
    corr = None
    if n >= 3 and np.std(obs_arr) > 0 and np.std(pred_arr) > 0:
        corr = float(np.corrcoef(obs_arr, pred_arr)[0, 1])
    return {
        "n": n,
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "corr": corr,
    }


def collect_rows(
    isd: list[tuple[Station, list[dict]]],
    igra: list[tuple[Station, dict[datetime, dict]]],
    fields: dict[int, dict[str, np.ndarray]],
    persistence: dict[str, np.ndarray],
    era5: dict[int, dict[str, np.ndarray]],
    lat: np.ndarray,
    lon: np.ndarray,
    static_z: np.ndarray,
    init: datetime,
    leads: list[int],
) -> list[dict]:
    rows: list[dict] = []
    # 地面站
    for station, records in isd:
        by_time = {r["time"]: r for r in records}
        i, j = nearest_index(lat, lon, station.lat, station.lon)
        model_elev = float(static_z[i, j])
        dz = model_elev - station.elev
        for lead in leads:
            t = init + timedelta(hours=lead)
            rec = by_time.get(t)
            if not rec:
                continue
            for var in VARIABLES_SURF:
                obs = rec.get(var)
                if obs is None:
                    continue
                model = float(fields[lead][var][i, j])
                era = float(era5[lead][var][i, j])
                per = float(persistence[var][i, j])
                adj = model + 0.0065 * dz if var == "2t" else None
                rows.append(
                    {
                        "source": "ISD",
                        "station": station.sid,
                        "station_name": station.name,
                        "lat": station.lat,
                        "lon": station.lon,
                        "elev": station.elev,
                        "model_elev": model_elev,
                        "dz": dz,
                        "time": t.isoformat(),
                        "lead_hours": lead,
                        "variable": var,
                        "obs": obs,
                        "aurora": model,
                        "aurora_adj": adj,
                        "era5": era,
                        "persistence": per,
                    }
                )
    # 探空
    for station, data in igra:
        i, j = nearest_index(lat, lon, station.lat, station.lon)
        model_elev = float(static_z[i, j])
        dz = model_elev - station.elev
        for lead in leads:
            t = init + timedelta(hours=lead)
            rec = data.get(t)
            if not rec:
                continue
            for var in VARIABLES_UPPER:
                obs = rec.get(var)
                if obs is None:
                    continue
                model = float(fields[lead][var][i, j])
                era = float(era5[lead][var][i, j])
                per = float(persistence[var][i, j])
                rows.append(
                    {
                        "source": "IGRA2",
                        "station": station.sid,
                        "station_name": station.name,
                        "lat": station.lat,
                        "lon": station.lon,
                        "elev": station.elev,
                        "model_elev": model_elev,
                        "dz": dz,
                        "time": t.isoformat(),
                        "lead_hours": lead,
                        "variable": var,
                        "obs": obs,
                        "aurora": model,
                        "aurora_adj": None,
                        "era5": era,
                        "persistence": per,
                    }
                )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["variable"], row["lead_hours"], "aurora")].append(row)
        groups[(row["variable"], row["lead_hours"], "era5")].append(row)
        groups[(row["variable"], row["lead_hours"], "persistence")].append(row)
        if row["aurora_adj"] is not None:
            groups[(row["variable"], row["lead_hours"], "aurora_adj")].append(row)
    out = []
    for (var, lead, method), items in sorted(groups.items()):
        obs = [r["obs"] for r in items]
        pred = [r[method] for r in items]
        m = metrics(obs, pred)
        out.append(
            {
                "variable": var,
                "lead_hours": lead,
                "method": method,
                **m,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(summary: list[dict], rows: list[dict], out: Path) -> None:
    if not rows:
        return
    variables = [v for v in VARIABLES_SURF + VARIABLES_UPPER if any(r["variable"] == v for r in summary)]
    if not variables:
        return
    fig, axes = plt.subplots(1, len(variables), figsize=(4.2 * len(variables), 3.6), squeeze=False)
    axes = axes.ravel()
    for ax, var in zip(axes, variables):
        for method, color in [("aurora", "C0"), ("aurora_adj", "C4"), ("era5", "C1"), ("persistence", "C3")]:
            pts = [
                r
                for r in summary
                if r["variable"] == var and r["method"] == method and r["rmse"] is not None
            ]
            pts.sort(key=lambda r: r["lead_hours"])
            if not pts:
                continue
            ax.plot(
                [r["lead_hours"] for r in pts],
                [r["rmse"] for r in pts],
                marker="o",
                label=method,
                color=color,
            )
        ax.set_title(var)
        ax.set_xlabel("lead (h)")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "rmse_vs_lead.png", dpi=150)
    plt.close(fig)

    # 2t 最大时效散点
    max_lead = max(r["lead_hours"] for r in rows)
    scatter_rows = [r for r in rows if r["variable"] == "2t" and r["lead_hours"] == max_lead]
    if scatter_rows:
        fig, ax = plt.subplots(figsize=(4.5, 4.0))
        ax.scatter([r["obs"] for r in scatter_rows], [r["aurora"] for r in scatter_rows], s=12, alpha=0.7)
        lo = min(min(r["obs"] for r in scatter_rows), min(r["aurora"] for r in scatter_rows))
        hi = max(max(r["obs"] for r in scatter_rows), max(r["aurora"] for r in scatter_rows))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("ISD 2t (K)")
        ax.set_ylabel("Aurora 2t (K)")
        ax.set_title(f"2t at +{max_lead}h")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "scatter_2t.png", dpi=150)
        plt.close(fig)

    # 台站地图
    stations = {(r["station"], r["lat"], r["lon"], r["source"]) for r in rows}
    if stations:
        fig, ax = plt.subplots(figsize=(8, 4.0))
        for source, color in [("ISD", "C0"), ("IGRA2", "C1")]:
            pts = [(lat, lon) for _, lat, lon, src in stations if src == source]
            if pts:
                ax.scatter([p[1] for p in pts], [p[0] for p in pts], s=14, label=source, color=color)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title("Pilot stations")
        fig.tight_layout()
        fig.savefig(out / "stations_map.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-time", type=parse_utc, default=parse_utc("2022-01-02T00:00:00"))
    parser.add_argument("--lead-hours", type=int, default=24, help="试点默认 +24h；必须是 6 的倍数")
    parser.add_argument(
        "--checkpoint-path",
        default=os.environ.get("AURORA_FINETUNED_CKPT", "aurora-0.25-finetuned.ckpt"),
        help="fine-tuned LoRA checkpoint 路径",
    )
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--out", default="results/obs_pilot")
    parser.add_argument("--isd-stations", type=int, default=60)
    parser.add_argument("--igra-stations", type=int, default=10)
    parser.add_argument("--isd-candidates", type=int, default=120)
    args = parser.parse_args()

    if args.lead_hours % 6 != 0 or args.lead_hours < 6:
        raise SystemExit("--lead-hours 必须是 6 的正整数倍")
    if not os.path.exists(args.checkpoint_path):
        raise SystemExit(f"找不到 checkpoint: {args.checkpoint_path}")

    cache_dir = Path(args.cache_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    leads = list(range(6, args.lead_hours + 1, 6))
    init = args.init_time

    log(f"Phase 0 pilot: init={init.isoformat()}, leads={leads}")
    isd, isd_info = prepare_isd(cache_dir, init, args.lead_hours, args.isd_stations, args.isd_candidates)
    target_times = {init + timedelta(hours=h) for h in leads if (init + timedelta(hours=h)).hour in (0, 12)}
    igra, igra_info = prepare_igra(cache_dir, target_times, args.igra_stations)

    fields, persistence, lat, lon, static_z = run_aurora(init, leads, args.checkpoint_path)
    era5 = load_era5_truth(init, leads, cache_dir)

    rows = collect_rows(isd, igra, fields, persistence, era5, lat, lon, static_z, init, leads)
    summary = summarize(rows)

    write_csv(
        out / "per_obs.csv",
        rows,
        [
            "source", "station", "station_name", "lat", "lon", "elev",
            "model_elev", "dz", "time", "lead_hours", "variable",
            "obs", "aurora", "aurora_adj", "era5", "persistence",
        ],
    )
    write_csv(
        out / "summary.csv",
        summary,
        ["variable", "lead_hours", "method", "n", "bias", "mae", "rmse", "corr"],
    )
    write_csv(
        out / "stations_isd.csv",
        isd_info,
        ["sid", "name", "lat", "lon", "elev", "country", "state",
         "case_2t", "case_msl", "case_10u", "year_2t", "year_msl", "year_10u"],
    )
    write_csv(
        out / "stations_igra.csv",
        igra_info,
        ["sid", "name", "lat", "lon", "elev", "n_t850", "n_z500"],
    )

    payload = {
        "phase": "Phase 0 pilot",
        "init_time": init.isoformat(),
        "lead_hours": args.lead_hours,
        "leads": leads,
        "model": {
            "checkpoint_path": os.path.basename(args.checkpoint_path),
            "use_lora": True,
            "ic_source": "HRES T0",
        },
        "truth": "ERA5 (WB2) as reference; ISD/IGRA2 as observations",
        "stations": {
            "isd_selected": len(isd),
            "igra_selected": len(igra),
            "isd_candidates": len(isd_info),
        },
        "summary": summary,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    plot_results(summary, rows, out)
    log("")
    log("=== Pilot summary (RMSE) ===")
    for r in summary:
        if r["rmse"] is None:
            continue
        log(
            f"{r['variable']:5s} +{r['lead_hours']:3d}h {r['method']:12s} "
            f"n={r['n']:4d} RMSE={r['rmse']:.4f} bias={r['bias']:+.4f} "
            f"MAE={r['mae']:.4f} corr={r['corr'] if r['corr'] is None else round(r['corr'], 4)}"
        )
    log("")
    log(f"结果已保存: {out}/")


if __name__ == "__main__":
    main()
