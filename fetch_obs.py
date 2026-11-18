"""Phase 1：批量下载 ISD-Lite 地面站和 IGRA2 探空原始数据。

设计目标：

- 只依赖标准库，不需要 GPU/模型；
- 直接访问 NOAA（忽略环境里的 http_proxy，避免依赖本机代理）；
- 断点续传：已存在且校验通过的文件直接跳过；
- 单个文件失败不终止整体任务，最后输出 manifest 和 summary；
- 适合用 setsid + nohup 在服务器上过夜运行。

示例：

    # 下载 2022 年全部可用 ISD 站 + 300 个全球分布的 IGRA2 站
    python -u fetch_obs.py --cache-dir cache --isd-limit 0 --igra-limit 300

    # 只下载 IGRA2
    python -u fetch_obs.py --source igra2 --igra-limit 100
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ISD_BASE = "https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data/{year}"
ISD_HISTORY = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
IGRA_LIST = "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt"
IGRA_BASE = "https://www.ncei.noaa.gov/pub/data/igra/data/data-por"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


@dataclass(frozen=True)
class Station:
    sid: str
    name: str
    lat: float
    lon: float
    elev: float
    begin: str = ""
    end: str = ""


def _bin_key(lat: float, lon: float) -> tuple[int, int]:
    return int((lat + 90.0) // 30.0), int((lon + 180.0) // 60.0)


def _round_robin(groups: dict, target: int, per_bin: int = 8) -> list:
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


def load_isd_stations(path: Path) -> list[Station]:
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
                    begin=begin,
                    end=end,
                )
            )
    return stations


def load_igra_stations(path: Path) -> list[Station]:
    stations: list[Station] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                sid = line[0:11].strip()
                # IGRA2 台站表的纬度/经度/海拔字段含符号位。
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


def select_global(stations: list[Station], limit: int, seed: int = 0) -> list[Station]:
    if limit <= 0 or limit >= len(stations):
        return stations
    rng = random.Random(seed)
    groups: dict[tuple[int, int], list[Station]] = defaultdict(list)
    for s in stations:
        groups[_bin_key(s.lat, s.lon)].append(s)
    for group in groups.values():
        rng.shuffle(group)
    return _round_robin(groups, limit, per_bin=8)


# ---------------------------------------------------------------------------
# 下载与校验
# ---------------------------------------------------------------------------


def _verify(path: Path, kind: str) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        if kind == "gzip":
            with gzip.open(path, "rb") as f:
                f.read(1)
        elif kind == "zip":
            with zipfile.ZipFile(path) as z:
                if not z.namelist():
                    return False
        return True
    except Exception:  # noqa: BLE001 - 损坏文件按失败处理
        return False


def download(
    url: str,
    path: Path,
    kind: str,
    timeout: int,
    retries: int,
    opener: urllib.request.OpenerDirector,
) -> tuple[str, int, str]:
    """返回 (status, bytes, error)。status: cached/ok/failed/404。"""
    if path.exists() and _verify(path, kind):
        return "cached", path.stat().st_size, ""
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    last_error = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "aurora-obs-fetch/1.0"}
            )
            with opener.open(req, timeout=timeout) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, length=1 << 20)
            if not _verify(tmp, kind):
                raise RuntimeError("文件校验失败")
            os.replace(tmp, path)
            return "ok", path.stat().st_size, ""
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code == 404:
                return "404", 0, last_error
        except Exception as exc:  # noqa: BLE001 - 网络错误统一重试
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < retries:
            time.sleep(2 * (attempt + 1))
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    return "failed", 0, last_error


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--source", choices=["all", "isd", "igra2"], default="all")
    parser.add_argument("--isd-limit", type=int, default=0, help="0 表示全部可用站")
    parser.add_argument("--igra-limit", type=int, default=300, help="0 表示全部可用站")
    parser.add_argument("--isd-workers", type=int, default=8)
    parser.add_argument("--igra-workers", type=int, default=4)
    parser.add_argument("--isd-timeout", type=int, default=60)
    parser.add_argument("--igra-timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--manifest-dir", default=None,
                        help="manifest 输出目录，默认 <cache-dir>/obs")
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else cache / "obs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    # 忽略环境中的代理，保证服务器直连；本机代理关机不影响本脚本。
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    isd_history = cache / "isd-history.csv"
    igra_list = cache / "igra2-station-list.txt"
    isd_dir = cache / "isd_lite" / str(args.year)
    igra_dir = cache / "igra2"
    isd_dir.mkdir(parents=True, exist_ok=True)
    igra_dir.mkdir(parents=True, exist_ok=True)

    log(f"cache={cache.resolve()} source={args.source} year={args.year}")
    log("检查台站元数据 ...")
    if args.source in ("all", "isd"):
        if not _verify(isd_history, "csv"):
            status, nbytes, err = download(
                ISD_HISTORY, isd_history, "csv", 120, args.retries, opener
            )
            log(f"ISD 台站表: {status} {nbytes} bytes {err}")
        if not _verify(isd_history, "csv"):
            raise SystemExit("ISD 台站表下载失败")
    if args.source in ("all", "igra2"):
        if not _verify(igra_list, "txt"):
            status, nbytes, err = download(
                IGRA_LIST, igra_list, "txt", 120, args.retries, opener
            )
            log(f"IGRA2 台站表: {status} {nbytes} bytes {err}")
        if not _verify(igra_list, "txt"):
            raise SystemExit("IGRA2 台站表下载失败")

    manifest: list[dict] = []
    total_ok = total_cached = total_failed = total_404 = 0
    total_bytes = 0

    if args.source in ("all", "isd"):
        stations = load_isd_stations(isd_history)
        selected = select_global(stations, args.isd_limit)
        log(f"ISD: 可用 {len(stations)} 站，本次下载 {len(selected)} 站")
        tasks = []
        for s in selected:
            url = f"{ISD_BASE.format(year=args.year)}/{s.sid}-{args.year}.gz"
            path = isd_dir / f"{s.sid}-{args.year}.gz"
            tasks.append((s, url, path))
        with ThreadPoolExecutor(max_workers=args.isd_workers) as ex:
            futures = {
                ex.submit(
                    download, url, path, "gzip", args.isd_timeout, args.retries, opener
                ): (s, url, path)
                for s, url, path in tasks
            }
            for done, fut in enumerate(as_completed(futures), start=1):
                s, url, path = futures[fut]
                try:
                    status, nbytes, err = fut.result()
                except Exception as exc:  # noqa: BLE001
                    status, nbytes, err = "failed", 0, f"{type(exc).__name__}: {exc}"
                if status == "ok":
                    total_ok += 1
                elif status == "cached":
                    total_cached += 1
                elif status == "404":
                    total_404 += 1
                else:
                    total_failed += 1
                total_bytes += nbytes
                manifest.append(
                    {
                        "source": "ISD",
                        "sid": s.sid,
                        "name": s.name,
                        "lat": s.lat,
                        "lon": s.lon,
                        "elev": s.elev,
                        "url": url,
                        "path": str(path),
                        "status": status,
                        "bytes": nbytes,
                        "error": err,
                    }
                )
                if done % 100 == 0 or done == len(futures):
                    log(
                        f"  ISD {done}/{len(futures)} "
                        f"ok={total_ok} cached={total_cached} failed={total_failed} 404={total_404}"
                    )

    if args.source in ("all", "igra2"):
        stations = load_igra_stations(igra_list)
        selected = select_global(stations, args.igra_limit)
        log(f"IGRA2: 可用 {len(stations)} 站，本次下载 {len(selected)} 站")
        tasks = []
        for s in selected:
            url = f"{IGRA_BASE}/{s.sid}-data.txt.zip"
            path = igra_dir / f"{s.sid}-data.txt.zip"
            tasks.append((s, url, path))
        with ThreadPoolExecutor(max_workers=args.igra_workers) as ex:
            futures = {
                ex.submit(
                    download, url, path, "zip", args.igra_timeout, args.retries, opener
                ): (s, url, path)
                for s, url, path in tasks
            }
            for done, fut in enumerate(as_completed(futures), start=1):
                s, url, path = futures[fut]
                try:
                    status, nbytes, err = fut.result()
                except Exception as exc:  # noqa: BLE001
                    status, nbytes, err = "failed", 0, f"{type(exc).__name__}: {exc}"
                if status == "ok":
                    total_ok += 1
                elif status == "cached":
                    total_cached += 1
                elif status == "404":
                    total_404 += 1
                else:
                    total_failed += 1
                total_bytes += nbytes
                manifest.append(
                    {
                        "source": "IGRA2",
                        "sid": s.sid,
                        "name": s.name,
                        "lat": s.lat,
                        "lon": s.lon,
                        "elev": s.elev,
                        "url": url,
                        "path": str(path),
                        "status": status,
                        "bytes": nbytes,
                        "error": err,
                    }
                )
                if done % 50 == 0 or done == len(futures):
                    log(
                        f"  IGRA2 {done}/{len(futures)} "
                        f"ok={total_ok} cached={total_cached} failed={total_failed} 404={total_404}"
                    )

    manifest_path = manifest_dir / f"manifest_{args.year}.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "source", "sid", "name", "lat", "lon", "elev",
            "url", "path", "status", "bytes", "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)

    summary = {
        "year": args.year,
        "source": args.source,
        "isd_limit": args.isd_limit,
        "igra_limit": args.igra_limit,
        "files": len(manifest),
        "ok": total_ok,
        "cached": total_cached,
        "failed": total_failed,
        "not_found": total_404,
        "bytes": total_bytes,
        "gb": round(total_bytes / 1024**3, 3),
        "manifest": str(manifest_path),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    summary_path = manifest_dir / f"fetch_summary_{args.year}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"完成: {json.dumps(summary, ensure_ascii=False)}")
    log(f"manifest: {manifest_path}")
    log(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
