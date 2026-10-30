"""抓取 Z500/T850 真值，供 evaluate.py 的 local 数据源使用。

只读取 pressure/spatial 中 z@500 和 t@850 的 4 MB chunk，不需要重新下载完整 13 层数据。
输出到 cache/local_case_truth.npz。
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
from arraylake import Client

from config import ERA5_REPO_ID, ERA5_SNAPSHOT, add_common_args, forecast_times, to_numpy_time


def fetch_one(ds, var: str, level: int, vt, attempts: int = 4) -> np.ndarray:
    for i in range(attempts):
        try:
            return ds[var].sel(
                time=to_numpy_time(vt), pressure_level=level
            ).values.astype(np.float32)
        except Exception as exc:  # noqa: BLE001 - 网络错误统一重试
            print(f"  {var}{level} {vt:%H:%M} 失败({i + 1}/{attempts}): "
                  f"{type(exc).__name__}: {exc}", flush=True)
            if i + 1 < attempts:
                time.sleep(5)
    raise RuntimeError(f"抓取 {var}{level} {vt.isoformat()} 失败")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--snapshot", default=ERA5_SNAPSHOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="覆盖已有缓存")
    args = parser.parse_args()

    _, valid = forecast_times(args.init_time, args.lead_hours)
    out_path = os.path.join(args.out, "local_case_truth.npz")
    if os.path.exists(out_path) and not args.force:
        cached = np.load(out_path)
        if np.array_equal(
            cached["times"].astype("datetime64[ns]"),
            np.array([to_numpy_time(t) for t in valid], dtype="datetime64[ns]"),
        ):
            print(f"缓存已存在且时次匹配: {out_path}")
            return
    os.makedirs(args.out, exist_ok=True)

    repo = Client().get_repo(ERA5_REPO_ID)
    if args.snapshot == "main":
        session = repo.readonly_session(branch="main")
    else:
        session = repo.readonly_session(snapshot_id=args.snapshot)
    ds = xr.open_zarr(session.store, group="pressure/spatial", chunks=None)
    ds = ds.rename({"valid_time": "time"})
    print(f"Earthmover 快照: {session.snapshot_id}")

    jobs = [(var, level, vt) for vt in valid for var, level in [("z", 500), ("t", 850)]]
    print(f"抓取 {len(jobs)} 个 chunk（z@500, t@850）...")
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(fetch_one, ds, var, level, vt): (var, level, vt)
            for var, level, vt in jobs
        }
        for future in futures:
            var, level, vt = futures[future]
            arr = future.result()
            if arr.shape != (721, 1440) or not np.isfinite(arr).all():
                raise RuntimeError(f"{var}{level} {vt} 数据异常: {arr.shape}")
            results[(var, level, vt)] = arr
            print(f"  {var}{level} {vt:%Y-%m-%dT%H:%M} OK", flush=True)

    z500 = np.stack([results[("z", 500, vt)] for vt in valid])
    t850 = np.stack([results[("t", 850, vt)] for vt in valid])
    np.savez(
        out_path,
        z500=z500,
        t850=t850,
        times=np.array([to_numpy_time(t) for t in valid], dtype="datetime64[ns]"),
        init_time=np.array(to_numpy_time(args.init_time)),
        lead_hours=args.lead_hours,
        snapshot=np.array(session.snapshot_id),
    )
    print(f"已保存: {out_path} (z500 {z500.shape}, t850 {t850.shape})")


if __name__ == "__main__":
    main()
