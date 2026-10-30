"""下载 ERA5 初始条件与验证真值，缓存为本地 NetCDF。联网只需成功这一次。"""

import argparse
import datetime as dt
import os
import time

import numpy as np
import xarray as xr
from arraylake import Client

from config import (
    ERA5_REPO_ID,
    ERA5_SNAPSHOT,
    LEVELS,
    PRESS_VARS,
    SINGLE_VARS,
    add_common_args,
    forecast_times,
    to_numpy_time,
)


def open_era5(snapshot: str):
    """打开 Earthmover 公共 ERA5 仓库，返回 (地面数据, 高空数据, 实际快照 ID)。"""
    repo = Client().get_repo(ERA5_REPO_ID)
    if snapshot == "main":
        session = repo.readonly_session(branch="main")
    else:
        session = repo.readonly_session(snapshot_id=snapshot)

    # store 会持有 session，函数返回后仍可继续读取。
    single = xr.open_zarr(session.store, group="single/spatial", chunks=None)
    press = xr.open_zarr(session.store, group="pressure/spatial", chunks=None)

    # 该仓库用 valid_time / pressure_level 命名，统一改名并按 Aurora 要求降序纬度(90 -> -90)
    single = single.rename({"valid_time": "time"}).sortby("latitude", ascending=False)
    press = press.rename({"valid_time": "time"}).sortby("latitude", ascending=False)
    if "pressure_level" in press.dims:
        press = press.rename({"pressure_level": "level"})
    return single, press, session.snapshot_id


def fetch(ds, attempts: int = 6):
    """读取并落内存，网络超时自动重试。"""
    for i in range(attempts):
        try:
            return ds.compute()
        except Exception as exc:  # noqa: BLE001 - 网络错误类型很多，统一重试
            print(f"读取失败({i + 1}/{attempts}): {type(exc).__name__}: {exc}，10 秒后重试...")
            if i + 1 < attempts:
                time.sleep(10)
    raise RuntimeError("多次重试后仍失败：换个时段再跑，或检查网络/Arraylake 凭据")


def validate(ds: xr.Dataset, times: list[dt.datetime], name: str) -> None:
    """检查时次、纬经度方向是否符合 Aurora 的网格约定。"""
    if "time" not in ds.coords:
        raise RuntimeError(f"{name} 缺少 time 坐标")

    got = ds.time.values.astype("datetime64[ns]").astype(np.int64)
    expected = [to_numpy_time(t).astype(np.int64) for t in times]
    missing = [str(t) for t, e in zip(times, expected) if e not in set(got)]
    if missing:
        raise RuntimeError(f"{name} 缺少时次: {missing}")

    lat = ds.latitude.values
    lon = ds.longitude.values
    if not (lat[0] == 90 and lat[-1] == -90 and np.all(np.diff(lat) < 0)):
        raise RuntimeError(f"{name} 纬度必须是从 90 到 -90 的降序网格")
    if not (lon[0] == 0 and lon[-1] < 360 and np.all(np.diff(lon) > 0)):
        raise RuntimeError(f"{name} 经度必须是 [0, 360) 的升序网格")

    for var in ds.data_vars:
        if bool(ds[var].isnull().all()):
            raise RuntimeError(f"{name} 变量 {var} 全为 NaN")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--snapshot",
        default=ERA5_SNAPSHOT,
        help=f"Earthmover ERA5 快照 ID，默认 {ERA5_SNAPSHOT}；传 main 使用最新分支",
    )
    args = parser.parse_args()

    history, valid = forecast_times(args.init_time, args.lead_hours)
    times = history + valid
    os.makedirs(args.out, exist_ok=True)

    print(f"起报 {args.init_time.isoformat()}；历史帧 {len(history)} 个，验证帧 {len(valid)} 个")
    single, press, snapshot_id = open_era5(args.snapshot)
    print(f"Earthmover 快照: {snapshot_id}")

    print(f"下载地面变量（{len(times)} 个时次）...")
    s = fetch(single[SINGLE_VARS].sel(time=[to_numpy_time(t) for t in times]))
    print(
        f"下载高空变量（{len(history)} 个时次 × {len(LEVELS)} 层 × "
        f"{len(PRESS_VARS)} 变量）..."
    )
    p = fetch(
        press[PRESS_VARS].sel(
            time=[to_numpy_time(t) for t in history],
            level=LEVELS,
        )
    )

    validate(s, times, "地面数据")
    validate(p, history, "高空数据")
    if list(p.level.values) != LEVELS:
        raise RuntimeError(f"气压层不匹配: {p.level.values.tolist()} != {LEVELS}")

    stamp = {
        "source": ERA5_REPO_ID,
        "source_snapshot": snapshot_id,
        "init_time": args.init_time.isoformat(),
        "lead_hours": args.lead_hours,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    for ds in (s, p):
        ds.attrs.update(stamp)

    single_path = os.path.join(args.out, "era5_single.nc")
    press_path = os.path.join(args.out, "era5_press.nc")
    s.to_netcdf(single_path)
    p.to_netcdf(press_path)
    print(f"缓存完成: {single_path} / {press_path}")


if __name__ == "__main__":
    main()
