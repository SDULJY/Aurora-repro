"""下载 ERA5 初始条件与验证真值，缓存为本地 NetCDF。联网只需成功这一次。"""
import time

import xarray as xr
from arraylake import Client

# ---- 参数（换日期就改这里）----
TIMES = ["2026-03-12T18:00:00",   # 起报前 6h（历史帧）
         "2026-03-13T00:00:00",   # 起报时刻
         "2026-03-13T06:00:00",   # 验证: +6h
         "2026-03-13T12:00:00",   # 验证: +12h
         "2026-03-13T18:00:00",   # 验证: +18h
         "2026-03-14T00:00:00"]   # 验证: +24h
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SINGLE_VARS = ["t2m", "u10", "v10", "msl"]   # Aurora 需要的 4 个地面变量
PRESS_VARS = ["t", "u", "v", "q", "z"]       # Aurora 需要的 5 个高空变量

# ---- 连接 Earthmover 公共 ERA5 仓库 ----
rsession = Client().get_repo("earthmover-public/era5").readonly_session("main")
single = xr.open_zarr(rsession.store, group="single/spatial", chunks=None)
press = xr.open_zarr(rsession.store, group="pressure/spatial", chunks=None)

# 该仓库用 valid_time / pressure_level 命名，统一改名并按 Aurora 要求降序纬度(90 -> -90)
single = single.rename({"valid_time": "time"}).sortby("latitude", ascending=False)
press = press.rename({"valid_time": "time"}).sortby("latitude", ascending=False)
if "pressure_level" in press.dims:
    press = press.rename({"pressure_level": "level"})

def fetch(ds, attempts=6):
    """读取并落内存，网络超时自动重试。"""
    for i in range(attempts):
        try:
            return ds.compute()
        except Exception as e:
            print(f"读取失败({i + 1}/{attempts}): {type(e).__name__}，10 秒后重试...")
            time.sleep(10)
    raise RuntimeError("多次重试后仍失败：换个时段再跑，或检查代理设置")

print("下载地面变量（3 个时次）...")
s = fetch(single[SINGLE_VARS].sel(time=TIMES))
print("下载高空变量（2 个时次 × 13 层 × 5 变量）...")
p = fetch(press[PRESS_VARS].sel(time=TIMES[:2], level=LEVELS))

s.to_netcdf("era5_single.nc")
p.to_netcdf("era5_press.nc")
print("缓存完成: era5_single.nc / era5_press.nc")
