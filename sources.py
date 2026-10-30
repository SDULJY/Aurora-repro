"""评估数据源：本地 2026 个例，以及可扩展的 WeatherBench2 多时次数据。

指标变量统一为 2t / msl / 10u / z500 / t850：
- 2t: K
- msl: hPa
- 10u: m/s
- z500: geopotential height m（由 geopotential / g 得到）
- t850: K
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import gcsfs
import numpy as np
import xarray as xr
import zarr
from arraylake import Client
from scipy.interpolate import RegularGridInterpolator

from config import (
    ERA5_REPO_ID,
    ERA5_SNAPSHOT,
    LEVELS,
    forecast_times,
    to_numpy_time,
)
from metrics import G

WB2_ERA5 = "weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr"
WB2_CLIM15 = (
    "weatherbench2/datasets/era5-hourly-climatology/"
    "1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr"
)
WB2_TIME_ORIGIN = np.datetime64("1959-01-01T00:00:00", "ns")
CROP_LAT = 720


@dataclasses.dataclass(frozen=True)
class VarSpec:
    name: str
    kind: str          # "surface" | "pressure"
    model_name: str    # Aurora Batch 变量名
    local_name: str    # ERA5 NetCDF 变量名
    wb2_name: str      # WeatherBench2 数组名
    clim_name: str     # 气候态数组名
    level: int | None = None
    scale: float = 1.0
    units: str = ""


VARIABLES = [
    VarSpec("2t", "surface", "2t", "t2m", "2m_temperature", "2m_temperature", units="K"),
    VarSpec(
        "msl", "surface", "msl", "msl", "mean_sea_level_pressure",
        "mean_sea_level_pressure", scale=1e-2, units="hPa",
    ),
    VarSpec(
        "10u", "surface", "10u", "u10", "10m_u_component_of_wind",
        "10m_u_component_of_wind", units="m/s",
    ),
    VarSpec(
        "z500", "pressure", "z", "z", "geopotential", "geopotential",
        level=500, scale=1.0 / G, units="m",
    ),
    VarSpec("t850", "pressure", "t", "t", "temperature", "temperature", level=850, units="K"),
]
VAR_BY_NAME = {v.name: v for v in VARIABLES}

IC_SURF = {"t2m": "2m_temperature", "u10": "10m_u_component_of_wind",
           "v10": "10m_v_component_of_wind", "msl": "mean_sea_level_pressure"}
IC_PRESS = {"z": "geopotential", "t": "temperature", "u": "u_component_of_wind",
            "v": "v_component_of_wind", "q": "specific_humidity"}


@dataclasses.dataclass
class CaseData:
    init_time: dt.datetime
    lead_hours: int
    valid_times: list[dt.datetime]
    single: xr.Dataset
    press: xr.Dataset
    truth: dict[str, np.ndarray]        # var -> (n_leads, 720, 1440)
    persistence: dict[str, np.ndarray]  # var -> (720, 1440)
    lat: np.ndarray
    lon: np.ndarray


def _time_array(times: list[dt.datetime]) -> np.ndarray:
    return np.array([to_numpy_time(t) for t in times], dtype="datetime64[ns]")


def _make_single(arrays: dict[str, np.ndarray], times, lat, lon) -> xr.Dataset:
    return xr.Dataset(
        {k: (("time", "latitude", "longitude"), v) for k, v in arrays.items()},
        coords={"time": _time_array(times), "latitude": lat, "longitude": lon},
    )


def _make_press(arrays: dict[str, np.ndarray], times, lat, lon) -> xr.Dataset:
    return xr.Dataset(
        {k: (("time", "level", "latitude", "longitude"), v) for k, v in arrays.items()},
        coords={
            "time": _time_array(times),
            "level": LEVELS,
            "latitude": lat,
            "longitude": lon,
        },
    )


def _assemble_case(
    data,
    init_time: dt.datetime,
    lead_hours: int,
    history,
    valid,
    lat,
    lon,
    levels,
) -> CaseData:
    """把 npz 中的 IC/真值数组组装成 CaseData（Earthmover 与 WB2 共用）。"""
    surf = {k: data[f"ic_surf_{k}"] for k in IC_SURF}
    press = {k: data[f"ic_press_{k}"] for k in IC_PRESS}
    single = _make_single(
        {k: surf[k] for k in ["t2m", "u10", "v10", "msl"]}, history, lat, lon
    )
    press_ds = _make_press(
        {k: press[k] for k in ["t", "u", "v", "q", "z"]}, history, lat, lon
    )
    truth = {v.name: data[f"truth_{v.name}"].astype(np.float32) for v in VARIABLES}
    persistence = {
        "2t": surf["t2m"][-1, :CROP_LAT],
        "msl": surf["msl"][-1, :CROP_LAT] * VAR_BY_NAME["msl"].scale,
        "10u": surf["u10"][-1, :CROP_LAT],
        "z500": press["z"][-1, levels.index(500), :CROP_LAT] * VAR_BY_NAME["z500"].scale,
        "t850": press["t"][-1, levels.index(850), :CROP_LAT],
    }
    return CaseData(
        init_time=init_time,
        lead_hours=lead_hours,
        valid_times=valid,
        single=single,
        press=press_ds,
        truth={k: v.astype(np.float32) for k, v in truth.items()},
        persistence={k: v.astype(np.float32) for k, v in persistence.items()},
        lat=lat[:CROP_LAT],
        lon=lon,
    )


class LocalSource:
    """读取仓库里已有的 2026-03-13 个例（surface + pressure 历史和真值缓存）。"""

    def __init__(self, data_dir: str, cache_dir: str):
        self.single_full = xr.open_dataset(os.path.join(data_dir, "era5_single.nc"))
        self.press_full = xr.open_dataset(os.path.join(data_dir, "era5_press.nc"))
        self.truth_cache_path = os.path.join(cache_dir, "local_case_truth.npz")

    def _require(self, ds: xr.Dataset, times, name: str) -> None:
        available = set(ds.time.values.astype("datetime64[ns]").astype(np.int64))
        missing = [str(t) for t in times if int(to_numpy_time(t).astype(np.int64)) not in available]
        if missing:
            raise RuntimeError(f"{name} 缺少时次 {missing}")

    def load_case(self, init_time: dt.datetime, lead_hours: int) -> CaseData:
        history, valid = forecast_times(init_time, lead_hours)
        self._require(self.single_full, history + valid, "era5_single.nc")
        self._require(self.press_full, history, "era5_press.nc")

        single = self.single_full.sel(time=_time_array(history))
        press = self.press_full.sel(time=_time_array(history), level=LEVELS)
        lat = self.single_full.latitude.values[:CROP_LAT]
        lon = self.single_full.longitude.values

        def surface_truth(name: str) -> np.ndarray:
            return np.stack([
                self.single_full[name].sel(time=to_numpy_time(vt)).values[:CROP_LAT]
                for vt in valid
            ])

        truth = {
            "2t": surface_truth("t2m"),
            "msl": surface_truth("msl") * VAR_BY_NAME["msl"].scale,
            "10u": surface_truth("u10"),
        }

        if not os.path.exists(self.truth_cache_path):
            raise RuntimeError(
                f"缺少 {self.truth_cache_path}；请先运行 pressure 真值抓取脚本"
            )
        cache = np.load(self.truth_cache_path)
        cached_times = cache["times"].astype("datetime64[ns]")
        if not np.array_equal(cached_times, _time_array(valid)):
            raise RuntimeError(
                "local_case_truth.npz 的时次与当前 --init-time/--lead-hours 不匹配"
            )
        truth["z500"] = cache["z500"][:, :CROP_LAT].astype(np.float32) * VAR_BY_NAME["z500"].scale
        truth["t850"] = cache["t850"][:, :CROP_LAT].astype(np.float32)

        init_np = to_numpy_time(init_time)
        persistence = {
            "2t": self.single_full["t2m"].sel(time=init_np).values[:CROP_LAT],
            "msl": self.single_full["msl"].sel(time=init_np).values[:CROP_LAT]
            * VAR_BY_NAME["msl"].scale,
            "10u": self.single_full["u10"].sel(time=init_np).values[:CROP_LAT],
            "z500": self.press_full["z"].sel(time=init_np, level=500).values[:CROP_LAT]
            * VAR_BY_NAME["z500"].scale,
            "t850": self.press_full["t"].sel(time=init_np, level=850).values[:CROP_LAT],
        }
        return CaseData(
            init_time=init_time,
            lead_hours=lead_hours,
            valid_times=valid,
            single=single,
            press=press,
            truth={k: v.astype(np.float32) for k, v in truth.items()},
            persistence={k: v.astype(np.float32) for k, v in persistence.items()},
            lat=lat,
            lon=lon,
        )


class WB2Source:
    """WeatherBench2 ERA5 0.25° 6 小时数据，适合多起报时次。

    每读取一个 case 会缓存为 npz，重复运行不会再次下载。
    """

    def __init__(self, cache_dir: str, workers: int = 8):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.workers = workers
        self.fs = gcsfs.GCSFileSystem(token="anon")
        self.g = zarr.open_group(self.fs.get_mapper(WB2_ERA5), mode="r")
        self.time_hours = np.asarray(self.g["time"][:], dtype=np.int64)
        self.lat = np.asarray(self.g["latitude"][:], dtype=np.float64)
        self.lon = np.asarray(self.g["longitude"][:], dtype=np.float64)
        self.levels = [int(x) for x in self.g["level"][:]]

    def index(self, t: dt.datetime) -> int:
        hours = int((to_numpy_time(t).astype("datetime64[h]") - WB2_TIME_ORIGIN.astype("datetime64[h]"))
                    / np.timedelta64(1, "h"))
        pos = int(np.searchsorted(self.time_hours, hours))
        if pos >= len(self.time_hours) or self.time_hours[pos] != hours:
            raise RuntimeError(f"WeatherBench2 数据中没有时次 {t.isoformat()}")
        return pos

    def _cache_path(self, init_time: dt.datetime, lead_hours: int) -> str:
        return os.path.join(
            self.cache_dir,
            f"wb2_{init_time:%Y%m%dT%H%M}_{lead_hours}h.npz",
        )

    def load_case(self, init_time: dt.datetime, lead_hours: int) -> CaseData:
        history, valid = forecast_times(init_time, lead_hours)
        cache_path = self._cache_path(init_time, lead_hours)
        if os.path.exists(cache_path):
            data = np.load(cache_path)
        else:
            data = self._download_case(init_time, lead_hours, history, valid)
            np.savez(cache_path, **data)

        return _assemble_case(
            data, init_time, lead_hours, history, valid, self.lat, self.lon, self.levels
        )

    def _download_case(self, init_time, lead_hours, history, valid) -> dict[str, np.ndarray]:
        hist_pos = [self.index(t) for t in history]
        valid_pos = [self.index(t) for t in valid]
        out: dict[str, np.ndarray] = {}
        print(f"  下载 IC：{len(hist_pos)} 个时次 × 5 层变量 ...", flush=True)
        for key, name in IC_PRESS.items():
            out[f"ic_press_{key}"] = np.stack(
                [np.asarray(self.g[name][p], dtype=np.float32) for p in hist_pos]
            )
        for key, name in IC_SURF.items():
            out[f"ic_surf_{key}"] = np.stack(
                [np.asarray(self.g[name][p], dtype=np.float32) for p in hist_pos]
            )
        print(f"  下载真值：{len(valid_pos)} 个时次 ...", flush=True)
        for spec in VARIABLES:
            if spec.kind == "surface":
                arr = np.stack(
                    [np.asarray(self.g[spec.wb2_name][p], dtype=np.float32)[:CROP_LAT]
                     for p in valid_pos]
                )
            else:
                lev_idx = self.levels.index(spec.level)
                arr = np.stack(
                    [np.asarray(self.g[spec.wb2_name][p], dtype=np.float32)[lev_idx, :CROP_LAT]
                     for p in valid_pos]
                )
            out[f"truth_{spec.name}"] = arr * spec.scale
        return out

    def _truth_cache_path(self, init_time: dt.datetime, lead_hours: int) -> str:
        return os.path.join(
            self.cache_dir,
            f"wb2truth_{init_time:%Y%m%dT%H%M}_{lead_hours}h.npz",
        )

    def load_truth(self, init_time: dt.datetime, lead_hours: int) -> dict[str, np.ndarray]:
        """只读取 ERA5 真值（paper 口径：HRES T0 IC + ERA5 truth）。"""
        path = self._truth_cache_path(init_time, lead_hours)
        if os.path.exists(path):
            data = np.load(path)
        else:
            data = self._download_truth(init_time, lead_hours)
            np.savez(path, **data)
        return {v.name: data[f"truth_{v.name}"].astype(np.float32) for v in VARIABLES}

    def _read_surface_truth(self, name: str, pos: int) -> np.ndarray:
        return np.asarray(self.g[name][pos], dtype=np.float32)[:CROP_LAT]

    def _read_pressure_truth(self, name: str, pos: int) -> np.ndarray:
        lev = 500 if name == "geopotential" else 850
        arr = np.asarray(self.g[name][pos], dtype=np.float32)
        return arr[self.levels.index(lev), :CROP_LAT]

    def _download_truth(self, init_time, lead_hours) -> dict[str, np.ndarray]:
        _, valid = forecast_times(init_time, lead_hours)
        positions = [self.index(t) for t in valid]
        tasks = {}
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for pos in positions:
                for name in ["2m_temperature", "mean_sea_level_pressure",
                             "10m_u_component_of_wind"]:
                    tasks[ex.submit(self._read_surface_truth, name, pos)] = (
                        "surface", name, pos,
                    )
                for name in ["geopotential", "temperature"]:
                    tasks[ex.submit(self._read_pressure_truth, name, pos)] = (
                        "pressure", name, pos,
                    )
            results = {}
            total = len(tasks)
            for done, fut in enumerate(as_completed(tasks), start=1):
                results[tasks[fut]] = fut.result()
                if done % 20 == 0 or done == total:
                    print(f"    WB2 truth {done}/{total} chunks", flush=True)

        out = {}
        out["truth_2t"] = np.stack(
            [results[("surface", "2m_temperature", p)] for p in positions]
        )
        out["truth_msl"] = np.stack(
            [results[("surface", "mean_sea_level_pressure", p)] for p in positions]
        ) * VAR_BY_NAME["msl"].scale
        out["truth_10u"] = np.stack(
            [results[("surface", "10m_u_component_of_wind", p)] for p in positions]
        )
        out["truth_z500"] = np.stack(
            [results[("pressure", "geopotential", p)] for p in positions]
        ) * VAR_BY_NAME["z500"].scale
        out["truth_t850"] = np.stack(
            [results[("pressure", "temperature", p)] for p in positions]
        )
        return out


class CachedCaseSource:
    """直接从 cache/em_*.npz 加载 case，不打开 Earthmover/Arraylake。

    适合 P0 官方对比：缓存里已经有 IC/真值，避免受 Arraylake API 代理问题影响。
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.lat = np.linspace(90.0, -90.0, 721)
        self.lon = np.linspace(0.0, 359.75, 1440)

    def _cache_path(self, init_time: dt.datetime, lead_hours: int) -> str:
        return os.path.join(
            self.cache_dir, f"em_{init_time:%Y%m%dT%H%M}_{lead_hours}h.npz"
        )

    def load_case(self, init_time: dt.datetime, lead_hours: int) -> CaseData:
        path = self._cache_path(init_time, lead_hours)
        if not os.path.exists(path):
            raise FileNotFoundError(f"没有缓存 {path}；请先用 EarthmoverSource 下载")
        history, valid = forecast_times(init_time, lead_hours)
        data = np.load(path)
        return _assemble_case(
            data, init_time, lead_hours, history, valid, self.lat, self.lon, LEVELS
        )


HRES_T0 = "weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
HRES_T0_ORIGIN = np.datetime64("2016-01-01T00:00:00", "ns")


class HREST0Source:
    """HRES T0 0.25° 6h 初始场（Aurora 官方 HRES 示例使用）。

    只提供 IC；每条 pressure chunk 是 13 层整块（约 54 MB），用多线程并发读取。
    """

    def __init__(self, workers: int = 10):
        self.workers = workers
        self.fs = gcsfs.GCSFileSystem(token="anon")
        self.g = zarr.open_group(self.fs.get_mapper(HRES_T0), mode="r")
        self.time_hours = np.asarray(self.g["time"][:], dtype=np.int64)
        self.lat = np.asarray(self.g["latitude"][:], dtype=np.float64)
        self.lon = np.asarray(self.g["longitude"][:], dtype=np.float64)
        self.levels = [int(x) for x in self.g["level"][:]]
        self.flip_lat = bool(self.lat[0] < self.lat[-1])
        if self.flip_lat:
            self.lat = self.lat[::-1]
        self.flip_lon = bool(self.lon[0] > self.lon[-1])
        if self.flip_lon:
            self.lon = self.lon[::-1]

    def index(self, t: dt.datetime) -> int:
        hours = int(
            (np.datetime64(t.replace(tzinfo=None), "h")
             - HRES_T0_ORIGIN.astype("datetime64[h]"))
            / np.timedelta64(1, "h")
        )
        pos = int(np.searchsorted(self.time_hours, hours))
        if pos >= len(self.time_hours) or self.time_hours[pos] != hours:
            raise RuntimeError(f"HRES T0 没有时次 {t.isoformat()}")
        return pos

    def _read(self, name: str, pos: int) -> np.ndarray:
        a = np.asarray(self.g[name][pos], dtype=np.float32)
        if self.flip_lat:
            a = a[..., ::-1, :]
        if self.flip_lon:
            a = a[..., ::-1]
        return a

    def load_ic(self, init_time: dt.datetime):
        history = [
            init_time - dt.timedelta(hours=6),
            init_time,
        ]
        positions = [self.index(t) for t in history]
        futures = {}
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for pos in positions:
                for name in ["geopotential", "temperature", "u_component_of_wind",
                             "v_component_of_wind", "specific_humidity"]:
                    futures[ex.submit(self._read, name, pos)] = ("press", name, pos)
                for name in ["2m_temperature", "10m_u_component_of_wind",
                             "10m_v_component_of_wind", "mean_sea_level_pressure"]:
                    futures[ex.submit(self._read, name, pos)] = ("surf", name, pos)
            results = {}
            total = len(futures)
            for done, fut in enumerate(as_completed(futures), start=1):
                results[futures[fut]] = fut.result()
                if done % 5 == 0 or done == total:
                    print(f"    HRES T0 {done}/{total} chunks", flush=True)

        press_names = {
            "geopotential": "z", "temperature": "t", "u_component_of_wind": "u",
            "v_component_of_wind": "v", "specific_humidity": "q",
        }
        surf_names = {
            "2m_temperature": "t2m", "10m_u_component_of_wind": "u10",
            "10m_v_component_of_wind": "v10", "mean_sea_level_pressure": "msl",
        }
        press_arrays = {}
        for name, key in press_names.items():
            arr = np.stack([results[("press", name, pos)] for pos in positions])
            if self.levels != LEVELS:
                order = [self.levels.index(lev) for lev in LEVELS]
                arr = arr[:, order]
            press_arrays[key] = arr
        surf_arrays = {
            key: np.stack([results[("surf", name, pos)] for pos in positions])
            for name, key in surf_names.items()
        }
        single = _make_single(surf_arrays, history, self.lat, self.lon)
        press = _make_press(press_arrays, history, self.lat, self.lon)
        return single, press


class EarthmoverSource:
    """Earthmover ERA5，按 4 MB/层的 chunk 读取，适合多起报时次。

    走代理时单 chunk 约 0.4-5 s；130 个 IC chunk + 20 个真值 chunk 用多线程下载。
    """

    def __init__(self, cache_dir: str, snapshot: str = ERA5_SNAPSHOT, workers: int = 8):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.workers = workers
        repo = Client().get_repo(ERA5_REPO_ID)
        if snapshot == "main":
            self.session = repo.readonly_session(branch="main")
        else:
            self.session = repo.readonly_session(snapshot_id=snapshot)
        self.single_ds = (
            xr.open_zarr(self.session.store, group="single/spatial", chunks=None)
            .rename({"valid_time": "time"})
            .sortby("latitude", ascending=False)
        )
        self.press_ds = (
            xr.open_zarr(self.session.store, group="pressure/spatial", chunks=None)
            .rename({"valid_time": "time"})
            .sortby("latitude", ascending=False)
        )
        self.lat = self.single_ds.latitude.values
        self.lon = self.single_ds.longitude.values

    def _cache_path(self, init_time: dt.datetime, lead_hours: int) -> str:
        return os.path.join(
            self.cache_dir, f"em_{init_time:%Y%m%dT%H%M}_{lead_hours}h.npz"
        )

    def _read(self, ds, var: str, t: dt.datetime, level=None, attempts: int = 4):
        for i in range(attempts):
            try:
                if level is None:
                    return ds[var].sel(time=to_numpy_time(t)).values.astype(np.float32)
                return ds[var].sel(
                    time=to_numpy_time(t), pressure_level=level
                ).values.astype(np.float32)
            except Exception as exc:  # noqa: BLE001 - 网络错误统一重试
                if i + 1 == attempts:
                    raise
                print(f"  重试 {var}{level or ''} {t:%H:%M}: {type(exc).__name__}", flush=True)
                time.sleep(3)

    def _download_case(self, init_time, lead_hours, history, valid) -> dict[str, np.ndarray]:
        futures = {}
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for t in history:
                for var in ["z", "t", "u", "v", "q"]:
                    for lev in LEVELS:
                        futures[ex.submit(self._read, self.press_ds, var, t, lev)] = (
                            "press", var, lev, t,
                        )
                for var in ["t2m", "u10", "v10", "msl"]:
                    futures[ex.submit(self._read, self.single_ds, var, t)] = ("surf", var, t)
            for vt in valid:
                for var, lev in [("z", 500), ("t", 850)]:
                    futures[ex.submit(self._read, self.press_ds, var, vt, lev)] = (
                        "truth_press", var, lev, vt,
                    )
                for var in ["t2m", "msl", "u10"]:
                    futures[ex.submit(self._read, self.single_ds, var, vt)] = (
                        "truth_surf", var, vt,
                    )

            results = {}
            total = len(futures)
            for done, future in enumerate(as_completed(futures), start=1):
                results[futures[future]] = future.result()
                if done % 20 == 0 or done == total:
                    print(f"    {done}/{total} chunks", flush=True)

        out = {}
        for var in ["z", "t", "u", "v", "q"]:
            out[f"ic_press_{var}"] = np.stack([
                np.stack([results[("press", var, lev, t)] for lev in LEVELS])
                for t in history
            ])
        for var in ["t2m", "u10", "v10", "msl"]:
            out[f"ic_surf_{var}"] = np.stack(
                [results[("surf", var, t)] for t in history]
            )
        out["truth_2t"] = np.stack(
            [results[("truth_surf", "t2m", vt)][:CROP_LAT] for vt in valid]
        )
        out["truth_msl"] = np.stack(
            [results[("truth_surf", "msl", vt)][:CROP_LAT] for vt in valid]
        ) * VAR_BY_NAME["msl"].scale
        out["truth_10u"] = np.stack(
            [results[("truth_surf", "u10", vt)][:CROP_LAT] for vt in valid]
        )
        out["truth_z500"] = np.stack(
            [results[("truth_press", "z", 500, vt)][:CROP_LAT] for vt in valid]
        ) * VAR_BY_NAME["z500"].scale
        out["truth_t850"] = np.stack(
            [results[("truth_press", "t", 850, vt)][:CROP_LAT] for vt in valid]
        )
        return out

    def load_case(self, init_time: dt.datetime, lead_hours: int) -> CaseData:
        history, valid = forecast_times(init_time, lead_hours)
        cache_path = self._cache_path(init_time, lead_hours)
        if os.path.exists(cache_path):
            data = np.load(cache_path)
        else:
            print(f"  下载 IC/真值（{self.workers} workers）...", flush=True)
            data = self._download_case(init_time, lead_hours, history, valid)
            np.savez(cache_path, **data)
            print(f"  缓存: {cache_path}", flush=True)
        return _assemble_case(
            data, init_time, lead_hours, history, valid, self.lat, self.lon, LEVELS
        )


class ClimatologySource:
    """WeatherBench2 1990-2019 6h 气候态（1.5°），插值到模型 0.25° 网格。"""

    def __init__(self, cache_dir: str, workers: int = 5, use_cache: bool = True):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.workers = workers
        self.use_cache = use_cache
        self.fs = gcsfs.GCSFileSystem(token="anon")
        self.g = zarr.open_group(self.fs.get_mapper(WB2_CLIM15), mode="r")
        self.hour = np.asarray(self.g["hour"][:], dtype=np.int64)
        self.dayofyear = np.asarray(self.g["dayofyear"][:], dtype=np.int64)
        self.lat = np.asarray(self.g["latitude"][:], dtype=np.float64)
        self.lon = np.asarray(self.g["longitude"][:], dtype=np.float64)
        self.levels = [int(x) for x in self.g["level"][:]]

    def _cache_path(self, spec: VarSpec, vt: dt.datetime) -> str:
        return os.path.join(
            self.cache_dir,
            f"clim15_{spec.name}_{vt:%Y%m%dT%H}.npy",
        )

    def _upsample(self, field, spec: VarSpec, target_lat, target_lon) -> np.ndarray:
        field = field.T * spec.scale  # (lon, lat) -> (lat, lon)
        lat_order = np.argsort(self.lat)
        lat_asc = self.lat[lat_order]
        field = field[lat_order, :]
        lon_order = np.argsort(self.lon)
        lon_asc = self.lon[lon_order]
        field = field[:, lon_order]
        lon_ext = np.append(lon_asc, 360.0)
        field_ext = np.concatenate([field, field[:, :1]], axis=1)

        interp = RegularGridInterpolator(
            (lat_asc, lon_ext), field_ext, method="linear", bounds_error=False, fill_value=None
        )
        tlon, tlat = np.meshgrid(target_lon, target_lat)
        points = np.stack([tlat.ravel(), tlon.ravel()], axis=-1)
        return interp(points).reshape(target_lat.size, target_lon.size).astype(np.float32)

    def load(self, valid_times, target_lat, target_lon) -> dict[str, np.ndarray]:
        """按变量整块读取所需 dayofyear，避免同一 chunk 被重复下载。"""
        hour_idxs = [int(np.where(self.hour == vt.hour)[0][0]) for vt in valid_times]
        day_idxs = [
            int(np.where(self.dayofyear == vt.timetuple().tm_yday)[0][0])
            for vt in valid_times
        ]
        dmin, dmax = min(day_idxs), max(day_idxs)

        def _load_spec(spec: VarSpec) -> np.ndarray:
            print(f"  气候态 {spec.name} ...", flush=True)
            cached = []
            for vt in valid_times:
                path = self._cache_path(spec, vt)
                ok = self.use_cache and os.path.exists(path)
                cached.append(np.load(path) if ok else None)
            if all(x is not None for x in cached):
                return np.stack(cached)

            if spec.kind == "surface":
                block = np.asarray(
                    self.g[spec.clim_name][:, dmin : dmax + 1], dtype=np.float32
                )
            else:
                lev_idx = self.levels.index(spec.level)
                block = np.asarray(
                    self.g[spec.clim_name][:, dmin : dmax + 1, lev_idx],
                    dtype=np.float32,
                )
            fields = []
            for vt, hour_idx, day_idx, cached_field in zip(
                valid_times, hour_idxs, day_idxs, cached
            ):
                if cached_field is not None:
                    fields.append(cached_field)
                    continue
                field = self._upsample(
                    block[hour_idx, day_idx - dmin], spec, target_lat, target_lon
                )
                if self.use_cache:
                    np.save(self._cache_path(spec, vt), field)
                fields.append(field)
            return np.stack(fields)

        with ThreadPoolExecutor(max_workers=min(self.workers, len(VARIABLES))) as ex:
            arrays = list(ex.map(_load_spec, VARIABLES))
        return {spec.name: arr for spec, arr in zip(VARIABLES, arrays)}
