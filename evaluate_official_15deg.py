"""Paper 口径 1.5° 基准：官方 Aurora 2022 预报 vs ERA5 1.5° 真值。

官方 1.5° 2022 数据集是 HRES T0 + fine-tuned 模型输出；P0 已在校验场
（2022-01-08 +24h）确认我们的同配置输出与官方 0.25° 2t 只差 0.064 K。
本脚本用官方 1.5° 数据（小 chunk、稳定）跑 4 季 × 10 天，避免 0.25° 54 MB
压力 chunk 在当前代理下卡住的问题。
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import gcsfs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from compare_official import OFFICIAL_15, _key
from config import forecast_times, parse_lead_hours, parse_utc
from evaluate import plot_summary, summarize, write_csv
from metrics import G, acc, area_weights, weighted_bias, weighted_rmse
from sources import VARIABLES, VAR_BY_NAME, WB2_CLIM15

ERA5_15 = (
    "weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)
ERA5_ORIGIN = np.datetime64("1959-01-01T00:00:00", "ns")
AUR_ORIGIN = np.datetime64("2022-01-01T12:00:00", "ns")


def _time_pos(g, origin, t):
    hours = int(
        (np.datetime64(t.replace(tzinfo=None), "h") - origin.astype("datetime64[h]"))
        / np.timedelta64(1, "h")
    )
    times = np.asarray(g["time"][:], dtype=np.int64)
    hits = np.where(times == hours)[0]
    if hits.size == 0:
        raise RuntimeError(f"{origin} 数据中没有 {t.isoformat()}")
    return int(hits[0])


def _orient(a):
    # 数据布局 (..., longitude, latitude) -> (..., latitude, longitude)
    return np.swapaxes(a, -1, -2)


def _official_fields(g, time_idx):
    out = {}
    out["2t"] = _orient(np.asarray(g[_key(g, "2m_temperature")][time_idx], dtype=np.float64))
    out["msl"] = _orient(
        np.asarray(g[_key(g, "mean_sea_level_pressure")][time_idx], dtype=np.float64)
    ) * VAR_BY_NAME["msl"].scale
    out["10u"] = _orient(
        np.asarray(g[_key(g, "10m_u_component_of_wind")][time_idx], dtype=np.float64)
    )
    z = np.asarray(g[_key(g, "geopotential")][time_idx], dtype=np.float64)
    t = np.asarray(g["temperature"][time_idx], dtype=np.float64)
    out["z500"] = _orient(z[:, 0]) * VAR_BY_NAME["z500"].scale
    out["t850"] = _orient(t[:, 2])
    return out


def _era_block(g, name, pos0, pos1, level_idx=None):
    a = np.asarray(g[name][pos0 : pos1 + 1], dtype=np.float64)
    if level_idx is not None:
        a = a[:, level_idx]
    return _orient(a)


def _clim_block(g, spec, valid_times, dmin, dmax, hour_idxs, day_idxs):
    if spec.kind == "surface":
        block = np.asarray(g[spec.clim_name][:, dmin : dmax + 1], dtype=np.float64)
    else:
        lev_idx = [int(x) for x in g["level"][:]].index(spec.level)
        block = np.asarray(
            g[spec.clim_name][:, dmin : dmax + 1, lev_idx], dtype=np.float64
        )
    fields = []
    for hour_idx, day_idx in zip(hour_idxs, day_idxs):
        field = np.swapaxes(block[hour_idx, day_idx - dmin], -1, -2)
        fields.append(field * spec.scale)
    return np.stack(fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-times", required=True,
                        help="逗号分隔的 2022 起报时刻（UTC）")
    parser.add_argument("--lead-hours", type=parse_lead_hours, default=240)
    parser.add_argument("--out", default="results/paper_official15_4season_10d")
    parser.add_argument("--climatology", choices=["wb2", "none"], default="wb2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--surface-only", action="store_true",
                        help="只评估 2t/msl/10u；跳过需要 12 MB truth chunk 的 z500/t850")
    args = parser.parse_args()
    init_times = [parse_utc(x.strip()) for x in args.init_times.split(",") if x.strip()]
    if not init_times:
        raise SystemExit("--init-times 为空")
    selected = [v for v in VARIABLES if not args.surface_only or v.kind == "surface"]
    if args.lead_hours != 240:
        print("提示：官方 1.5° 数据有 40 个时效，脚本默认按全部 40 个读取")
    os.makedirs(args.out, exist_ok=True)

    fs = gcsfs.GCSFileSystem(token="anon")
    g_aur = zarr.open_group(fs.get_mapper(OFFICIAL_15), mode="r")
    g_era = zarr.open_group(fs.get_mapper(ERA5_15), mode="r")
    g_clim = None
    if args.climatology == "wb2":
        g_clim = zarr.open_group(fs.get_mapper(WB2_CLIM15), mode="r")

    lat = np.asarray(g_era["latitude"][:], dtype=np.float64)
    lon = np.asarray(g_era["longitude"][:], dtype=np.float64)
    if not np.allclose(lat, np.asarray(g_aur["latitude"][:], dtype=np.float64)):
        raise RuntimeError("官方与 ERA5 1.5° 纬度网格不一致")
    if not np.allclose(lon, np.asarray(g_aur["longitude"][:], dtype=np.float64)):
        raise RuntimeError("官方与 ERA5 1.5° 经度网格不一致")
    weights = area_weights(lat)
    leads = [int(x) for x in g_aur["prediction_timedelta"][:]]
    assert leads == list(range(6, 241, 6)), leads[:3]
    n_leads = len(leads)

    if g_clim is not None:
        clim_hour = np.asarray(g_clim["hour"][:], dtype=np.int64)
        clim_day = np.asarray(g_clim["dayofyear"][:], dtype=np.int64)
        clim_lat = np.asarray(g_clim["latitude"][:], dtype=np.float64)
        clim_lon = np.asarray(g_clim["longitude"][:], dtype=np.float64)
        if not (np.allclose(clim_lat, lat) and np.allclose(clim_lon, lon)):
            raise RuntimeError("气候态与 ERA5 1.5° 网格不一致")

    all_rows = []
    for ci, init in enumerate(init_times, start=1):
        print(f"[{ci}/{len(init_times)}] {init.isoformat()}", flush=True)
        _, valid = forecast_times(init, args.lead_hours)
        assert len(valid) == n_leads
        aur_idx = _time_pos(g_aur, AUR_ORIGIN, init)
        era_pos = [_time_pos(g_era, ERA5_ORIGIN, vt) for vt in valid]
        init_pos = _time_pos(g_era, ERA5_ORIGIN, init)

        aur = _official_fields(g_aur, aur_idx)
        p0, p1 = min(era_pos), max(era_pos)
        sel = [p - p0 for p in era_pos]
        truth = {
            "2t": _era_block(g_era, "2m_temperature", p0, p1)[sel],
            "msl": _era_block(g_era, "mean_sea_level_pressure", p0, p1)[sel]
            * VAR_BY_NAME["msl"].scale,
            "10u": _era_block(g_era, "10m_u_component_of_wind", p0, p1)[sel],
        }
        persistence = {
            "2t": _orient(np.asarray(g_era["2m_temperature"][init_pos], dtype=np.float64)),
            "msl": _orient(np.asarray(g_era["mean_sea_level_pressure"][init_pos],
                                      dtype=np.float64)) * VAR_BY_NAME["msl"].scale,
            "10u": _orient(np.asarray(g_era["10m_u_component_of_wind"][init_pos],
                                      dtype=np.float64)),
        }
        if not args.surface_only:
            truth["z500"] = _era_block(
                g_era, "geopotential", p0, p1,
                [int(x) for x in g_era["level"][:]].index(500),
            )[sel] * VAR_BY_NAME["z500"].scale
            truth["t850"] = _era_block(
                g_era, "temperature", p0, p1,
                [int(x) for x in g_era["level"][:]].index(850),
            )[sel]
            persistence["z500"] = _orient(np.asarray(
                g_era["geopotential"][init_pos,
                                      [int(x) for x in g_era["level"][:]].index(500)],
                dtype=np.float64)) * VAR_BY_NAME["z500"].scale
            persistence["t850"] = _orient(np.asarray(
                g_era["temperature"][init_pos,
                                     [int(x) for x in g_era["level"][:]].index(850)],
                dtype=np.float64))
        climatology = None
        if g_clim is not None:
            hour_idxs = [int(np.where(clim_hour == vt.hour)[0][0]) for vt in valid]
            day_idxs = [int(np.where(clim_day == vt.timetuple().tm_yday)[0][0])
                        for vt in valid]
            dmin, dmax = min(day_idxs), max(day_idxs)
            climatology = {
                spec.name: _clim_block(g_clim, spec, valid, dmin, dmax,
                                       hour_idxs, day_idxs)
                for spec in selected
            }

        for i, vt in enumerate(valid):
            lead = int((vt - init).total_seconds() // 3600)
            for spec in selected:
                truth_i = truth[spec.name][i]
                fields = {"aurora_official": aur[spec.name][i],
                          "persistence": persistence[spec.name]}
                if climatology is not None:
                    fields["climatology"] = climatology[spec.name][i]
                for method, field in fields.items():
                    row = {
                        "case": init.isoformat(),
                        "valid_time": vt.isoformat(),
                        "lead_hours": lead,
                        "variable": spec.name,
                        "method": method,
                        "units": spec.units,
                        "rmse": weighted_rmse(field, truth_i, weights),
                        "bias": weighted_bias(field, truth_i, weights),
                        "acc": None,
                    }
                    if method != "climatology" and climatology is not None:
                        row["acc"] = acc(field, truth_i, climatology[spec.name][i], weights)
                    all_rows.append(row)

    summary = summarize(all_rows)
    write_csv(os.path.join(args.out, "per_case.csv"), all_rows,
              ["case", "valid_time", "lead_hours", "variable", "method", "units",
               "rmse", "bias", "acc"])
    write_csv(os.path.join(args.out, "summary.csv"), summary,
              ["variable", "lead_hours", "method", "n_cases", "units",
               "rmse_mean", "rmse_std", "bias_mean", "bias_std", "acc_mean", "acc_std"])
    payload = {
        "source": "official_aurora_2022_1.5deg",
        "init_times": [t.isoformat() for t in init_times],
        "lead_hours": args.lead_hours,
        "variables": {v.name: v.units for v in selected},
        "climatology": args.climatology,
        "official_dataset": OFFICIAL_15,
        "era5_truth": ERA5_15,
        "note": "official 1.5° dataset is HRES T0 + fine-tuned model output; "
                "P0 matched our same-config 0.25° output to 0.064 K (2t)",
        "summary": summary,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {args.out}/summary.csv, per_case.csv, summary.json")
    plot_summary(summary, args.out)


if __name__ == "__main__":
    main()
