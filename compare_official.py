"""P0：把我们的 Aurora 输出与官方 Aurora 2022 数据集在同一初始时刻/时效上对比。

官方 2022 数据集是论文模型的输出（不是 ERA5 真值）。对比场可以判断我们的
IC/预处理/checkpoint 是否与官方一致；同时给出双方相对 ERA5 的 RMSE 作参考。
"""

import argparse
import datetime as dt
import json
import os

import gcsfs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.interpolate import RegularGridInterpolator

from aurora_utils import build_batch, choose_device, load_model, run_rollout
from config import CHECKPOINT_NAME, LEVELS, MODEL_STEP_HOURS, parse_lead_hours, parse_utc
from metrics import G, area_weights, weighted_bias, weighted_rmse
from sources import CachedCaseSource, EarthmoverSource

OFFICIAL_15 = (
    "weatherbench2/datasets/aurora/"
    "2022-240x121_equiangular_with_poles_conservative.zarr"
)
OFFICIAL_025 = "weatherbench2/datasets/aurora/2022-1440x721.zarr"
OFFICIAL_ORIGIN = np.datetime64("2022-01-01T12:00:00", "ns")

# 变量 -> (官方数组名, 官方 level 索引, 尺度, 单位)
# 官方 3 个 level 的顺序是 [500, 700, 850]。
SPECS = {
    "2t": ("2m_temperature", None, 1.0, "K"),
    "msl": ("mean_sea_level_pressure", None, 1e-2, "hPa"),
    "10u": ("10m_u_component_of_wind", None, 1.0, "m/s"),
    "z500": ("geopotential\t", 0, 1.0 / G, "m"),
    "t850": ("temperature", 2, 1.0, "K"),
}


def _key(g, name: str) -> str:
    keys = list(g.array_keys())
    for candidate in (name, name.rstrip("\t"), name + "\t"):
        if candidate in keys:
            return candidate
    raise KeyError(f"官方数据里找不到 {name!r}; keys={keys}")


def _orient(field: np.ndarray, nlat: int, nlon: int) -> np.ndarray:
    if field.shape == (nlat, nlon):
        return field
    if field.shape == (nlon, nlat):
        return field.T
    raise ValueError(f"字段形状 {field.shape} 与 (lat={nlat}, lon={nlon}) 不匹配")


def _time_index(g, init_time: dt.datetime) -> int:
    hours = int(
        (np.datetime64(init_time.replace(tzinfo=None), "h")
         - OFFICIAL_ORIGIN.astype("datetime64[h]"))
        / np.timedelta64(1, "h")
    )
    times = np.asarray(g["time"][:], dtype=np.int64)
    hits = np.where(times == hours)[0]
    if hits.size == 0:
        raise RuntimeError(f"官方数据没有起报时刻 {init_time.isoformat()}")
    return int(hits[0])


def _lead_index(g, lead_hours: int) -> int:
    leads = np.asarray(g["prediction_timedelta"][:], dtype=np.int64)
    hits = np.where(leads == lead_hours)[0]
    if hits.size == 0:
        raise RuntimeError(f"官方数据没有 +{lead_hours}h 时效")
    return int(hits[0])


def _interp_to_grid(field, src_lat, src_lon, dst_lat, dst_lon):
    """把 0.25° 字段插值到目标网格（目标顺序保持 dst 原样）。"""
    lat_order = np.argsort(src_lat)
    lat_asc = np.asarray(src_lat)[lat_order]
    f = field[lat_order, :]
    lon_order = np.argsort(src_lon)
    lon_asc = np.asarray(src_lon)[lon_order]
    f = f[:, lon_order]
    lon_ext = np.append(lon_asc, 360.0)
    f_ext = np.concatenate([f, f[:, :1]], axis=1)
    interp = RegularGridInterpolator(
        (lat_asc, lon_ext), f_ext, method="linear", bounds_error=False, fill_value=None
    )
    tlon, tlat = np.meshgrid(dst_lon, dst_lat)
    pts = np.stack([tlat.ravel(), tlon.ravel()], axis=-1)
    return interp(pts).reshape(len(dst_lat), len(dst_lon))


def extract_fields(pred) -> dict[str, np.ndarray]:
    """从模型输出的 0.25° Batch 中取出 5 个评估变量。"""
    return {
        "2t": pred.surf_vars["2t"].numpy().squeeze(),
        "msl": pred.surf_vars["msl"].numpy().squeeze() * SPECS["msl"][2],
        "10u": pred.surf_vars["10u"].numpy().squeeze(),
        "z500": pred.atmos_vars["z"].numpy().squeeze()[LEVELS.index(500)]
        * SPECS["z500"][2],
        "t850": pred.atmos_vars["t"].numpy().squeeze()[LEVELS.index(850)],
    }


def _metrics(pred, truth, lat):
    w = area_weights(np.asarray(lat, dtype=np.float64))
    corr = float(np.corrcoef(pred.ravel(), truth.ravel())[0, 1])
    return {
        "rmse": weighted_rmse(pred, truth, w),
        "bias": weighted_bias(pred, truth, w),
        "corr": corr,
    }


def _plot(fields, official, lat, lon, path, title):
    variables = list(fields)
    origin = "upper" if lat[0] > lat[-1] else "lower"
    fig, axes = plt.subplots(len(variables), 3, figsize=(15, 3.1 * len(variables)))
    for row, var in enumerate(variables):
        ours = fields[var]
        off = official[var]
        diff = ours - off
        vmax = max(float(np.nanpercentile(np.abs(diff), 99)), 1e-9)
        panels = [
            (ours, f"ours {var}", None),
            (off, f"official {var}", None),
            (diff, f"ours - official", vmax),
        ]
        for col, (data, label, sym) in enumerate(panels):
            ax = axes[row, col]
            if sym is None:
                im = ax.imshow(data, origin=origin, aspect="auto")
            else:
                im = ax.imshow(
                    data, origin=origin, aspect="auto", cmap="RdBu_r", vmin=-sym, vmax=sym
                )
            ax.set_title(label, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-time", type=parse_utc,
                        default=parse_utc("2022-01-08T00:00:00"))
    parser.add_argument("--lead-hours", type=parse_lead_hours, default=24)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--checkpoint", default=CHECKPOINT_NAME)
    parser.add_argument("--checkpoint-path", default=None,
                        help="本地 checkpoint 路径；设置后直接用 load_checkpoint_local")
    parser.add_argument("--use-lora", action="store_true",
                        help="fine-tuned checkpoint 含 LoRA 权重时需要")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--out", default="results/p0_official")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--strict-025", action="store_true",
                        help="额外读取官方 0.25° 2t（41 MB chunk，较慢）做逐格对比")
    args = parser.parse_args()

    if args.lead_hours > 240:
        raise SystemExit("官方数据只到 +240h")
    os.makedirs(args.out, exist_ok=True)
    device = choose_device(args.device)

    cache_path = os.path.join(
        args.cache_dir,
        f"em_{args.init_time:%Y%m%dT%H%M}_{args.lead_hours}h.npz",
    )
    if os.path.exists(cache_path):
        print(f"使用缓存 case: {cache_path}", flush=True)
        source = CachedCaseSource(args.cache_dir)
    else:
        print("加载 Earthmover IC/真值 ...", flush=True)
        source = EarthmoverSource(args.cache_dir, workers=args.workers)
    case = source.load_case(args.init_time, args.lead_hours)
    steps = args.lead_hours // MODEL_STEP_HOURS
    print(f"加载模型 {args.checkpoint_path or args.checkpoint} ...", flush=True)
    model = load_model(
        device,
        checkpoint=args.checkpoint,
        checkpoint_path=args.checkpoint_path,
        use_lora=args.use_lora,
    )
    preds = run_rollout(model, build_batch(case.single, case.press, args.init_time), steps)
    pred = preds[-1]

    # 我们的 0.25° 场
    ours_025 = extract_fields(pred)

    fs = gcsfs.GCSFileSystem(token="anon")
    g15 = zarr.open_group(fs.get_mapper(OFFICIAL_15), mode="r")
    time_idx = _time_index(g15, args.init_time)
    lead_idx = _lead_index(g15, args.lead_hours)
    lat15 = np.asarray(g15["latitude"][:], dtype=np.float64)
    lon15 = np.asarray(g15["longitude"][:], dtype=np.float64)
    levels15 = [int(x) for x in g15["level"][:]]
    print(f"官方 1.5°: time_idx={time_idx}, lead_idx={lead_idx}, "
          f"levels={levels15}", flush=True)

    ours_15, off_15, truth_15 = {}, {}, {}
    for var, (name, lev_idx, scale, _) in SPECS.items():
        key = _key(g15, name)
        if lev_idx is None:
            arr = g15[key][time_idx, lead_idx]
        else:
            arr = g15[key][time_idx, lead_idx, lev_idx]
        off = _orient(np.asarray(arr, dtype=np.float64), len(lat15), len(lon15)) * scale
        off_15[var] = off
        ours_15[var] = _interp_to_grid(
            ours_025[var], case.lat, case.lon, lat15, lon15
        )
        truth_15[var] = _interp_to_grid(
            case.truth[var][-1], case.lat, case.lon, lat15, lon15
        )

    print("\n=== 1.5° 网格：ours vs official / 双方 vs ERA5 ===")
    results_15 = {}
    for var, (_, _, _, units) in SPECS.items():
        a = _metrics(ours_15[var], off_15[var], lat15)
        b = _metrics(ours_15[var], truth_15[var], lat15)
        c = _metrics(off_15[var], truth_15[var], lat15)
        results_15[var] = {
            "units": units,
            "ours_vs_official": a,
            "ours_vs_era5": b,
            "official_vs_era5": c,
        }
        print(f"{var:5s} ours-official RMSE={a['rmse']:.4f} {units:5s} "
              f"corr={a['corr']:.5f} | ours-ERA5={b['rmse']:.4f} | "
              f"official-ERA5={c['rmse']:.4f}")

    # 保存 1.5° 结果和场
    np.savez(
        os.path.join(args.out, "fields.npz"),
        lat15=lat15, lon15=lon15,
        **{f"ours_{k}": v for k, v in ours_15.items()},
        **{f"official_{k}": v for k, v in off_15.items()},
        **{f"truth_{k}": v for k, v in truth_15.items()},
    )
    _plot(
        ours_15, off_15, lat15, lon15,
        os.path.join(args.out, "official_vs_ours_15deg.png"),
        f"Aurora official vs ours, init {args.init_time:%Y-%m-%dT%H:%M}, "
        f"+{args.lead_hours}h (1.5 deg)",
    )

    payload = {
        "init_time": args.init_time.isoformat(),
        "lead_hours": args.lead_hours,
        "checkpoint": args.checkpoint,
        "checkpoint_path": args.checkpoint_path,
        "official_15deg": OFFICIAL_15,
        "official_time_index": time_idx,
        "official_lead_index": lead_idx,
        "at_15deg": results_15,
        "note": "official dataset is model output, not ERA5 truth; "
                "official_vs_era5 is for context",
    }

    # 可选：官方 0.25° 2t 逐格对比
    if args.strict_025:
        print("\n读取官方 0.25° 2t（41 MB chunk，可能需要几分钟）...", flush=True)
        g025 = zarr.open_group(fs.get_mapper(OFFICIAL_025), mode="r")
        t_idx = _time_index(g025, args.init_time)
        l_idx = _lead_index(g025, args.lead_hours)
        off_2t = np.asarray(g025[_key(g025, "2m_temperature")][t_idx, l_idx], dtype=np.float64)
        m = _metrics(ours_025["2t"], off_2t, case.lat)
        payload["at_025deg_2t"] = m
        print(f"0.25° 2t ours-official: RMSE={m['rmse']:.4f} K "
              f"bias={m['bias']:+.4f} corr={m['corr']:.6f}")
        np.savez(
            os.path.join(args.out, "fields_2t_025.npz"),
            ours=ours_025["2t"], official=off_2t,
            lat=case.lat, lon=case.lon,
        )
        _plot(
            {"2t": ours_025["2t"]}, {"2t": off_2t}, case.lat, case.lon,
            os.path.join(args.out, "official_vs_ours_2t_025.png"),
            f"2t official vs ours, init {args.init_time:%Y-%m-%dT%H:%M}, "
            f"+{args.lead_hours}h (0.25 deg), RMSE={m['rmse']:.4f} K",
        )

    with open(os.path.join(args.out, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.out}/comparison.json, fields.npz, "
          f"official_vs_ours_15deg.png")
    if args.strict_025:
        print(f"            {args.out}/fields_2t_025.npz, official_vs_ours_2t_025.png")


if __name__ == "__main__":
    main()
