"""P0 同配置对比：HRES T0 IC + fine-tuned checkpoint vs 官方 Aurora 2022 输出。

官方 HRES T0 示例使用 HRES T0 初始场和 aurora-0.25-finetuned.ckpt。
本脚本用同一配置跑一个 2022 个例，并与官方 1.5° 数据集做场对比。
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

from aurora_utils import build_batch, choose_device, load_model, run_rollout
from compare_official import (
    OFFICIAL_15,
    OFFICIAL_025,
    SPECS,
    _key,
    _lead_index,
    _metrics,
    _orient,
    _plot,
    _time_index,
    extract_fields,
)
from config import CHECKPOINT_NAME, MODEL_STEP_HOURS, forecast_times, parse_lead_hours, parse_utc
from sources import CachedCaseSource, HREST0Source


def _interp(field, src_lat, src_lon, dst_lat, dst_lon):
    from compare_official import _interp_to_grid
    return _interp_to_grid(field, src_lat, src_lon, dst_lat, dst_lon)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-time", type=parse_utc,
                        default=parse_utc("2022-01-08T00:00:00"))
    parser.add_argument("--lead-hours", type=parse_lead_hours, default=24)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--checkpoint", default="aurora-0.25-finetuned.ckpt")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--use-lora", action="store_true",
                        help="fine-tuned checkpoint 含 LoRA 权重时需要")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--out", default="results/p0_official_matched")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--no-truth", action="store_true",
                        help="不加载缓存 ERA5 真值（只做 ours-vs-official）")
    parser.add_argument("--strict-025-2t", action="store_true",
                        help="额外读取官方 0.25° 2t（41 MB chunk）做逐格对比")
    args = parser.parse_args()

    if args.lead_hours > 240:
        raise SystemExit("官方数据只到 +240h")
    os.makedirs(args.out, exist_ok=True)
    device = choose_device(args.device)
    history, valid = forecast_times(args.init_time, args.lead_hours)
    steps = args.lead_hours // MODEL_STEP_HOURS

    print("读取 HRES T0 IC ...", flush=True)
    ic_source = HREST0Source(workers=args.workers)
    single, press = ic_source.load_ic(args.init_time)
    print(f"模型 {args.checkpoint_path or args.checkpoint} ...", flush=True)
    model = load_model(
        device,
        checkpoint=args.checkpoint,
        checkpoint_path=args.checkpoint_path,
        use_lora=args.use_lora,
    )
    preds = run_rollout(model, build_batch(single, press, args.init_time), steps)
    ours_025 = extract_fields(preds[-1])

    truth_case = None
    if not args.no_truth:
        cache_path = os.path.join(
            args.cache_dir, f"em_{args.init_time:%Y%m%dT%H%M}_{args.lead_hours}h.npz"
        )
        if os.path.exists(cache_path):
            print(f"加载缓存 ERA5 真值: {cache_path}", flush=True)
            truth_case = CachedCaseSource(args.cache_dir).load_case(
                args.init_time, args.lead_hours
            )
        else:
            print(f"没有缓存真值 {cache_path}，跳过 ERA5 对比", flush=True)

    fs = gcsfs.GCSFileSystem(token="anon")
    g15 = zarr.open_group(fs.get_mapper(OFFICIAL_15), mode="r")
    time_idx = _time_index(g15, args.init_time)
    lead_idx = _lead_index(g15, args.lead_hours)
    lat15 = np.asarray(g15["latitude"][:], dtype=np.float64)
    lon15 = np.asarray(g15["longitude"][:], dtype=np.float64)
    print(f"官方 1.5°: time_idx={time_idx}, lead_idx={lead_idx}", flush=True)

    src_lat = ic_source.lat[:720]
    src_lon = ic_source.lon
    ours_15, off_15, truth_15 = {}, {}, {}
    for var, (name, lev_idx, scale, _) in SPECS.items():
        key = _key(g15, name)
        arr = g15[key][time_idx, lead_idx] if lev_idx is None else \
            g15[key][time_idx, lead_idx, lev_idx]
        off = _orient(np.asarray(arr, dtype=np.float64), len(lat15), len(lon15)) * scale
        off_15[var] = off
        ours_15[var] = _interp(ours_025[var], src_lat, src_lon, lat15, lon15)
        if truth_case is not None:
            truth_15[var] = _interp(
                truth_case.truth[var][-1], truth_case.lat, truth_case.lon, lat15, lon15
            )

    print("\n=== 1.5° 网格：HRES T0+finetuned vs 官方 ===")
    results = {}
    for var, (_, _, _, units) in SPECS.items():
        a = _metrics(ours_15[var], off_15[var], lat15)
        item = {"units": units, "ours_vs_official": a}
        if truth_case is not None:
            item["ours_vs_era5"] = _metrics(ours_15[var], truth_15[var], lat15)
            item["official_vs_era5"] = _metrics(off_15[var], truth_15[var], lat15)
        results[var] = item
        line = (f"{var:5s} ours-official RMSE={a['rmse']:.4f} {units:5s} "
                f"corr={a['corr']:.5f}")
        if truth_case is not None:
            line += (f" | ours-ERA5={item['ours_vs_era5']['rmse']:.4f}"
                     f" | official-ERA5={item['official_vs_era5']['rmse']:.4f}")
        print(line)

    np.savez(
        os.path.join(args.out, "fields.npz"),
        lat15=lat15, lon15=lon15,
        **{f"ours_{k}": v for k, v in ours_15.items()},
        **{f"official_{k}": v for k, v in off_15.items()},
        **{f"truth_{k}": v for k, v in truth_15.items()},
        ours_2t_025=ours_025["2t"],
        lat_025=ic_source.lat[:720],
        lon_025=ic_source.lon,
    )
    _plot(
        ours_15, off_15, lat15, lon15,
        os.path.join(args.out, "official_vs_ours_matched_15deg.png"),
        f"HRES T0 + finetuned vs official, init {args.init_time:%Y-%m-%dT%H:%M}, "
        f"+{args.lead_hours}h (1.5 deg)",
    )
    payload = {
        "init_time": args.init_time.isoformat(),
        "lead_hours": args.lead_hours,
        "ic_source": "hres_t0",
        "checkpoint": args.checkpoint,
        "checkpoint_path": args.checkpoint_path,
        "official_15deg": OFFICIAL_15,
        "official_time_index": time_idx,
        "official_lead_index": lead_idx,
        "at_15deg": results,
        "note": "official dataset is model output, not ERA5 truth",
    }

    if args.strict_025_2t:
        print("\n读取官方 0.25° 2t（41 MB chunk）...", flush=True)
        g025 = zarr.open_group(fs.get_mapper(OFFICIAL_025), mode="r")
        t_idx = _time_index(g025, args.init_time)
        l_idx = _lead_index(g025, args.lead_hours)
        off_2t = np.asarray(
            g025[_key(g025, "2m_temperature")][t_idx, l_idx], dtype=np.float64
        )
        m = _metrics(ours_025["2t"], off_2t, ic_source.lat[:720])
        payload["at_025deg_2t"] = m
        print(f"0.25° 2t ours-official: RMSE={m['rmse']:.4f} K "
              f"bias={m['bias']:+.4f} corr={m['corr']:.6f}")

    with open(os.path.join(args.out, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.out}/comparison.json, fields.npz, "
          f"official_vs_ours_matched_15deg.png")


if __name__ == "__main__":
    main()
