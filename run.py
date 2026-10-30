"""Aurora 0.25° 预报复现：读本地缓存 -> 组装 Batch -> rollout -> 对比 ERA5 算 RMSE。"""

import os

# 24 GB 显卡上 fp32 会 OOM；该设置降低分配器碎片导致的 OOM 风险。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from aurora_utils import build_batch, choose_device, load_model, run_rollout
from config import (
    CHECKPOINT_NAME,
    HF_REPO_ID,
    HF_REVISION,
    LEVELS,
    MODEL_STEP_HOURS,
    add_common_args,
    forecast_times,
    to_numpy_time,
)


def require_times(ds: xr.Dataset, times, name: str, data_dir: str) -> None:
    available = set(ds.time.values.astype("datetime64[ns]").astype(np.int64))
    missing = [
        str(t)
        for t in times
        if int(to_numpy_time(t).astype(np.int64)) not in available
    ]
    if missing:
        raise SystemExit(
            f"{name} 缺少时次 {missing}；请先运行 fetch_data.py，"
            f"并使用相同的 --init-time / --lead-hours（数据目录: {data_dir}）"
        )


def weighted_rmse(pred: np.ndarray, truth: np.ndarray, lat: np.ndarray) -> float:
    """纬度加权全球 RMSE（cos(lat) 权重，经度均匀网格）。"""
    w = np.cos(np.deg2rad(lat))
    return float(
        np.sqrt((w[:, None] * (pred - truth) ** 2).sum() / (w.sum() * pred.shape[1]))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="推理设备，默认自动选择 CUDA（不可用时回退 CPU）",
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        help="ERA5 NetCDF 所在目录，默认当前目录",
    )
    args = parser.parse_args()

    history, valid = forecast_times(args.init_time, args.lead_hours)
    steps = args.lead_hours // MODEL_STEP_HOURS
    os.makedirs(args.out, exist_ok=True)
    device = choose_device(args.device)

    single_path = os.path.join(args.data_dir, "era5_single.nc")
    press_path = os.path.join(args.data_dir, "era5_press.nc")
    for path in (single_path, press_path):
        if not os.path.exists(path):
            raise SystemExit(
                f"缺少 {path}，请先运行: "
                f"python -u fetch_data.py --out {args.data_dir}"
            )
    single_full = xr.open_dataset(single_path)
    press = xr.open_dataset(press_path)

    require_times(single_full, history + valid, "era5_single.nc", args.data_dir)
    require_times(press, history, "era5_press.nc", args.data_dir)
    if list(press.level.values) != LEVELS:
        raise SystemExit(f"era5_press.nc 的气压层不匹配: {press.level.values.tolist()}")

    single = single_full.sel(time=[to_numpy_time(t) for t in history])
    press = press.sel(time=[to_numpy_time(t) for t in history], level=LEVELS)
    print(f"数据就绪: 地面 {dict(single.sizes)}, 高空 {dict(press.sizes)}")

    batch = build_batch(single, press, args.init_time)
    print("Batch 组装完成")

    model = load_model(device)
    print(
        f"模型已加载，设备: {device}，"
        f"autocast: {'on' if device == 'cuda' else 'off'}"
    )

    t0 = time.perf_counter()
    preds = run_rollout(model, batch, steps)
    elapsed = time.perf_counter() - t0
    print(
        f"预测完成 +{args.lead_hours}h，用时 {elapsed:.1f}s"
        f"（{elapsed / steps:.2f}s/步）"
    )

    # 用预测自身的网格做检验，避免硬编码 720 行。
    pred_lat = preds[0].metadata.lat.detach().cpu().numpy().astype(np.float64)
    pred_lon = preds[0].metadata.lon.detach().cpu().numpy().astype(np.float64)
    truth_lat = single_full.latitude.values[: pred_lat.size]
    truth_lon = single_full.longitude.values
    if not np.allclose(pred_lat, truth_lat):
        raise RuntimeError("预测纬度与 ERA5 真值网格不一致")
    if not np.allclose(pred_lon, truth_lon):
        raise RuntimeError("预测经度与 ERA5 真值网格不一致")

    rmses = []
    print("各时效 2 米温度全球加权 RMSE:")
    for i, (pred, vt) in enumerate(zip(preds, valid), start=1):
        truth = single_full["t2m"].sel(time=to_numpy_time(vt)).values.squeeze()
        truth = truth[: pred_lat.size]
        pred_2t = pred.surf_vars["2t"].numpy().squeeze()
        if pred_2t.shape != truth.shape:
            raise RuntimeError(
                f"+{i * MODEL_STEP_HOURS}h 预测/真值形状不一致: "
                f"{pred_2t.shape} vs {truth.shape}"
            )
        if not np.isfinite(pred_2t).all():
            raise RuntimeError(f"+{i * MODEL_STEP_HOURS}h 预测包含 NaN/Inf")
        rmse = weighted_rmse(pred_2t, truth, pred_lat)
        rmses.append(rmse)
        print(
            f"  +{i * MODEL_STEP_HOURS}h ({vt:%Y-%m-%dT%H:%M} UTC)  "
            f"RMSE = {rmse:.3f} K"
        )

    preds_path = os.path.join(args.out, "preds_2t.npy")
    np.save(preds_path, np.stack([p.surf_vars["2t"].numpy().squeeze() for p in preds]))

    # 最后一个时效的预测与偏差图
    pred_2t = preds[-1].surf_vars["2t"].numpy().squeeze()
    truth_2t = single_full["t2m"].sel(time=to_numpy_time(valid[-1])).values.squeeze()
    truth_2t = truth_2t[: pred_lat.size]
    bias = pred_2t - truth_2t

    fig_path = os.path.join(args.out, f"verify_{args.lead_hours}h.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].imshow(pred_2t, origin="upper")
    axes[0].set_title(f"Aurora +{args.lead_hours}h 2t (K)")
    im = axes[1].imshow(bias, origin="upper", cmap="RdBu_r", vmin=-5, vmax=5)
    axes[1].set_title(f"bias vs ERA5, RMSE={rmses[-1]:.2f}K")
    fig.colorbar(im, ax=axes[1])
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存: {fig_path}")

    results = {
        "init_time": args.init_time.isoformat(),
        "lead_hours": args.lead_hours,
        "device": device,
        "model": {
            "repo": HF_REPO_ID,
            "checkpoint": CHECKPOINT_NAME,
            "revision": HF_REVISION,
        },
        "era5": {
            "snapshot": single_full.attrs.get("source_snapshot", "unknown"),
        },
        "rmse_2t_k": {
            f"+{i * MODEL_STEP_HOURS}h": rmse
            for i, rmse in enumerate(rmses, start=1)
        },
        "runtime_seconds": round(elapsed, 2),
        "files": {
            "preds": preds_path,
            "figure": fig_path,
        },
    }
    results_path = os.path.join(args.out, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"指标已保存: {results_path}")


if __name__ == "__main__":
    main()
