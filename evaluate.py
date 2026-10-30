"""Phase 1 评估：多起报时次 + 五变量（2t/msl/10u/z500/t850）+ RMSE/ACC + 基线。"""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aurora_utils import build_batch, choose_device, load_model, run_rollout
from config import (
    CHECKPOINT_NAME,
    DEFAULT_INIT_TIME,
    HF_REPO_ID,
    HF_REVISION,
    LEVELS,
    MODEL_STEP_HOURS,
    forecast_times,
    parse_lead_hours,
    parse_utc,
)
from metrics import G, acc, area_weights, weighted_bias, weighted_rmse
from sources import (
    VARIABLES,
    VAR_BY_NAME,
    CaseData,
    ClimatologySource,
    EarthmoverSource,
    HREST0Source,
    LocalSource,
    WB2Source,
)

TARGET_LAT = np.linspace(90.0, -89.75, 720)
TARGET_LON = np.linspace(0.0, 359.75, 1440)


def extract_prediction(pred, spec) -> np.ndarray:
    if spec.kind == "surface":
        field = pred.surf_vars[spec.model_name].numpy().squeeze()
    else:
        field = pred.atmos_vars[spec.model_name].numpy().squeeze()
        field = field[LEVELS.index(spec.level)]
    return field * spec.scale


def evaluate_case(
    case: CaseData,
    preds,
    climatology: dict[str, np.ndarray] | None,
) -> list[dict]:
    if not np.allclose(case.lat, TARGET_LAT):
        raise RuntimeError("case 纬度网格与模型输出不一致")
    if not np.allclose(case.lon, TARGET_LON):
        raise RuntimeError("case 经度网格与模型输出不一致")
    weights = area_weights(case.lat)
    rows = []
    for i, (pred, vt) in enumerate(zip(preds, case.valid_times)):
        lead = int((vt - case.init_time).total_seconds() // 3600)
        pred_lat = pred.metadata.lat.detach().cpu().numpy()
        pred_lon = pred.metadata.lon.detach().cpu().numpy()
        if not (np.allclose(pred_lat, case.lat) and np.allclose(pred_lon, case.lon)):
            raise RuntimeError(f"+{lead}h 预测网格与真值网格不一致")

        for spec in VARIABLES:
            truth = case.truth[spec.name][i]
            if truth.shape != (case.lat.size, case.lon.size):
                raise RuntimeError(f"{spec.name} +{lead}h 真值形状异常: {truth.shape}")
            fields = {
                "aurora": extract_prediction(pred, spec),
                "persistence": case.persistence[spec.name],
            }
            if climatology is not None:
                fields["climatology"] = climatology[spec.name][i]
            for method, field in fields.items():
                if field.shape != truth.shape:
                    raise RuntimeError(
                        f"{spec.name} +{lead}h {method} 形状异常: {field.shape}"
                    )
                if not np.isfinite(field).all():
                    raise RuntimeError(f"{spec.name} +{lead}h {method} 含 NaN/Inf")
                row = {
                    "case": case.init_time.isoformat(),
                    "valid_time": vt.isoformat(),
                    "lead_hours": lead,
                    "variable": spec.name,
                    "method": method,
                    "units": spec.units,
                    "rmse": weighted_rmse(field, truth, weights),
                    "bias": weighted_bias(field, truth, weights),
                    "acc": None,
                }
                if method != "climatology" and climatology is not None:
                    row["acc"] = acc(field, truth, climatology[spec.name][i], weights)
                rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["variable"], row["lead_hours"], row["method"])].append(row)

    summary = []
    for (variable, lead, method), items in sorted(groups.items()):
        def stats(key):
            values = np.array(
                [x[key] for x in items if x[key] is not None and np.isfinite(x[key])],
                dtype=np.float64,
            )
            if values.size == 0:
                return None, None
            return float(values.mean()), float(values.std(ddof=0))

        rmse_mean, rmse_std = stats("rmse")
        bias_mean, bias_std = stats("bias")
        acc_mean, acc_std = stats("acc")
        summary.append({
            "variable": variable,
            "lead_hours": lead,
            "method": method,
            "n_cases": len(items),
            "units": items[0]["units"],
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "bias_mean": bias_mean,
            "bias_std": bias_std,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
        })
    return summary


def plot_summary(summary: list[dict], out_dir: str) -> None:
    leads = sorted({row["lead_hours"] for row in summary})
    variables = [v.name for v in VARIABLES]
    specs = VAR_BY_NAME

    for metric, fname, methods in [
        ("rmse", "rmse_vs_lead.png", ["aurora", "persistence", "climatology"]),
        ("acc", "acc_vs_lead.png", ["aurora", "persistence"]),
    ]:
        if not any(
            row[f"{metric}_mean"] is not None and np.isfinite(row[f"{metric}_mean"])
            for row in summary
        ):
            continue
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        axes = axes.ravel()
        for ax, variable in zip(axes, variables):
            for method in methods:
                points = [
                    row for row in summary
                    if row["variable"] == variable
                    and row["method"] == method
                    and row[f"{metric}_mean"] is not None
                    and np.isfinite(row[f"{metric}_mean"])
                ]
                points.sort(key=lambda r: r["lead_hours"])
                if not points:
                    continue
                ax.plot(
                    [p["lead_hours"] for p in points],
                    [p[f"{metric}_mean"] for p in points],
                    marker="o", label=method,
                )
            ax.set_title(f"{variable} ({specs[variable].units})")
            ax.set_xlabel("lead time (h)")
            ax.set_ylabel(metric.upper())
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        if len(variables) < len(axes):
            axes[len(variables)].axis("off")
        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"图已保存: {path}")


def write_csv(path: str, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=["local", "earthmover", "wb2", "paper"], default="local"
    )
    parser.add_argument(
        "--init-times",
        default=DEFAULT_INIT_TIME,
        help="逗号分隔的起报时刻（UTC ISO 8601），默认 2026-03-13T00:00:00",
    )
    parser.add_argument("--lead-hours", type=parse_lead_hours, default=24)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out", default="results", help="结果目录")
    parser.add_argument("--data-dir", default=".", help="本地 NetCDF 目录（local）")
    parser.add_argument("--cache-dir", default="cache", help="下载/中间数据缓存目录")
    parser.add_argument(
        "--climatology", choices=["wb2", "none"], default="wb2",
        help="ACC 用的气候态；none 表示只算 RMSE/bias",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8,
                        help="下载并发数（Earthmover/WB2/HRES/气候态）")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint 名；默认 pretrained，paper 源默认 finetuned")
    parser.add_argument("--checkpoint-path", default=None,
                        help="本地 checkpoint 路径（fine-tuned 推荐）")
    parser.add_argument("--use-lora", action="store_true",
                        help="fine-tuned LoRA checkpoint 需要")
    parser.add_argument("--no-clim-cache", action="store_true",
                        help="不把逐时次气候态字段写到磁盘（10 天多 case 推荐）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = (
            "aurora-0.25-finetuned.ckpt" if args.source == "paper" else CHECKPOINT_NAME
        )
    if args.source == "paper":
        args.use_lora = True

    init_times = [parse_utc(x.strip()) for x in args.init_times.split(",") if x.strip()]
    if args.max_cases is not None:
        init_times = init_times[: args.max_cases]
    if not init_times:
        raise SystemExit("--init-times 为空")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    ic_source = None
    truth_source = None
    if args.source == "local":
        source = LocalSource(args.data_dir, args.cache_dir)
    elif args.source == "earthmover":
        source = EarthmoverSource(args.cache_dir, workers=args.workers)
    elif args.source == "paper":
        ic_source = HREST0Source(workers=args.workers)
        truth_source = WB2Source(args.cache_dir, workers=args.workers)
        source = None
    else:
        source = WB2Source(args.cache_dir, workers=args.workers)

    if args.dry_run:
        print(f"source={args.source}, cases={len(init_times)}, lead={args.lead_hours}h")
        print("variables:", [v.name for v in VARIABLES])
        for init in init_times:
            history, valid = forecast_times(init, args.lead_hours)
            print(f"  init {init.isoformat()}: history={[t.isoformat() for t in history]}")
            print(f"    valid={[t.isoformat() for t in valid]}")
            if args.source == "wb2":
                print(f"    cache={source._cache_path(init, args.lead_hours)}")
                print(f"    indices={[source.index(t) for t in history + valid]}")
            elif args.source == "earthmover":
                print(f"    cache={source._cache_path(init, args.lead_hours)}")
            elif args.source == "paper":
                print(f"    hres_idx={ic_source.index(init)}")
                print(f"    wb2_truth_cache="
                      f"{truth_source._truth_cache_path(init, args.lead_hours)}")
        return

    device = choose_device(args.device)
    model = load_model(
        device,
        checkpoint=args.checkpoint,
        checkpoint_path=args.checkpoint_path,
        use_lora=args.use_lora,
    )
    print(f"模型已加载，设备: {device}，checkpoint: {args.checkpoint}")

    clim_source = None
    if args.climatology == "wb2":
        print("加载 1990-2019 6h 气候态（1.5° -> 0.25°）...")
        clim_source = ClimatologySource(
            args.cache_dir,
            workers=args.workers,
            use_cache=not args.no_clim_cache,
        )

    all_rows = []
    for case_idx, init in enumerate(init_times, start=1):
        print(f"[{case_idx}/{len(init_times)}] {init.isoformat()}")
        if args.source == "paper":
            history, valid = forecast_times(init, args.lead_hours)
            single, press = ic_source.load_ic(init)
            truth = truth_source.load_truth(init, args.lead_hours)
            persistence = {
                "2t": single["t2m"].values[-1][:720],
                "msl": single["msl"].values[-1][:720] * VAR_BY_NAME["msl"].scale,
                "10u": single["u10"].values[-1][:720],
                "z500": press["z"].values[-1, LEVELS.index(500)][:720]
                * VAR_BY_NAME["z500"].scale,
                "t850": press["t"].values[-1, LEVELS.index(850)][:720],
            }
            case = CaseData(
                init_time=init,
                lead_hours=args.lead_hours,
                valid_times=valid,
                single=single,
                press=press,
                truth=truth,
                persistence=persistence,
                lat=ic_source.lat[:720],
                lon=ic_source.lon,
            )
        else:
            case = source.load_case(init, args.lead_hours)
        steps = args.lead_hours // MODEL_STEP_HOURS
        batch = build_batch(case.single, case.press, init)
        preds = run_rollout(model, batch, steps)
        case_clim = None
        if clim_source is not None:
            case_clim = clim_source.load(case.valid_times, TARGET_LAT, TARGET_LON)
        rows = evaluate_case(case, preds, case_clim)
        all_rows.extend(rows)
        for row in rows:
            if row["method"] == "aurora" and row["lead_hours"] == args.lead_hours:
                print(
                    f"  {row['variable']:5s} +{row['lead_hours']}h "
                    f"RMSE={row['rmse']:.3f} {row['units']} "
                    f"ACC={row['acc'] if row['acc'] is None else round(row['acc'], 3)}"
                )

    summary = summarize(all_rows)
    write_csv(
        os.path.join(args.out, "per_case.csv"),
        all_rows,
        ["case", "valid_time", "lead_hours", "variable", "method", "units",
         "rmse", "bias", "acc"],
    )
    write_csv(
        os.path.join(args.out, "summary.csv"),
        summary,
        ["variable", "lead_hours", "method", "n_cases", "units",
         "rmse_mean", "rmse_std", "bias_mean", "bias_std", "acc_mean", "acc_std"],
    )
    payload = {
        "source": args.source,
        "ic_source": "hres_t0" if args.source == "paper" else args.source,
        "truth_source": "wb2_era5" if args.source == "paper" else args.source,
        "init_times": [t.isoformat() for t in init_times],
        "lead_hours": args.lead_hours,
        "variables": {v.name: v.units for v in VARIABLES},
        "climatology": args.climatology,
        "model": {
            "repo": HF_REPO_ID,
            "checkpoint": args.checkpoint,
            "checkpoint_path": args.checkpoint_path,
            "use_lora": args.use_lora,
            "revision": HF_REVISION,
        },
        "metric_note": "纬度加权（cos lat）；z500 为 geopotential/g（m），msl 为 hPa",
        "summary": summary,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {args.out}/summary.csv, per_case.csv, summary.json")
    plot_summary(summary, args.out)


if __name__ == "__main__":
    main()
