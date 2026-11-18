"""Phase 2/3：Aurora 对 ISD 地面站和 IGRA2 探空的观测验证。

- Phase 2：2t / msl / 10u，ISD-Lite 地面站；
- Phase 3：t850 / z500，IGRA2 探空；
- 基线：persistence、ERA5-at-station；
- 可选气候态 ACC（--climatology wb2，首次运行会下载 1.5° 气候态）。

用法：

    python -u verify_obs.py \
      --init-times 2022-01-02T00:00:00,... \
      --lead-hours 240 \
      --checkpoint-path /path/to/aurora-0.25-finetuned.ckpt \
      --cache-dir cache --out results/obs_phase23_2022
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aurora_utils import build_batch, choose_device, load_model, load_static, run_rollout
from config import LEVELS, forecast_times, parse_utc
from metrics import G
from obs_sources import (
    Station,
    TARGET_LAT,
    load_igra_cache,
    load_isd_cache,
    value_at,
)
from sources import HREST0Source

TARGET_LON = np.linspace(0.0, 359.75, 1440)
SURFACE_VARS = ["2t", "msl", "10u"]
UPPER_VARS = ["t850", "z500"]
ALL_VARS = SURFACE_VARS + UPPER_VARS
METHODS = ["aurora", "aurora_adj", "era5", "era5_adj", "persistence", "persistence_adj"]


def log(message: str) -> None:
    print(message, flush=True)


def season_of(init) -> str:
    return {1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON",
            11: "SON", 12: "DJF"}[init.month]


def region_of(lat: float) -> str:
    if abs(lat) >= 60:
        return "polar"
    if abs(lat) < 23.5:
        return "tropics"
    return "midlat"


def nearest_index(lat: np.ndarray, lon: np.ndarray, slat: float, slon: float) -> tuple[int, int]:
    i = int(np.argmin(np.abs(lat - slat)))
    lon360 = slon % 360.0
    j = int(np.argmin(np.abs(((lon - lon360 + 180.0) % 360.0) - 180.0)))
    return i, j


def extract_fields(pred) -> dict[str, np.ndarray]:
    z = pred.atmos_vars["z"].numpy().squeeze()
    t = pred.atmos_vars["t"].numpy().squeeze()
    return {
        "2t": pred.surf_vars["2t"].numpy().squeeze(),
        "msl": pred.surf_vars["msl"].numpy().squeeze() * 1e-2,
        "10u": pred.surf_vars["10u"].numpy().squeeze(),
        "z500": z[LEVELS.index(500)] / G,
        "t850": t[LEVELS.index(850)],
    }


def load_truth(cache_dir: Path, init, lead_hours: int) -> dict[str, np.ndarray]:
    path = cache_dir / f"wb2truth_{init:%Y%m%dT%H%M}_{lead_hours}h.npz"
    if not path.exists():
        # paper 运行缓存的是 240h；短 smoke 直接复用前几个时效。
        path = cache_dir / f"wb2truth_{init:%Y%m%dT%H%M}_240h.npz"
    data = np.load(path)
    return {
        "2t": data["truth_2t"],
        "msl": data["truth_msl"],
        "10u": data["truth_10u"],
        "z500": data["truth_z500"],
        "t850": data["truth_t850"],
    }


def metrics(obs: list[float], pred: list[float]) -> dict:
    o = np.asarray(obs, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(o) & np.isfinite(p)
    o, p = o[mask], p[mask]
    n = int(o.size)
    if n == 0:
        return {"n": 0, "bias": None, "mae": None, "rmse": None, "corr": None}
    diff = p - o
    corr = None
    if n >= 3 and np.std(o) > 0 and np.std(p) > 0:
        corr = float(np.corrcoef(o, p)[0, 1])
    return {
        "n": n,
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "corr": corr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-times", required=True)
    parser.add_argument("--lead-hours", type=int, default=240)
    parser.add_argument("--checkpoint-path", default="aurora-0.25-finetuned.ckpt")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--out", default="results/obs_phase23_2022")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--climatology", choices=["none", "wb2"], default="none")
    parser.add_argument("--source", choices=["both", "isd", "igra2"], default="both")
    args = parser.parse_args()

    if args.lead_hours % 6 != 0:
        raise SystemExit("--lead-hours 必须是 6 的倍数")
    if not os.path.exists(args.checkpoint_path):
        raise SystemExit(f"找不到 checkpoint: {args.checkpoint_path}")
    cache_dir = Path(args.cache_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    init_times = [parse_utc(x.strip()) for x in args.init_times.split(",") if x.strip()]
    if not init_times:
        raise SystemExit("--init-times 为空")

    isd_stations, isd_data = ([], {})
    igra_stations, igra_data = ([], {})
    if args.source in ("both", "isd"):
        isd_stations, isd_data = load_isd_cache(cache_dir)
        log(f"ISD 台站: {len(isd_stations)}")
    if args.source in ("both", "igra2"):
        igra_stations, igra_data = load_igra_cache(cache_dir)
        log(f"IGRA2 台站: {len(igra_stations)}")

    device = choose_device(args.device)
    log(f"加载模型（{device}）...")
    model = load_model(device, checkpoint_path=args.checkpoint_path, use_lora=True)
    source = HREST0Source(workers=4, cache_dir=str(cache_dir))
    static_z = load_static()["z"].numpy()[:720] / G
    clim_source = None
    if args.climatology == "wb2":
        from sources import ClimatologySource
        clim_source = ClimatologySource(str(cache_dir), workers=4, use_cache=True)

    # 预计算台站最近格点、模式地形、区域
    isd_meta = []
    for s in isd_stations:
        i, j = nearest_index(TARGET_LAT, TARGET_LON, s.lat, s.lon)
        isd_meta.append((s, i, j, float(static_z[i, j]), region_of(s.lat)))
    igra_meta = []
    for s in igra_stations:
        i, j = nearest_index(TARGET_LAT, TARGET_LON, s.lat, s.lon)
        igra_meta.append((s, i, j, float(static_z[i, j]), region_of(s.lat)))

    groups: dict[tuple, list[list[float]]] = defaultdict(lambda: [[], []])
    case_rmse: dict[tuple, dict[int, float]] = defaultdict(dict)
    rows = {k: [] for k in [
        "case_idx", "lead_hours", "valid_time", "source", "station", "lat", "lon",
        "elev", "model_elev", "dz", "variable", "obs",
        "aurora", "aurora_adj", "era5", "era5_adj", "persistence", "persistence_adj",
        "clim",
    ]}

    for ci, init in enumerate(init_times):
        valid_times = [init + timedelta(hours=h) for h in range(6, args.lead_hours + 1, 6)]
        log(f"[{ci+1}/{len(init_times)}] {init.isoformat()} ...")
        single, press = source.load_ic(init)
        batch = build_batch(single, press, init)
        preds = run_rollout(model, batch, len(valid_times))
        truth = load_truth(cache_dir, init, args.lead_hours)
        persistence = {
            "2t": single["t2m"].values[-1][:720],
            "msl": single["msl"].values[-1][:720] * 1e-2,
            "10u": single["u10"].values[-1][:720],
            "z500": press["z"].values[-1, LEVELS.index(500)][:720] / G,
            "t850": press["t"].values[-1, LEVELS.index(850)][:720],
        }
        clim = None
        if clim_source is not None:
            clim = clim_source.load(valid_times, TARGET_LAT, TARGET_LON)

        season = season_of(init)
        for li, (pred, vt) in enumerate(zip(preds, valid_times)):
            lead = (li + 1) * 6
            hour = int(vt.timestamp() // 3600)
            fields = extract_fields(pred)
            truth_i = {k: truth[k][li] for k in ALL_VARS}

            def add_obs(source_name, s, i, j, elev, dz, region, obs_values, model_elev):
                # 2t 海拔订正：所有方法使用同一个 dz
                for var, obs in obs_values.items():
                    if not np.isfinite(obs):
                        continue
                    av = float(fields[var][i, j])
                    ev = float(truth_i[var][i, j])
                    pv = float(persistence[var][i, j])
                    a_adj = av + 0.0065 * dz if var == "2t" else None
                    e_adj = ev + 0.0065 * dz if var == "2t" else None
                    p_adj = pv + 0.0065 * dz if var == "2t" else None
                    clim_v = None
                    if clim is not None:
                        clim_v = float(clim[var][li][i, j])
                    rows["case_idx"].append(ci)
                    rows["lead_hours"].append(lead)
                    rows["valid_time"].append(vt.isoformat())
                    rows["source"].append(source_name)
                    rows["station"].append(s.sid)
                    rows["lat"].append(s.lat)
                    rows["lon"].append(s.lon)
                    rows["elev"].append(elev)
                    rows["model_elev"].append(model_elev)
                    rows["dz"].append(dz)
                    rows["variable"].append(var)
                    rows["obs"].append(float(obs))
                    rows["aurora"].append(av)
                    rows["aurora_adj"].append(np.nan if a_adj is None else a_adj)
                    rows["era5"].append(ev)
                    rows["era5_adj"].append(np.nan if e_adj is None else e_adj)
                    rows["persistence"].append(pv)
                    rows["persistence_adj"].append(np.nan if p_adj is None else p_adj)
                    rows["clim"].append(np.nan if clim_v is None else clim_v)
                    for method, value in [
                        ("aurora", av), ("aurora_adj", a_adj),
                        ("era5", ev), ("era5_adj", e_adj),
                        ("persistence", pv), ("persistence_adj", p_adj),
                    ]:
                        if value is None:
                            continue
                        key = (var, lead, method, source_name)
                        groups[key][0].append(float(obs))
                        groups[key][1].append(value)
                    # 每个 case 的 RMSE，用于跨 case 均值/标准差
                    for method, value in [("aurora", av), ("era5", ev), ("persistence", pv)]:
                        key = (var, lead, method, source_name)
                        case_rmse[key].setdefault(ci, []).append((float(obs), value))

            if isd_meta:
                for s, i, j, model_elev, region in isd_meta:
                    rec = isd_data[s.sid]
                    obs_values = {
                        var: value_at(rec, hour, var, tolerance=0.5)
                        for var in SURFACE_VARS
                    }
                    if all(not np.isfinite(v) for v in obs_values.values()):
                        continue
                    dz = model_elev - s.elev
                    add_obs("ISD", s, i, j, s.elev, dz, region, obs_values, model_elev)

            if igra_meta and vt.hour in (0, 12):
                for s, i, j, model_elev, region in igra_meta:
                    rec = igra_data[s.sid]
                    obs_values = {
                        var: value_at(rec, hour, var, tolerance=0.01)
                        for var in UPPER_VARS
                    }
                    if all(not np.isfinite(v) for v in obs_values.values()):
                        continue
                    dz = model_elev - s.elev
                    add_obs("IGRA2", s, i, j, s.elev, dz, region, obs_values, model_elev)

        del preds, single, press, truth, persistence, clim

    # 汇总
    summary = []
    for (var, lead, method, source_name), (obs, pred) in sorted(groups.items()):
        m = metrics(obs, pred)
        per_case = []
        for vals in case_rmse[(var, lead, method, source_name)].values():
            if not vals:
                continue
            o = np.asarray([x[0] for x in vals]); p = np.asarray([x[1] for x in vals])
            per_case.append(float(np.sqrt(np.mean((p - o) ** 2))))
        rmse_mean = float(np.mean(per_case)) if per_case else None
        rmse_std = float(np.std(per_case)) if per_case else None
        summary.append({
            "variable": var, "lead_hours": lead, "method": method, "source": source_name,
            **m, "n_cases": len(per_case), "rmse_case_mean": rmse_mean, "rmse_case_std": rmse_std,
        })

    # ACC：按 case/lead 在台站上计算异常相关，再对 case 取均值。
    acc_mean_map: dict[tuple, tuple[float | None, float | None, int]] = {}
    if args.climatology == "wb2":
        arr = {k: np.asarray(v) for k, v in rows.items()}
        acc_data: dict[tuple, dict[int, list[tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        methods = ["aurora", "aurora_adj", "era5", "era5_adj",
                   "persistence", "persistence_adj"]
        for idx in range(len(arr["obs"])):
            clim_v = float(arr["clim"][idx])
            if not np.isfinite(clim_v):
                continue
            obs = float(arr["obs"][idx])
            var = str(arr["variable"][idx])
            lead = int(arr["lead_hours"][idx])
            source_name = str(arr["source"][idx])
            case_idx = int(arr["case_idx"][idx])
            for method in methods:
                value = float(arr[method][idx])
                if not np.isfinite(value):
                    continue
                acc_data[(var, lead, method, source_name)][case_idx].append(
                    (obs - clim_v, value - clim_v)
                )
        for key, per_case in acc_data.items():
            accs = []
            for values in per_case.values():
                if len(values) < 3:
                    continue
                o = np.asarray([x[0] for x in values])
                p = np.asarray([x[1] for x in values])
                if np.std(o) > 0 and np.std(p) > 0:
                    accs.append(float(np.corrcoef(o, p)[0, 1]))
            if accs:
                acc_mean_map[key] = (float(np.mean(accs)), float(np.std(accs)), len(accs))
    for row in summary:
        key = (row["variable"], row["lead_hours"], row["method"], row["source"])
        if key in acc_mean_map:
            row["acc_mean"], row["acc_std"], row["n_acc_cases"] = acc_mean_map[key]
        else:
            row["acc_mean"], row["acc_std"], row["n_acc_cases"] = None, None, 0

    # per-observation 数组
    np.savez(
        out / "per_obs.npz",
        **{k: np.asarray(v) for k, v in rows.items()},
    )
    fields = ["variable", "lead_hours", "method", "source", "n", "bias", "mae",
              "rmse", "corr", "n_cases", "rmse_case_mean", "rmse_case_std",
              "acc_mean", "acc_std", "n_acc_cases"]
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"init_times": [t.isoformat() for t in init_times],
                   "lead_hours": args.lead_hours,
                   "isd_stations": len(isd_stations),
                   "igra_stations": len(igra_stations),
                   "climatology": args.climatology,
                   "summary": summary}, f, ensure_ascii=False, indent=2)

    # 图：RMSE vs lead（1 行 × 5 列横向布局）
    fig, axes = plt.subplots(
        1, len(ALL_VARS), figsize=(4.0 * len(ALL_VARS), 3.8), sharex=True
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, var in zip(axes, ALL_VARS):
        for method, style in [("aurora", "-o"), ("era5", "-s"), ("persistence", "-^")]:
            pts = [r for r in summary if r["variable"] == var and r["method"] == method]
            pts.sort(key=lambda r: r["lead_hours"])
            if pts:
                ax.plot([r["lead_hours"] for r in pts], [r["rmse"] for r in pts], style, label=method, ms=3)
        ax.set_title(var); ax.set_xlabel("lead (h)"); ax.set_ylabel("RMSE")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    for ax in axes[len(ALL_VARS):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "rmse_vs_lead.png", dpi=150); plt.close(fig)

    log(f"结果已保存: {out}/summary.csv, per_obs.npz, rmse_vs_lead.png")


if __name__ == "__main__":
    main()
