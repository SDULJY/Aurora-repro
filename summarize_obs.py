"""对 Phase 2/3 的 per_obs.npz 做区域/季节/美国vs非美国分层汇总。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ["aurora", "aurora_adj", "era5", "era5_adj", "persistence", "persistence_adj"]


def metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    mask = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[mask], pred[mask]
    n = int(o.size)
    if n == 0:
        return {"n": 0, "bias": None, "mae": None, "rmse": None, "corr": None}
    d = p - o
    corr = None
    if n >= 3 and np.std(o) > 0 and np.std(p) > 0:
        corr = float(np.corrcoef(o, p)[0, 1])
    return {"n": n, "bias": float(np.mean(d)), "mae": float(np.mean(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d**2))), "corr": corr}


def region_of(lat: float) -> str:
    a = abs(float(lat))
    return "polar" if a >= 60 else ("tropics" if a < 23.5 else "midlat")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/obs_phase23_2022")
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()
    results = Path(args.results)
    data = np.load(results / "per_obs.npz")
    meta = json.load(open(results / "summary.json", encoding="utf-8"))
    init_times = meta["init_times"]
    seasons = []
    for t in init_times:
        month = int(t[5:7])
        seasons.append("DJF" if month in (12, 1, 2) else
                       "MAM" if month in (3, 4, 5) else
                       "JJA" if month in (6, 7, 8) else "SON")

    # ISD 站 -> 国家
    country = {}
    path = Path(args.cache_dir) / "obs_parsed" / "stations_isd.csv"
    if path.exists():
        for row in csv.DictReader(open(path, encoding="utf-8")):
            country[row["sid"]] = row.get("country", "")

    groups = defaultdict(lambda: [[], []])
    n = len(data["obs"])
    for i in range(n):
        source = str(data["source"][i])
        if source != "ISD":
            continue
        var = str(data["variable"][i])
        lead = int(data["lead_hours"][i])
        case = int(data["case_idx"][i])
        station = str(data["station"][i])
        lat = float(data["lat"][i])
        obs = float(data["obs"][i])
        region = region_of(lat)
        season = seasons[case]
        is_us = country.get(station, "") == "US"
        for method in METHODS:
            pred = float(data[method][i])
            if not np.isfinite(pred):
                continue
            for key in [("region", region), ("season", season),
                        ("us", "US" if is_us else "non-US")]:
                gkey = (key[0], key[1], var, lead, method)
                groups[gkey][0].append(obs)
                groups[gkey][1].append(pred)

    rows = []
    for (kind, group, var, lead, method), (obs, pred) in sorted(groups.items()):
        rows.append({"group_type": kind, "group": group, "variable": var,
                     "lead_hours": lead, "method": method,
                     **metrics(np.asarray(obs), np.asarray(pred))})
    out = results / "summary_stratified.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        fields = ["group_type", "group", "variable", "lead_hours", "method",
                  "n", "bias", "mae", "rmse", "corr"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
