"""预下载 Phase 2/3 所需的 HRES T0 初始场。

每个 case 会缓存为 cache/hres_ic_<init>.npz（约 547 MB），
后续 evaluate / verify 直接复用，不再重复访问 GCS。

用法：

    python -u fetch_hres_ic.py
    python -u fetch_hres_ic.py --init-times 2022-01-02T00:00:00,2022-04-01T00:00:00
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from config import parse_utc
from sources import HREST0Source

DEFAULT_INIT_TIMES = [
    "2022-01-02T00:00:00",
    "2022-01-02T12:00:00",
    "2022-01-16T00:00:00",
    "2022-01-16T12:00:00",
    "2022-04-01T00:00:00",
    "2022-04-01T12:00:00",
    "2022-04-15T00:00:00",
    "2022-04-15T12:00:00",
    "2022-07-01T00:00:00",
    "2022-07-01T12:00:00",
    "2022-07-15T00:00:00",
    "2022-07-15T12:00:00",
    "2022-10-01T00:00:00",
    "2022-10-01T12:00:00",
    "2022-10-15T00:00:00",
    "2022-10-15T12:00:00",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-times",
        default=",".join(DEFAULT_INIT_TIMES),
        help="逗号分隔的起报时刻；默认 16 个 paper case",
    )
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    init_times = [
        parse_utc(x.strip()) for x in args.init_times.split(",") if x.strip()
    ]
    if not init_times:
        raise SystemExit("--init-times 为空")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for init in init_times:
            path = cache_dir / f"hres_ic_{init:%Y%m%dT%H%M}.npz"
            log(f"{init.isoformat()} -> {path} (exists={path.exists()})")
        return

    log(f"HRES T0 预下载：{len(init_times)} 个 case，cache={cache_dir.resolve()}")
    source = HREST0Source(workers=args.workers, cache_dir=str(cache_dir))
    results = []
    for i, init in enumerate(init_times, start=1):
        path = cache_dir / f"hres_ic_{init:%Y%m%dT%H%M}.npz"
        if path.exists():
            log(f"[{i}/{len(init_times)}] {init.isoformat()} 已缓存，跳过")
            results.append(
                {"init_time": init.isoformat(), "status": "cached", "bytes": path.stat().st_size}
            )
            continue
        log(f"[{i}/{len(init_times)}] {init.isoformat()} 下载中 ...")
        t0 = time.time()
        try:
            source.load_ic(init)
            elapsed = time.time() - t0
            nbytes = path.stat().st_size if path.exists() else 0
            log(f"  完成 {elapsed:.1f}s，缓存 {nbytes / 1024**2:.0f} MB")
            results.append(
                {"init_time": init.isoformat(), "status": "ok",
                 "bytes": nbytes, "seconds": round(elapsed, 1)}
            )
        except Exception as exc:  # noqa: BLE001 - 单 case 失败不终止
            elapsed = time.time() - t0
            log(f"  失败 {elapsed:.1f}s: {type(exc).__name__}: {exc}")
            results.append(
                {"init_time": init.isoformat(), "status": "failed",
                 "error": f"{type(exc).__name__}: {exc}", "seconds": round(elapsed, 1)}
            )

    payload = {
        "cases": len(init_times),
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "cached": sum(1 for r in results if r["status"] == "cached"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out = cache_dir / "hres_ic_prefetch_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log(f"完成: ok={payload['ok']} cached={payload['cached']} failed={payload['failed']}")
    log(f"summary: {out}")


if __name__ == "__main__":
    main()
