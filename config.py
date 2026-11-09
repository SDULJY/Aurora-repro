"""共享配置与 CLI：统一 UTC 时间，由起报时刻自动推导所有验证时刻。"""

from __future__ import annotations

import argparse
import datetime as dt
import os

import numpy as np

MODEL_STEP_HOURS = 6
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SINGLE_VARS = ["t2m", "u10", "v10", "msl"]   # Aurora 需要的 4 个地面变量
PRESS_VARS = ["t", "u", "v", "q", "z"]       # Aurora 需要的 5 个高空变量

DEFAULT_INIT_TIME = "2026-03-13T00:00:00"

# 固定版本：HF 仓库 2026-10-28 的 main，同时包含 0.25 checkpoint 与静态场。
HF_REPO_ID = "microsoft/aurora"
HF_REVISION = "a96afd7ee6d65e3bd2d476f3be798a25a56f2296"
CHECKPOINT_NAME = "aurora-0.25-pretrained.ckpt"
STATIC_NAME = "aurora-0.25-static.pickle"

ERA5_REPO_ID = "earthmover-public/era5"
# 2026-10-28 解析到的 main 快照；若快照过期，fetch_data.py 可传 --snapshot main。
ERA5_SNAPSHOT = "TZ2KX6A24CJ5DG17T7MG"


def display_path(path: str | None) -> str | None:
    """结果元数据只记录文件名，避免把本地绝对路径写入公开结果。"""
    return os.path.basename(path) if path else None


def parse_utc(value: str) -> dt.datetime:
    """把 ISO 8601 字符串解析为 UTC datetime；不带时区时按 UTC 处理。"""
    try:
        t = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"无法解析时间 {value!r}，示例: 2026-03-13T00:00:00"
        ) from exc
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc)


def parse_lead_hours(value: str) -> int:
    """预报时效必须是模型步长（6h）的正整数倍。"""
    try:
        hours = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"无效的时效 {value!r}") from exc
    if hours <= 0 or hours % MODEL_STEP_HOURS:
        raise argparse.ArgumentTypeError(
            f"时效必须是 {MODEL_STEP_HOURS} 的正整数倍，收到 {hours}"
        )
    return hours


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """fetch_data.py 与 run.py 共用的参数。"""
    parser.add_argument(
        "--init-time",
        type=parse_utc,
        default=parse_utc(DEFAULT_INIT_TIME),
        help=f"起报时刻（UTC，ISO 8601），默认 {DEFAULT_INIT_TIME}",
    )
    parser.add_argument(
        "--lead-hours",
        type=parse_lead_hours,
        default=24,
        help="最长预报时效（小时，6 的倍数），默认 24",
    )
    parser.add_argument(
        "--out",
        default=".",
        help="输出目录（fetch 存 NetCDF，run 存预测/图/指标），默认当前目录",
    )


def forecast_times(
    init_time: dt.datetime, lead_hours: int
) -> tuple[list[dt.datetime], list[dt.datetime]]:
    """返回 (历史帧 [t-6h, t], 验证帧 [t+6h, ..., t+lead_hours])。"""
    if lead_hours <= 0 or lead_hours % MODEL_STEP_HOURS:
        raise ValueError(f"lead_hours 必须是 {MODEL_STEP_HOURS} 的正整数倍")
    history = [init_time - dt.timedelta(hours=MODEL_STEP_HOURS), init_time]
    valid = [
        init_time + dt.timedelta(hours=h)
        for h in range(MODEL_STEP_HOURS, lead_hours + 1, MODEL_STEP_HOURS)
    ]
    return history, valid


def to_numpy_time(t: dt.datetime) -> np.datetime64:
    """xarray/NetCDF 使用不带时区的 datetime64[ns]，这里统一转换。"""
    return np.datetime64(t.replace(tzinfo=None), "ns")
