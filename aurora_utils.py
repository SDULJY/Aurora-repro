"""Aurora 模型加载、Batch 组装与 rollout 的共享函数。"""

import os

# 24 GB 显卡上 fp32 会 OOM；该设置降低分配器碎片导致的 OOM 风险。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pickle

import numpy as np
import torch
import xarray as xr
from aurora import Aurora, Batch, Metadata, rollout
from huggingface_hub import hf_hub_download

from config import CHECKPOINT_NAME, HF_REPO_ID, HF_REVISION, LEVELS, STATIC_NAME


def choose_device(requested: str) -> str:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 --device cuda，但当前环境没有可用的 CUDA")
    return requested


def load_static() -> dict[str, torch.Tensor]:
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=STATIC_NAME,
        revision=HF_REVISION,
    )
    with open(path, "rb") as f:
        static = pickle.load(f)
    return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in static.items()}


def build_batch(single: xr.Dataset, press: xr.Dataset, init_time) -> Batch:
    """把 ERA5 变量名/布局映射到 Aurora 的 Batch。"""

    def t(da):
        return torch.from_numpy(np.asarray(da.values, dtype=np.float32))[None]

    return Batch(
        surf_vars={
            "2t": t(single["t2m"]),
            "10u": t(single["u10"]),
            "10v": t(single["v10"]),
            "msl": t(single["msl"]),
        },
        static_vars=load_static(),
        atmos_vars={
            "t": t(press["t"]),
            "u": t(press["u"]),
            "v": t(press["v"]),
            "q": t(press["q"]),
            "z": t(press["z"]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(single.latitude.values.astype(np.float32)),
            lon=torch.from_numpy(single.longitude.values.astype(np.float32)),
            time=(init_time,),
            atmos_levels=LEVELS,
        ),
    )


def load_model(
    device: str,
    checkpoint: str = CHECKPOINT_NAME,
    checkpoint_path: str | None = None,
    use_lora: bool = False,
) -> Aurora:
    model = Aurora(use_lora=use_lora, autocast=device == "cuda")
    if checkpoint_path:
        model.load_checkpoint_local(checkpoint_path, strict=True)
    else:
        model.load_checkpoint(HF_REPO_ID, checkpoint, revision=HF_REVISION)
    model.eval().to(device)
    if device == "cuda":
        # 只开 backbone 仍会在 24 GB 卡上 OOM；encoder/decoder 也走 bf16 后峰值约 19.9 GB。
        model.autocast_encoder = True
        model.autocast_decoder = True
    return model


def run_rollout(model: Aurora, batch: Batch, steps: int) -> list[Batch]:
    with torch.inference_mode():
        return [p.to("cpu") for p in rollout(model, batch, steps=steps)]
