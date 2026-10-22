"""Aurora +6h/+24h 预报复现：读本地缓存 -> 组装 Batch -> rollout -> 对比 ERA5 算 RMSE。"""
import datetime
import os
import pickle

import numpy as np
import torch
import xarray as xr
from aurora import Aurora, Batch, Metadata, rollout
from huggingface_hub import hf_hub_download

# ---- 参数（与 fetch_data.py 的日期联动）----
STEPS = 4                            # 每步 6h；跑通后改成 4 = 完整 24 小时预报
TIMES = ["2026-03-12T18:00:00", "2026-03-13T00:00:00"]   # 历史帧、起报帧
VALID_TIME = "2026-03-14T00:00:00"   # 起报 +24h   # 起报 +6h（改 STEPS 时同步改这里）
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# ---- 1. 本地数据 ----
assert os.path.exists("era5_single.nc") and os.path.exists("era5_press.nc"), \
    "缺少本地数据缓存，请先运行: python -u fetch_data.py"
single_full = xr.open_dataset("era5_single.nc")   # 3 个时次的地面变量
press = xr.open_dataset("era5_press.nc")          # 2 个时次 × 13 层的高空变量
single = single_full.sel(time=TIMES)
print(f"本地数据就绪: 地面{dict(single_full.sizes)} 高空{dict(press.sizes)}")

# ---- 2. 静态变量（HF 缓存过则秒取，自动走镜像）----
static_path = hf_hub_download(repo_id="microsoft/aurora", filename="aurora-0.25-static.pickle")
with open(static_path, "rb") as f:
    static = pickle.load(f)
static = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in static.items()}
print("静态变量就绪:", list(static.keys()))

# ---- 3. 组装 Batch：变量名 ERA5->Aurora，布局 (T=2,[13],lat,lon)->(B=1,T=2,[13],lat,lon) ----
def t(da):
    return torch.from_numpy(np.asarray(da.values, dtype=np.float32))[None]

batch = Batch(
    surf_vars={"2t": t(single["t2m"]), "10u": t(single["u10"]),
               "10v": t(single["v10"]), "msl": t(single["msl"])},
    static_vars=static,
    atmos_vars={"t": t(press["t"]), "u": t(press["u"]), "v": t(press["v"]),
                "q": t(press["q"]), "z": t(press["z"])},
    metadata=Metadata(
        lat=torch.from_numpy(single.latitude.values.astype(np.float32)),
        lon=torch.from_numpy(single.longitude.values.astype(np.float32)),
        time=(datetime.datetime.fromisoformat(TIMES[-1]).replace(tzinfo=datetime.timezone.utc),),
        atmos_levels=LEVELS,
    ),
)
print("Batch 组装完成")

# ---- 4. 加载模型并自回归预测 ----
device = "cpu"
print(f"使用设备: {device}")
model = Aurora(use_lora=False)   # bf16 混合精度省显存；CPU 下该参数被忽略
model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
model.eval().to(device)
print(f"模型已加载，开始推理（CPU 每步约 2-3 分钟）...")

with torch.inference_mode():
    preds = [p.to("cpu") for p in rollout(model, batch, steps=STEPS)]
print(f"预测完成：+{STEPS * 6}h")

# ---- 5. 与 ERA5 真值比较：各时效的纬度加权全球 RMSE ----
w = np.cos(np.deg2rad(single_full.latitude.values[:720]))   # 降序纬度权重，对齐模型输出的 720 行
truth_2t = single_full.sel(time=VALID_TIME)["t2m"].values.squeeze()[:720]   # +24h 真值(最后画图用)
init = np.datetime64(TIMES[-1])
print("各时效 2米温度 全球加权 RMSE:")
np.save("preds_2t.npy", np.stack([p.surf_vars["2t"].numpy().squeeze() for p in preds]))
for i, p in enumerate(preds, start=1):      # 每步与各自有效时刻的真值比较
    vt = init + np.timedelta64(6 * i, "h")
    truth_i = single_full.sel(time=vt)["t2m"].values.squeeze()[:720]
    pred_2t = p.surf_vars["2t"].numpy().squeeze()
    bias = pred_2t - truth_i
    rmse = float(np.sqrt((bias**2 * w[:, None]).sum() / (w.sum() * bias.shape[1])))
    print(f"  +{i * 6}h ({str(vt)[:16]} UTC)  RMSE = {rmse:.3f} K")
pred_2t = preds[-1].surf_vars["2t"].numpy().squeeze()       # 用最后一步画误差图
bias = pred_2t - truth_2t

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].imshow(pred_2t, origin="upper"); axes[0].set_title(f"Aurora +{STEPS * 6}h 2t (K)")
im = axes[1].imshow(bias, origin="upper", cmap="RdBu_r", vmin=-5, vmax=5)
axes[1].set_title(f"bias vs ERA5, RMSE={rmse:.2f}K"); fig.colorbar(im, ax=axes[1])
fig.savefig(f"verify_{STEPS * 6}h.png", dpi=150, bbox_inches="tight")
print(f"图已保存: verify_{STEPS * 6}h.png")
