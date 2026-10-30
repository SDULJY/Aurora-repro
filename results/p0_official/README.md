# P0：官方 Aurora 2022 输出一致性检查

## 设置

- 起报：2022-01-08T00 UTC，+24h
- 我们的配置：ERA5 IC（Earthmover）+ `aurora-0.25-pretrained.ckpt`
- 官方数据：`weatherbench2/datasets/aurora/2022-240x121_equiangular_with_poles_conservative.zarr`
  - `time_idx=13`（2022-01-08T00），`lead_idx=3`（+24h）
  - 官方 2022 数据集是**模型输出**，不是 ERA5 真值
- 官方 Aurora 示例（HRES T0）使用 **HRES T0 初始场 + fine-tuned 模型**；因此这次对比**不是同配置**对比

## 结果（1.5° 网格，纬度加权）

| 变量 | 单位 | ours-vs-official RMSE | corr | ours-vs-ERA5 | official-vs-ERA5 |
|---|---|---|---|---|---|
| 2t | K | 1.140 | 0.99818 | 0.738 | 1.245 |
| msl | hPa | 0.607 | 0.99886 | 0.522 | 0.639 |
| 10u | m/s | 0.748 | 0.98683 | 0.784 | 0.901 |
| z500 | m | 3.938 | 0.99992 | 4.518 | 4.241 |
| t850 | K | 0.676 | 0.99876 | 0.623 | 0.805 |

图：`official_vs_ours_15deg.png`（左：我们，中：官方，右：差值）。原始场在 `fields.npz`，机器可读指标在 `comparison.json`。

## 隔离 checkpoint：fine-tuned + 相同 ERA5 IC

为了区分 checkpoint 和 IC 的影响，保持同一个 ERA5 IC，只把 checkpoint 换成 `aurora-0.25-finetuned.ckpt`（LoRA，需 `use_lora=True`）：

| 变量 | 单位 | fine-tuned vs official | corr | fine-tuned vs ERA5 |
|---|---|---|---|---|
| 2t | K | 0.855 | 0.99891 | 0.972 |
| msl | hPa | 0.501 | 0.99926 | 0.554 |
| 10u | m/s | 0.666 | 0.98946 | 0.812 |
| z500 | m | 2.967 | 0.99995 | 4.157 |
| t850 | K | 0.530 | 0.99924 | 0.647 |

与 pretrained 的 2t 1.140 K / z500 3.938 m 相比，fine-tuned 明显更接近官方，说明 **checkpoint 是差异的主要来源之一**。但 fine-tuned 模型在 ERA5 IC/真值上反而更差（2t 0.972 K vs pretrained 0.738 K），符合它专门用 HRES T0 微调、与 ERA5 分布不一致的预期。剩余差异应主要来自 **HRES T0 vs ERA5 初始场**。

## 同配置：HRES T0 + fine-tuned（LoRA）

用官方 HRES T0 示例的配置重跑：`hres_t0/2016-2022-6h-1440x721.zarr` IC + `aurora-0.25-finetuned.ckpt`（`use_lora=True`），同一个 2022-01-08T00 +24h：

| 变量 | 单位 | ours-vs-official RMSE | corr | ours-vs-ERA5 | official-vs-ERA5 |
|---|---|---|---|---|---|
| 2t | K | 0.731 | 0.99935 | 1.182 | 1.245 |
| msl | hPa | 0.365 | 0.99967 | 0.612 | 0.639 |
| 10u | m/s | 0.503 | 0.99218 | 0.865 | 0.901 |
| z500 | m | 0.972 | 0.99999 | 4.239 | 4.241 |
| t850 | K | 0.240 | 0.99985 | 0.787 | 0.805 |

- **z500 只差 0.97 m，corr 0.99999；t850 差 0.24 K；msl 差 0.36 hPa。** 双方对 ERA5 的 RMSE 也几乎相同（z500 4.239 vs 4.241 m），说明同配置下我们的 pipeline 和官方数据集一致。
- 2t/10u 的差异更大（0.73 K / 0.50 m/s），差值图集中在海岸线和山地，符合 **0.25° → 1.5° 重网格化**（官方 conservative coarsening vs 我们的双线性插值）造成的差异，而不是模型/IC 不一致。
- **0.25° 2t 逐格对比：RMSE 0.064 K，bias +0.006 K，corr 0.999994**；1.5° 下的 0.731 K 确实主要是重网格化误差。
- 同配置的场图在 `results/p0_official_matched/official_vs_ours_matched_15deg.png`。
- 这也解释了为什么 pretrained+ERA5 看起来“更好”：它用 ERA5 IC 且预训练在 ERA5 上，对 ERA5 真值有优势；fine-tuned 模型专门适配 HRES T0，和官方 2022 数据集才是同配置。

## 结论

- **同配置（HRES T0 + fine-tuned LoRA）下，我们的输出与官方数据集一致**：z500 0.972 m（corr 0.99999）、t850 0.240 K、msl 0.365 hPa、10u 0.503 m/s、2t 0.731 K；双方对 ERA5 的 RMSE 也几乎相同（z500 4.239 vs 4.241 m，t850 0.787 vs 0.805 K）。这说明 IC/预处理/rollout/单位/层次/时效都对上了。
- 2t/10u 的剩余差异集中在海岸线和山地，符合 0.25°→1.5° 重网格化造成的差异；**0.25° 2t 逐格 RMSE 仅 0.064 K**，进一步确认了这一点。
- 不同配置（ERA5 IC + pretrained）与官方数据集的差异是预期的：官方数据集是 **HRES T0 + fine-tuned** 的模型输出，不是 ERA5 真值，也不是同配置基准。
- 在这个 case 上，pretrained+ERA5 对照 ERA5 的 RMSE 反而优于官方（2t 0.738 vs 1.245 K），因为 pretrained 预训练在 ERA5 上、且用 ERA5 IC；这不代表模型更强，只是评估口径不同。
- 因此：**HF 教程复现（ERA5 + pretrained）和官方 2022 数据集（HRES T0 + fine-tuned）是两条不同的配置**；同配置校验已通过。

## 复现命令

```bash
# 使用缓存的 2022-01-08T00 ERA5 case
python compare_official.py \
  --init-time 2022-01-08T00:00:00 --lead-hours 24 \
  --checkpoint aurora-0.25-pretrained.ckpt \
  --out results/p0_official
```

如需 fine-tuned checkpoint + 本地文件（仍用 ERA5 IC，用于隔离 checkpoint）：

```bash
python compare_official.py \
  --init-time 2022-01-08T00:00:00 --lead-hours 24 \
  --checkpoint aurora-0.25-finetuned.ckpt \
  --checkpoint-path /path/to/aurora-0.25-finetuned.ckpt --use-lora \
  --out results/p0_official_finetuned
```

同配置（HRES T0 IC + fine-tuned LoRA）：

```bash
python compare_official_matched.py \
  --init-time 2022-01-08T00:00:00 --lead-hours 24 \
  --checkpoint aurora-0.25-finetuned.ckpt \
  --checkpoint-path /path/to/aurora-0.25-finetuned.ckpt --use-lora \
  --workers 10 --strict-025-2t --out results/p0_official_matched
```
