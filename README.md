# Aurora 0.25° 24h 预报复现

复现 Hugging Face × Earthmover 教程：ERA5 初始条件 -> aurora-0.25-pretrained 自回归 4 步 -> 与 ERA5 再分析分析场（作为验证真值代理）对比。

## 结果（2026-03-13 00UTC 起报，2 米温度全球纬度加权 RMSE）

RTX 4090，bf16 autocast：

| 时效 | RMSE |
|---|---|
| +6h  | 0.522 K |
| +12h | 0.569 K |
| +18h | 0.684 K |
| +24h | 0.742 K |

CPU fp32 参考值为 0.520 / 0.567 / 0.680 / 0.740 K，GPU bf16 与它的差异 ≤ 0.004 K。

## Phase 1 单例多变量（2026-03-13 00UTC，+24h）

| 变量 | 单位 | Aurora RMSE | 持续性 RMSE | 气候态 RMSE | Aurora ACC |
|---|---|---|---|---|---|
| 2t | K | 0.742 | 2.228 | 3.066 | 0.970 |
| msl | hPa | 0.562 | 5.128 | 7.714 | 0.997 |
| 10u | m/s | 0.834 | 3.805 | 4.445 | 0.982 |
| z500 | m | 4.886 | 56.154 | 92.960 | 0.999 |
| t850 | K | 0.647 | 3.123 | 3.876 | 0.986 |

- RMSE 为纬度加权（cos lat）；z500 为 geopotential height（geopotential / g），msl 为 hPa；`truth` 为 ERA5 分析场。
- ACC 使用 WeatherBench2 的 1990–2019 6 小时气候态（1.5°）插值到 0.25° 网格。
- 这是**单个起报个例**，ACC/RMSE 不能代替多年统计；完整逐时效表见 `results/local/summary.csv`，曲线见 `results/local/rmse_vs_lead.png`、`acc_vs_lead.png`。

## Phase 1 多起报统计（2022-01，4 个起报，+24h）

4 个起报：2022-01-01 / 01-08 / 01-15 / 01-22 00 UTC。表中为均值 ± 跨个例标准差。

| 变量 | 单位 | Aurora RMSE | persistence RMSE | climatology RMSE | Aurora ACC | persistence ACC |
|---|---|---|---|---|---|---|
| 2t | K | 0.752 ± 0.013 | 2.464 | 2.992 | 0.968 ± 0.002 | 0.652 ± 0.051 |
| msl | hPa | 0.560 ± 0.040 | 6.268 | 7.042 | 0.997 ± 0.000 | 0.606 ± 0.117 |
| 10u | m/s | 0.824 ± 0.030 | 3.950 | 3.837 | 0.977 ± 0.001 | 0.475 ± 0.071 |
| z500 | m | 5.109 ± 0.299 | 64.256 | 84.917 | 0.998 ± 0.000 | 0.707 ± 0.060 |
| t850 | K | 0.646 ± 0.015 | 3.241 | 3.624 | 0.984 ± 0.001 | 0.586 ± 0.048 |

- 数据源：Earthmover ERA5（快照 `TZ2KX6A24CJ5DG17T7MG`），4 MB/层 chunk；走代理时约 2 分钟/case。
- ACC 气候态：WeatherBench2 1990–2019 6h（1.5°）插值到 0.25°。
- 完整逐时效表：`results/em_jan/summary.csv`、`per_case.csv`、`summary.json`；曲线：`rmse_vs_lead.png`、`acc_vs_lead.png`。
- 4 个 case 都在 1 月，季节覆盖有限；更多季节/时次可用同一命令追加。

## P0 官方 2022 输出对比

官方 2022 数据集是**模型输出**，其示例使用 **HRES T0 初始场 + fine-tuned（LoRA）模型**。

- 不同配置（ERA5 IC + pretrained）在 2022-01-08T00 +24h：2t 1.140 K、msl 0.607 hPa、10u 0.748 m/s、z500 3.938 m、t850 0.676 K。
- 同配置（HRES T0 IC + `aurora-0.25-finetuned.ckpt` + `use_lora=True`）：2t 0.731 K、msl 0.365 hPa、10u 0.503 m/s、**z500 0.972 m（corr 0.99999）**、t850 0.240 K；双方对 ERA5 的 RMSE 也几乎一致（z500 4.239 vs 4.241 m）。
- **0.25° 2t 逐格对比：RMSE 0.064 K（corr 0.999994）**；1.5° 下的 2t 差异主要是重网格化造成的。
- 结论：同配置下我们的 pipeline 与官方数据集一致。详见 `results/p0_official/README.md`、`results/p0_official_matched/official_vs_ours_matched_15deg.png`。

## 环境

- Python 3.12
- CUDA 12.4 + PyTorch 2.5.1（GPU 推理需要；CPU 也能跑，但慢）
- 显存：24 GB（实测峰值约 19.9 GB；fp32 会 OOM，必须开 bf16 autocast）
- 依赖：`requirements.txt`（直接依赖）或 `requirements-lock.txt`（完整锁定）

```bash
pip install -r requirements.txt
# 或精确复现：
pip install -r requirements-lock.txt
```

## 数据与模型版本

- HF：`microsoft/aurora@a96afd7ee6d65e3bd2d476f3be798a25a56f2296`（`aurora-0.25-pretrained.ckpt` + `aurora-0.25-static.pickle`）
- Earthmover ERA5：快照 `TZ2KX6A24CJ5DG17T7MG`（2026-10-28 的 main）
- WeatherBench2（备选数据源 + 气候态）：`weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr`
- 气候态：`weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_...zarr`
- 验证口径：所有 RMSE/ACC 的 `truth` 都是 **ERA5 再分析分析场**（初始场也来自 ERA5），不是站点/卫星原始观测
- 数据更新后可给 `fetch_data.py` 传 `--snapshot main`
- 旧脚本下载的 NetCDF 没有 `source_snapshot` 属性，`results.json` 会记为 `unknown`；重新运行 `fetch_data.py` 后会写入实际快照

## 运行

### 单例 24h 预报

```bash
# 1) 下载 ERA5（首次约 650 MB；之后复用缓存）
python -u fetch_data.py --init-time 2026-03-13T00:00:00 --lead-hours 24 --out .

# 2) GPU 推理 + 检验
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u run.py \
  --init-time 2026-03-13T00:00:00 --lead-hours 24 --device auto --out .
```

- `--lead-hours` 必须是 6 的倍数；所有验证时刻由起报时刻自动推导，不再需要手改 `VALID_TIME`。
- `--out` 在 fetch 中是 NetCDF 目录，在 run 中是预测/图/指标目录。
- `--data-dir` 指定 run.py 读取 NetCDF 的目录（默认当前目录）。
- `--device auto|cuda|cpu`；`auto` 优先 CUDA。
- run.py 会在 GPU 上自动开启 backbone/encoder/decoder 的 bf16 autocast，并设置 `PYTORCH_CUDA_ALLOC_CONF`；命令里显式写出只是为了排查内存问题时更直观。

### 多变量 / 多时次评估

```bash
# 1) 本地个例：补抓 Z500/T850 真值（8 个 4 MB chunk）
python -u fetch_truth.py --init-time 2026-03-13T00:00:00 --lead-hours 24 --out cache

# 2) 单例五变量 + ACC + 基线
python -u evaluate.py --source local \
  --init-times 2026-03-13T00:00:00 --lead-hours 24 \
  --device auto --climatology wb2 --out results/local

# 3) 多起报时次（Earthmover，4 MB/层 chunk；可用代理加速）
export http_proxy=http://<proxy-host>:<port>
export https_proxy=http://<proxy-host>:<port>
python -u evaluate.py --source earthmover \
  --init-times 2022-01-01T00:00:00,2022-04-01T00:00:00,2022-07-01T00:00:00,2022-10-01T00:00:00 \
  --lead-hours 24 --device auto --climatology wb2 --workers 8 --out results/em
```

- `--source earthmover` 按 4 MB/层读取，`--workers` 控制并发；首次为每个 case 下载约 150 个 chunk（约 600 MB），之后复用 `cache/em_*.npz`。
- `--source wb2` 是另一条路径：WeatherBench2 ERA5（1959-01-01 至 2023-01-10），压力场是整块读取；2026-10-28 直连 GCS 实测 2–4 MB/s，54 MB chunk 可正常读取。
- 两种 source 都可以用 `--dry-run` 先检查时次/索引/缓存路径，不下载。
- 气候态为 1990–2019；跨季节多 case 时，1.5° 气候态会一次性读取所需日期的 chunk。
- `evaluate.py` 的指标：`summary.csv`（按变量/时效/方法聚合）、`per_case.csv`（逐 case）、`summary.json`、`rmse_vs_lead.png`、`acc_vs_lead.png`。
- 基线：persistence（初始场保持不变）与气候态；ACC 只对 Aurora 和 persistence 计算（气候态自身异常为 0）。

### Paper 口径（HRES T0 + fine-tuned + 10 天）

```bash
# fine-tuned checkpoint（约 5 GB，LoRA）需先下载到本地
python -u evaluate.py --source paper \
  --init-times 2022-01-02T00:00:00,2022-01-02T12:00:00,2022-01-16T00:00:00,2022-01-16T12:00:00,2022-04-01T00:00:00,2022-04-01T12:00:00,2022-04-15T00:00:00,2022-04-15T12:00:00,2022-07-01T00:00:00,2022-07-01T12:00:00,2022-07-15T00:00:00,2022-07-15T12:00:00,2022-10-01T00:00:00,2022-10-01T12:00:00,2022-10-15T00:00:00,2022-10-15T12:00:00 \
  --lead-hours 240 --workers 10 \
  --checkpoint-path /path/to/aurora-0.25-finetuned.ckpt \
  --climatology wb2 --no-clim-cache --out results/paper_2022_4season_10d
```

- `--source paper` = **HRES T0 IC + WB2 ERA5 truth + `aurora-0.25-finetuned.ckpt`（自动 `use_lora=True`）**，与官方 2022 数据集同配置；P0 已校验：z500 0.97 m、0.25° 2t 0.064 K。
- 16 个 case：4 季 × 2 日 × 00/12 UTC，10 天（40 个 6h 时效）。
- 首次运行会下载 HRES T0 IC（约 570 MB/case）和 WB2 ERA5 truth（实际下载约 1.6–2 GB/case，缓存 npz 约 0.83 GB）；truth 缓存为 `cache/wb2truth_*.npz`，中断后可续跑。
- `--no-clim-cache` 避免 10 天 × 多 case 产生大量逐时次气候态文件；ACC 按 case 现算。
- 数据量大，建议后台运行；本次 16 case 实测约 **1 小时 45 分钟**（直连/代理恢复后约 3–5 分钟/case）。

**完整 0.25° 结果**（2026-10-28，16 case × 4 季 × 10 天；HRES T0 IC + fine-tuned LoRA，ERA5 0.25° 真值，纬度加权）：

| 变量 | 单位 | +24h RMSE / ACC | +120h RMSE / ACC | +240h RMSE / ACC | persistence +240h | climatology +240h |
|---|---|---|---|---|---|---|
| 2t | K | 1.035 ± 0.088 / 0.928 | 1.480 ± 0.116 / 0.842 | 2.495 ± 0.245 / 0.471 | 3.482 | 2.620 |
| msl | hPa | 0.625 ± 0.028 / 0.996 | 2.681 ± 0.266 / 0.923 | 6.313 ± 0.662 / 0.468 | 9.714 | 7.083 |
| 10u | m/s | 0.893 ± 0.015 / 0.973 | 2.170 ± 0.075 / 0.824 | 3.641 ± 0.150 / 0.400 | 5.593 | 3.918 |
| z500 | m | 4.489 ± 0.247 / 0.998 | 26.584 ± 2.338 / 0.946 | 69.792 ± 7.059 / 0.519 | 113.840 | 79.439 |
| t850 | K | 0.824 ± 0.044 / 0.973 | 1.581 ± 0.069 / 0.889 | 3.094 ± 0.225 / 0.452 | 4.780 | 3.334 |

2t 分季节（RMSE / ACC）：

| 季节 | +24h | +120h | +240h |
|---|---|---|---|
| DJF | 1.172 / 0.925 | 1.653 / 0.838 | 2.794 / 0.435 |
| MAM | 1.030 / 0.933 | 1.441 / 0.849 | 2.514 / 0.451 |
| JJA | 0.977 / 0.925 | 1.469 / 0.822 | 2.405 / 0.452 |
| SON | 0.960 / 0.928 | 1.356 / 0.858 | 2.268 / 0.544 |

- 输出：`results/paper_2022_4season_10d/`（summary.csv、per_case.csv、summary.json、rmse_vs_lead.png、acc_vs_lead.png）。
- 与官方 1.5° 同日期基准交叉核对：官方 2t +24h / +240h = 0.850 / 2.370 K，msl = 0.568 / 6.265 hPa，10u = 0.706 / 3.534 m/s。我们的 0.25° 值略高是网格更细、未经过 1.5° 保守重网格平滑所致，季节排序一致。

**网络（2026-10-28 更新）**：直连 GCS 已恢复，12 MB 与 54 MB（压缩后约 24.6 MB）chunk 实测 2.7–3.9 MB/s，0.25° paper 全流程已跑通。若直连不稳，可用局域网代理 `http://10.25.210.207:7897`，但代理依赖本机开机。

官方 1.5° 2022 数据集的 10 天基准命令（备用）：

```bash
python -u evaluate_official_15deg.py \
  --init-times 2022-01-02T00:00:00,2022-01-02T12:00:00,2022-01-16T00:00:00,2022-01-16T12:00:00,2022-04-01T00:00:00,2022-04-01T12:00:00,2022-04-15T00:00:00,2022-04-15T12:00:00,2022-07-01T00:00:00,2022-07-01T12:00:00,2022-07-15T00:00:00,2022-07-15T12:00:00,2022-10-01T00:00:00,2022-10-01T12:00:00,2022-10-15T00:00:00,2022-10-15T12:00:00 \
  --lead-hours 240 --climatology none --workers 8 \
  --out results/paper_official15_4season_10d
```

- 官方 1.5° 2022 数据集是 **HRES T0 + fine-tuned** 模型输出；P0 已验证同配置 0.25° +24h 与我们的输出一致（2t 0.064 K）。
- 真值为 ERA5 1.5°；16 case × 40 个时效。本次先跑了地面变量；直连恢复后可直接去掉 `--surface-only`，并把 `--climatology none` 改成 `wb2`，补齐 Z500/T850 与 ACC。

**已跑通的地面变量结果**（`--surface-only`，16 case × 4 季 × 10 天，官方 1.5° 预报 vs ERA5 1.5° 真值）：

| 变量 | 单位 | +24h | +120h | +240h | persistence +240h |
|---|---|---|---|---|---|
| 2t | K | 0.850 ± 0.081 | 1.330 ± 0.110 | 2.370 ± 0.243 | 3.345 |
| msl | hPa | 0.568 ± 0.024 | 2.651 ± 0.266 | 6.265 ± 0.657 | 9.695 |
| 10u | m/s | 0.706 ± 0.016 | 2.035 ± 0.078 | 3.534 ± 0.159 | 5.394 |

- 16 个 case：2022-01-02/16、04-01/15、07-01/15、10-01/15，00/12 UTC。
- 输出：`results/paper_official15_surface_4season_10d/`（summary.csv、per_case.csv、rmse_vs_lead.png）。
- **Z500/T850 与 ACC 待补**：本次只跑了地面变量。直连 GCS 已恢复（12 MB/41.8 MB chunk 均可读），重跑时去掉 `--surface-only` 并加 `--climatology wb2` 即可。

## 硬件与耗时

| 设备 | 每步 | 4 步 24h 预报 |
|---|---|---|
| RTX 4090（bf16 autocast） | ~1.8 s | ~7 s（模型加载另约 16 s） |
| CPU | ~2–3 min | ~8–12 min |

显存峰值约 19.9 GB；fp32 在 24 GB 卡上会 OOM。

Earthmover 多时次数据：约 150 个 4 MB chunk/case；8 workers 下载约 1–3 分钟/case（代理速度良好时）。

## 输出文件

- `era5_single.nc` / `era5_press.nc` — ERA5 缓存（gitignore）
- `preds_2t.npy` — 各时效 2t 预测，形状 `(steps, 720, 1440)`
- `results.json` — 单例 2t 的起报时刻、模型/数据版本、RMSE、耗时
- `verify_{lead}h.png` — 最后一个时效的预测与偏差图
- `results/local/` — Phase 1 单例五变量指标与曲线
- `results/em_jan/` — 本次 4 起报评估输出（README 命令写到 `results/em/`，目录名由 `--out` 决定）
- `results/p0_official/` — 与官方 2022 数据集的场对比（报告、指标、图）
- `results/p0_official_matched/` — HRES T0 + fine-tuned 同配置对比
- `results/paper_2022_4season_10d/` — paper 口径 16 case × 10 天完整评估（0.25°，已完成）
- `results/paper_official15_4season_10d/` — 官方 1.5° 2022 数据集的 4 季 × 10 天基准命令（5 变量；本次未完整跑，地面变量见下）
- `results/paper_official15_surface_4season_10d/` — 已跑通的地面变量 4 季 × 10 天基准（2t/msl/10u）
- `results/wb2/` — WeatherBench2 多时次评估输出（备选路径）
- `bias_evolution.png` — 早期手工生成的四时效偏差图（`evaluate.py` 生成的是逐时效 RMSE/ACC 曲线，不生成该偏差图）

## 文件

- `config.py` — 共享配置、CLI 参数、时间推导
- `fetch_data.py` — 从 Earthmover 拉取 ERA5 并缓存
- `fetch_truth.py` — 补抓 Z500/T850 真值（local 数据源）
- `aurora_utils.py` — 模型加载、Batch 组装、rollout
- `metrics.py` — 纬度加权 RMSE / bias / ACC
- `sources.py` — local / Earthmover / WeatherBench2 数据源、气候态读取
- `evaluate.py` — 多变量/多时次评估 harness
- `compare_official.py` — 与官方 Aurora 2022 数据集的场对比（P0）
- `compare_official_matched.py` — HRES T0 + fine-tuned 同配置 P0 对比
- `evaluate_official_15deg.py` — 官方 1.5° 2022 数据的 4 季 × 10 天基准
- `run.py` — 单例 2t 快速验证与画图
- `requirements.txt` / `requirements-lock.txt`

数据缓存（`*.nc`）、预测数组（`*.npy`、`*.npz`）和下载缓存（`cache/`）不入库，可由脚本重新生成。

## 已知限制

- Phase 1 目前有 2026-03-13 单例和 2022-01 四个起报；季节覆盖仍有限，更多时次可用 `--source earthmover` 追加。
- Earthmover 直连时单 chunk 可能要几十秒；设置 `http_proxy`/`https_proxy` 后可到 1–5 s/chunk。
- ACC 使用 1.5° 气候态插值到 0.25°，与直接在 0.25° 气候态上计算不完全等价。
- 1.5° 气候态插值会平滑小尺度异常，单例 ACC 偏高（0.97–0.999），多时次统计会更有参考价值。
- `--snapshot` 固定的 Earthmover 快照若被清理，可改用 `--snapshot main`。
