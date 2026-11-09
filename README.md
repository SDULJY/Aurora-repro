# Aurora 0.25° 全球天气预报复现

> A reproducible benchmark of Microsoft Aurora 0.25° global weather forecasting.

复现 Microsoft Aurora 0.25° 全球天气预报模型，覆盖：

- **两种配置**：`pretrained + ERA5 IC`，以及论文/官方示例使用的 `fine-tuned LoRA + HRES T0 IC`
- **五个变量**：2t、msl、10u、z500、t850
- **多起报、多季节、10 天时效**：RMSE / bias / ACC，并与 persistence、climatology 基线对比
- **官方一致性校验**：与 WeatherBench2 的 Aurora 2022 数据集逐场对比

> 验证真值统一使用 **ERA5 再分析分析场**（用作观测代理），不是站点或卫星原始观测。

## 结果速览

### 1. 官方同配置 10 天基准（0.25°）

**配置**：HRES T0 初始场 + `aurora-0.25-finetuned.ckpt`（LoRA）+ ERA5 0.25° 真值。

**样本**：2022 年 4 季 × 每季 2 个日期 × 00/12 UTC，共 **16 个起报**；时效 +6h 至 +240h，每 6h 一个，共 40 个时效。表中为跨 16 个 case 的均值 ± 标准差；RMSE 为纬度加权。

| 变量 | 单位 | +24h RMSE / ACC | +120h RMSE / ACC | +240h RMSE / ACC | persistence +240h RMSE | climatology +240h RMSE |
|---|---|---|---|---|---|---|
| 2t | K | **1.035 ± 0.088 / 0.928** | 1.480 ± 0.116 / 0.842 | 2.495 ± 0.245 / 0.471 | 3.482 | 2.620 |
| msl | hPa | **0.625 ± 0.028 / 0.996** | 2.681 ± 0.266 / 0.923 | 6.313 ± 0.662 / 0.468 | 9.714 | 7.083 |
| 10u | m/s | **0.893 ± 0.015 / 0.973** | 2.170 ± 0.075 / 0.824 | 3.641 ± 0.150 / 0.400 | 5.593 | 3.918 |
| z500 | m | **4.489 ± 0.247 / 0.998** | 26.584 ± 2.338 / 0.946 | 69.792 ± 7.059 / 0.519 | 113.840 | 79.439 |
| t850 | K | **0.824 ± 0.044 / 0.973** | 1.581 ± 0.069 / 0.889 | 3.094 ± 0.225 / 0.452 | 4.780 | 3.334 |

2t 分季节（RMSE / ACC）：

| 季节 | +24h | +120h | +240h |
|---|---|---|---|
| DJF | 1.172 / 0.925 | 1.653 / 0.838 | 2.794 / 0.435 |
| MAM | 1.030 / 0.933 | 1.441 / 0.849 | 2.514 / 0.451 |
| JJA | 0.977 / 0.925 | 1.469 / 0.822 | 2.405 / 0.452 |
| SON | 0.960 / 0.928 | 1.356 / 0.858 | 2.268 / 0.544 |

完整逐时效、逐 case 结果：

- [`results/paper_2022_4season_10d/summary.csv`](results/paper_2022_4season_10d/summary.csv)
- [`results/paper_2022_4season_10d/per_case.csv`](results/paper_2022_4season_10d/per_case.csv)
- [`results/paper_2022_4season_10d/rmse_vs_lead.png`](results/paper_2022_4season_10d/rmse_vs_lead.png)
- [`results/paper_2022_4season_10d/acc_vs_lead.png`](results/paper_2022_4season_10d/acc_vs_lead.png)

### 2. 与官方 Aurora 2022 数据集的一致性校验

官方 2022 数据集是**模型输出**，不是 ERA5 真值；其示例配置为 **HRES T0 初始场 + fine-tuned 模型**。我们用同一配置、同一初始时刻 `2022-01-08T00`、同一时效 `+24h`，在 1.5° 网格上做逐场对比：

| 变量 | 单位 | 我们 vs 官方 RMSE | 我们 vs ERA5 | 官方 vs ERA5 |
|---|---|---|---|---|
| 2t | K | 0.731 | 1.182 | 1.245 |
| msl | hPa | 0.365 | 0.612 | 0.639 |
| 10u | m/s | 0.503 | 0.865 | 0.901 |
| z500 | m | **0.972** | 4.239 | 4.241 |
| t850 | K | **0.240** | 0.787 | 0.805 |

- **z500 只差 0.972 m（corr 0.99999）**，t850 差 0.240 K，msl 差 0.365 hPa，说明 IC、预处理、rollout、单位、层次和时效都对上了。
- 0.25° 网格上 2t 的逐格 RMSE 仅 **0.064 K**（corr 0.999994）；1.5° 下 2t/10u 的剩余差异主要来自 0.25° → 1.5° 重网格化。
- 详细报告：[`results/p0_official/README.md`](results/p0_official/README.md)；场图：[`results/p0_official_matched/official_vs_ours_matched_15deg.png`](results/p0_official_matched/official_vs_ours_matched_15deg.png)。

### 3. 官方 1.5° 2022 数据集的 4 季 × 10 天基准

用官方 1.5° 预报与 ERA5 1.5° 真值做同日期交叉核对（16 个 case；官方数据同为 HRES T0 + fine-tuned 配置）：

| 变量 | 单位 | +24h RMSE | +120h RMSE | +240h RMSE | persistence +240h RMSE |
|---|---|---|---|---|---|
| 2t | K | 0.850 ± 0.081 | 1.330 ± 0.110 | 2.370 ± 0.243 | 3.345 |
| msl | hPa | 0.568 ± 0.024 | 2.651 ± 0.266 | 6.265 ± 0.657 | 9.695 |
| 10u | m/s | 0.706 ± 0.016 | 2.035 ± 0.078 | 3.534 ± 0.159 | 5.394 |

我们的 0.25° 结果略高于官方 1.5°，符合预期：0.25° 保留了更多小尺度结构，而官方数据经过 1.5° 保守重网格平滑；两者的季节排序和变量间相对关系一致。

<details>
<summary><strong>早期单例与 2022-01 验证结果</strong>（点击展开）</summary>

#### 单例 24h（2026-03-13 00 UTC，2t）

RTX 4090，bf16 autocast；CPU fp32 参考值为 0.520 / 0.567 / 0.680 / 0.740 K。

| 时效 | +6h | +12h | +18h | +24h |
|---|---|---|---|---|
| RMSE | 0.522 K | 0.569 K | 0.684 K | 0.742 K |

#### 单例五变量 +24h

| 变量 | 单位 | Aurora RMSE | persistence | climatology | Aurora ACC |
|---|---|---|---|---|---|
| 2t | K | 0.742 | 2.228 | 3.066 | 0.970 |
| msl | hPa | 0.562 | 5.128 | 7.714 | 0.997 |
| 10u | m/s | 0.834 | 3.805 | 4.445 | 0.982 |
| z500 | m | 4.886 | 56.154 | 92.960 | 0.999 |
| t850 | K | 0.647 | 3.123 | 3.876 | 0.986 |

#### 2022-01 四个起报 +24h

起报：2022-01-01 / 01-08 / 01-15 / 01-22 00 UTC；均值 ± 跨个例标准差。

| 变量 | 单位 | Aurora RMSE | persistence | climatology | Aurora ACC | persistence ACC |
|---|---|---|---|---|---|---|
| 2t | K | 0.752 ± 0.013 | 2.464 | 2.992 | 0.968 ± 0.002 | 0.652 ± 0.051 |
| msl | hPa | 0.560 ± 0.040 | 6.268 | 7.042 | 0.997 ± 0.000 | 0.606 ± 0.117 |
| 10u | m/s | 0.824 ± 0.030 | 3.950 | 3.837 | 0.977 ± 0.001 | 0.475 ± 0.071 |
| z500 | m | 5.109 ± 0.299 | 64.256 | 84.917 | 0.998 ± 0.000 | 0.707 ± 0.060 |
| t850 | K | 0.646 ± 0.015 | 3.241 | 3.624 | 0.984 ± 0.001 | 0.586 ± 0.048 |

结果目录：`results/local/`、`results/em_jan/`。

</details>

## 快速开始

### 环境

| 项目 | 要求 |
|---|---|
| Python | 3.12 |
| CUDA / PyTorch | CUDA 12.4 + PyTorch 2.5.1 |
| GPU 显存 | 24 GB（实测峰值约 19.9 GB；fp32 会 OOM，需 bf16 autocast） |
| 其他 | CPU 也可运行，但速度慢很多 |

```bash
pip install -r requirements.txt
# 或精确复现：
pip install -r requirements-lock.txt
```

### 数据与模型版本

| 资源 | 版本 / 路径 |
|---|---|
| Aurora 模型 | `microsoft/aurora@a96afd7ee6d65e3bd2d476f3be798a25a56f2296` |
| pretrained checkpoint | `aurora-0.25-pretrained.ckpt` |
| fine-tuned checkpoint | `aurora-0.25-finetuned.ckpt`（LoRA，约 5 GB，需单独下载） |
| 静态场 | `aurora-0.25-static.pickle` |
| Earthmover ERA5 | 快照 `TZ2KX6A24CJ5DG17T7MG` |
| ERA5 0.25°（WB2） | `weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr` |
| HRES T0 0.25° | `weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr` |
| 气候态 | `weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_...zarr` |

- 验证真值统一为 **ERA5 再分析分析场**，不是站点观测。
- Earthmover 快照更新后，`fetch_data.py` 可传 `--snapshot main`。
- 数据、预测数组、checkpoint 和下载缓存均不入库，可由脚本重新生成。

### 1. 单例 24h 预报

```bash
# 下载 ERA5（首次约 650 MB；之后复用缓存）
python -u fetch_data.py \
  --init-time 2026-03-13T00:00:00 --lead-hours 24 --out .

# GPU 推理 + 2t 检验
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u run.py \
  --init-time 2026-03-13T00:00:00 --lead-hours 24 --device auto --out .
```

- `--lead-hours` 必须是 6 的倍数，验证时刻由起报时刻自动推导。
- `--device auto|cuda|cpu`；`auto` 优先 CUDA。
- GPU 上自动开启 bf16 autocast；命令中的 `PYTORCH_CUDA_ALLOC_CONF` 用于降低显存碎片。

### 2. 多变量 / 多时次评估

```bash
# 补抓 Z500/T850 真值（local 数据源）
python -u fetch_truth.py \
  --init-time 2026-03-13T00:00:00 --lead-hours 24 --out cache

# 单例五变量 + ACC + 基线
python -u evaluate.py --source local \
  --init-times 2026-03-13T00:00:00 --lead-hours 24 \
  --device auto --climatology wb2 --out results/local

# 多起报时次（Earthmover，4 MB/层 chunk）
# 如需代理，请先设置自己的 http_proxy / https_proxy
python -u evaluate.py --source earthmover \
  --init-times 2022-01-01T00:00:00,2022-04-01T00:00:00,2022-07-01T00:00:00,2022-10-01T00:00:00 \
  --lead-hours 24 --device auto --climatology wb2 --workers 8 --out results/em
```

- `--source earthmover` 按 4 MB/层读取，`--workers` 控制并发；每个 case 约 150 个 chunk，首次下载约 600 MB，之后复用 `cache/em_*.npz`。
- `--source wb2` 直接读 WeatherBench2 ERA5；压力场按整块读取，适合 GCS 访问较快的机器。
- `--source paper` 使用 HRES T0 IC + WB2 ERA5 真值 + fine-tuned LoRA，见下一节。
- 各数据源都支持 `--dry-run`，可先检查时次、索引和缓存路径而不下载。

### 3. Paper 口径：16 个起报 × 10 天

```bash
python -u evaluate.py --source paper \
  --init-times 2022-01-02T00:00:00,2022-01-02T12:00:00,2022-01-16T00:00:00,2022-01-16T12:00:00,2022-04-01T00:00:00,2022-04-01T12:00:00,2022-04-15T00:00:00,2022-04-15T12:00:00,2022-07-01T00:00:00,2022-07-01T12:00:00,2022-07-15T00:00:00,2022-07-15T12:00:00,2022-10-01T00:00:00,2022-10-01T12:00:00,2022-10-15T00:00:00,2022-10-15T12:00:00 \
  --lead-hours 240 --workers 10 \
  --checkpoint-path /path/to/aurora-0.25-finetuned.ckpt \
  --climatology wb2 --no-clim-cache \
  --out results/paper_2022_4season_10d
```

- `--source paper` 会自动启用 `use_lora=True`，与官方 2022 数据集同配置。
- 首次运行会下载 HRES T0 IC（约 570 MB/case）和 WB2 ERA5 真值（约 1.6–2 GB/case）；真值缓存为 `cache/wb2truth_*.npz`，中断后可续跑。
- `--no-clim-cache` 避免 10 天 × 多 case 产生大量逐时次气候态文件；ACC 按 case 现算。
- 本次 16 个 case 实测约 **1 小时 45 分钟**（网络恢复后约 3–5 分钟/case）。

### 4. 官方 1.5° 2022 数据集基准（可选）

```bash
python -u evaluate_official_15deg.py \
  --init-times 2022-01-02T00:00:00,2022-01-02T12:00:00,2022-01-16T00:00:00,2022-01-16T12:00:00,2022-04-01T00:00:00,2022-04-01T12:00:00,2022-04-15T00:00:00,2022-04-15T12:00:00,2022-07-01T00:00:00,2022-07-01T12:00:00,2022-07-15T00:00:00,2022-07-15T12:00:00,2022-10-01T00:00:00,2022-10-01T12:00:00,2022-10-15T00:00:00,2022-10-15T12:00:00 \
  --lead-hours 240 --workers 8 \
  --out results/paper_official15_4season_10d
```

- 该脚本读取官方 1.5° 预报和 ERA5 1.5° 真值；默认包含 Z500/T850 和 ACC。
- 如果只想先跑地面变量，可加 `--surface-only`；本次已跑通的结果在 `results/paper_official15_surface_4season_10d/`。

## 评估口径

- **RMSE**：全球纬度加权，权重 ∝ cos(lat)。
- **ACC**：相对 1990–2019 6 小时气候态（1.5°）的异常相关系数；气候态双线性插值到 0.25°。
- **z500**：位势高度，单位 m（geopotential / g）；**msl**：hPa；**2t / t850**：K；**10u**：m/s。
- **基线**：persistence（初始场保持不变）与 climatology（同日历时气候态）。
- **样本**：16 个 case = 4 季 × 2 个日期 × 00/12 UTC；每个 case 40 个时效（+6h 至 +240h）。
- **真值**：ERA5 再分析分析场；初始场在 paper 配置中来自 HRES T0，在 pretrained 配置中来自 ERA5。

## 结果文件

| 目录 | 内容 |
|---|---|
| `results/paper_2022_4season_10d/` | 主结果：16 case × 10 天，5 变量，0.25° |
| `results/paper_official15_surface_4season_10d/` | 官方 1.5° 预报的地面变量基准（2t/msl/10u） |
| `results/p0_official/` | 官方 2022 数据集一致性报告与场图 |
| `results/p0_official_matched/` | HRES T0 + fine-tuned 同配置对比 |
| `results/em_jan/` | 2022-01 四个起报的 Phase 1 结果 |
| `results/local/` | 2026-03-13 单例五变量结果 |

每个结果目录通常包含：

- `summary.csv` — 按变量 / 时效 / 方法聚合
- `per_case.csv` — 逐 case 明细
- `summary.json` — 可机读的完整指标与配置
- `rmse_vs_lead.png`、`acc_vs_lead.png` — 逐时效曲线

## 项目结构

| 文件 | 作用 |
|---|---|
| `config.py` | 共享配置、CLI 参数、时间推导 |
| `fetch_data.py` | 从 Earthmover 拉取 ERA5 并缓存 |
| `fetch_truth.py` | 补抓 Z500/T850 真值（local 数据源） |
| `aurora_utils.py` | 模型加载、Batch 组装、rollout |
| `metrics.py` | 纬度加权 RMSE / bias / ACC |
| `sources.py` | local / Earthmover / WeatherBench2 数据源、气候态读取 |
| `evaluate.py` | 多变量、多时次评估 harness |
| `compare_official.py` | 与官方 Aurora 2022 数据集的场对比（P0） |
| `compare_official_matched.py` | HRES T0 + fine-tuned 同配置 P0 对比 |
| `evaluate_official_15deg.py` | 官方 1.5° 2022 数据的 4 季 × 10 天基准 |
| `run.py` | 单例 2t 快速验证与画图 |
| `requirements.txt` / `requirements-lock.txt` | 直接依赖 / 完整锁定版本 |

## 已知限制

- **真值口径**：ERA5 是再分析产品，不是站点/卫星观测；结论应表述为“相对 ERA5 的误差”，不是“相对真实观测的误差”。
- **样本规模**：16 个 case 覆盖 2022 年四季，但不是多年气候统计；表中的标准差只反映跨 case 差异，不能替代多年显著性检验。
- **ACC 气候态**：使用 1.5° 气候态插值到 0.25°，与直接在 0.25° 气候态上计算不完全等价，单例 ACC 可能偏高。
- **官方 1.5° 基准**：当前只跑了地面变量（2t/msl/10u）；Z500/T850 与 ACC 可按第 4 节命令补齐。
- **显存**：fp32 在 24 GB 卡上会 OOM；需要 bf16 autocast 和 `expandable_segments`。
- **数据快照**：Earthmover 快照若被清理，改用 `--snapshot main`；不同快照的数值可能有细微差异。

## 数据来源与致谢

- **Microsoft Aurora**：模型代码与权重来自 [`microsoft/aurora`](https://github.com/microsoft/aurora)（MIT License；具体以官方仓库/模型卡为准），通过 Hugging Face 固定 revision 使用。
- **ERA5**：Copernicus Climate Change Service 再分析数据；经 Earthmover Arraylake 与 WeatherBench2 访问，使用时请遵循 Copernicus 数据许可与署名要求。
- **WeatherBench2**：提供 ERA5、HRES T0、Aurora 2022 预报和 1990–2019 气候态数据集；请引用其原始论文/数据说明。
- 本仓库代码目前未单独声明开源许可证；第三方代码、模型和数据遵循各自原始许可证。
