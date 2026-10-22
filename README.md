# Aurora 0.25° 24h 预报复现

复现 Hugging Face × Earthmover 教程：ERA5 初始条件 -> aurora-0.25-pretrained 自回归 4 步 -> 与 ERA5 真值对比。

## 结果（2026-03-13 00UTC 起报，2 米温度全球纬度加权 RMSE）
| 时效 | RMSE |
|---|---|
| +6h  | 0.520 K |
| +12h | 0.567 K |
| +18h | 0.680 K |
| +24h | 0.740 K |

## 文件
- `fetch_data.py` — 从 Earthmover 拉取 ERA5 并缓存为本地 NetCDF（含重试）
- `run.py` — 组装 Batch、rollout 推理、逐时效 RMSE 检验与画图
- `verify_24h.png` / `bias_evolution.png` — 误差分布与四时效演化图

数据缓存（*.nc）与预测数组（*.npy）不入库，可由脚本重新生成。
