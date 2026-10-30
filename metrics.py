"""纬度加权 RMSE / bias / ACC。所有指标都在同一网格上计算。"""

import numpy as np

G = 9.80665


def area_weights(lat: np.ndarray) -> np.ndarray:
    """规则经纬网格的面积权重（经度均匀时正比于 cos(lat)）。"""
    return np.cos(np.deg2rad(lat))


def weighted_rmse(pred: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> float:
    return float(
        np.sqrt((weights[:, None] * (pred - truth) ** 2).sum() / (weights.sum() * pred.shape[1]))
    )


def weighted_bias(pred: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> float:
    return float(
        (weights[:, None] * (pred - truth)).sum() / (weights.sum() * pred.shape[1])
    )


def acc(pred: np.ndarray, truth: np.ndarray, clim: np.ndarray, weights: np.ndarray) -> float:
    """Anomaly correlation coefficient，异常相对于气候态。"""
    f_anom = pred - clim
    o_anom = truth - clim
    numerator = (weights[:, None] * f_anom * o_anom).sum()
    denominator = np.sqrt(
        (weights[:, None] * f_anom**2).sum()
        * (weights[:, None] * o_anom**2).sum()
    )
    return float(numerator / denominator) if denominator > 0 else float("nan")
