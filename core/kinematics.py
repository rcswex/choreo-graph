"""
B2 · 运动学计算器
从质心时间序列计算各阶导数、曲率、物质导数
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class KinematicTrace:
    """单个质心的完整运动学分析结果"""
    node_id: str
    timestamps: np.ndarray       # (T,)
    position: np.ndarray         # (T, 3)
    velocity: np.ndarray         # (T, 3)
    acceleration: np.ndarray     # (T, 3)
    jerk: np.ndarray             # (T, 3)  — 3阶：急动度（jerk）
    snap: np.ndarray             # (T, 3)  — 4阶
    speed: np.ndarray            # (T,)    — 速率标量
    curvature: np.ndarray        # (T,)    — 路径曲率
    jerk_magnitude: np.ndarray   # (T,)    — 急动度幅度


class KinematicsCalculator:
    """
    从位置序列 (T, 3) 计算各阶运动学量。
    使用中心差分，边界用前/后向差分。
    """
    
    def __init__(self, smooth_window: int = 5):
        """
        Args:
            smooth_window: Savitzky-Golay平滑窗口宽度（必须为奇数）
        """
        self.smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
    
    def compute(self, node_id: str, timestamps: np.ndarray,
                positions: np.ndarray) -> KinematicTrace:
        """
        完整运动学计算流水线。
        
        Args:
            node_id: 质心标识
            timestamps: (T,) 时间戳序列（秒）
            positions: (T, 3) 三维坐标序列
        
        Returns:
            KinematicTrace 完整运动学数据
        """
        pos = self._smooth(positions)
        
        # 各阶导数 — 直接用 timestamps 作为坐标
        vel = self._time_derivative(pos, timestamps)
        acc = self._time_derivative(vel, timestamps)
        jrk = self._time_derivative(acc, timestamps)
        snp = self._time_derivative(jrk, timestamps)
        
        # 标量量
        speed = np.linalg.norm(vel, axis=1)
        jerk_mag = np.linalg.norm(jrk, axis=1)
        curv = self._curvature(vel, acc)
        curv = np.nan_to_num(curv, nan=0.0)
        
        return KinematicTrace(
            node_id=node_id,
            timestamps=timestamps,
            position=pos,
            velocity=vel,
            acceleration=acc,
            jerk=jrk,
            snap=snp,
            speed=speed,
            curvature=curv,
            jerk_magnitude=jerk_mag,
        )
    
    def material_derivative(self, field_values: np.ndarray,
                            velocity: np.ndarray,
                            timestamps: np.ndarray,
                            spatial_gradient: np.ndarray) -> np.ndarray:
        """
        计算物质导数 Df/Dt = ∂f/∂t + (v·∇)f
        
        Args:
            field_values: (T, D) 场量时间序列
            velocity: (T, 3) 速度场
            timestamps: (T,) 时间戳
            spatial_gradient: (T, D, 3) 场量的空间梯度
        
        Returns:
            (T, D) 物质导数
        """
        dt = np.gradient(timestamps)
        local_deriv = self._derivative(field_values, dt)
        
        # 对流项 (v·∇)f
        # spatial_gradient[t, d, i] * velocity[t, i] -> sum over i
        convective = np.einsum('tdi,ti->td', spatial_gradient, velocity)
        
        return local_deriv + convective
    
    def jerk_integral(self, jerk: np.ndarray, timestamps: np.ndarray) -> float:
        """
        计算最小急动度代价（Flash & Hogan 1985）
        J = ∫ ‖急动度‖² dt
        """
        jerk_sq = np.sum(jerk ** 2, axis=1)
        jerk_sq = np.nan_to_num(jerk_sq, nan=0.0)
        return float(np.trapezoid(jerk_sq, timestamps))
    
    def speed_variation_cost(self, speed: np.ndarray,
                             timestamps: np.ndarray) -> float:
        """
        速率变化代价 ∫ (d|v|/dt)² dt — 气韵平稳度
        """
        dspeed = np.gradient(speed, timestamps)
        dspeed = np.nan_to_num(dspeed, nan=0.0)
        return float(np.trapezoid(dspeed ** 2, timestamps))
    
    def curvature_deviation_cost(self, curvature: np.ndarray,
                                  timestamps: np.ndarray,
                                  target_curvature: float = 0.0) -> float:
        """
        曲率偏差代价 ∫ (κ - κ₀)² dt — 风格曲率偏好
        """
        dev = (curvature - target_curvature) ** 2
        dev = np.nan_to_num(dev, nan=0.0)
        return float(np.trapezoid(dev, timestamps))
    
    # ── 内部方法 ──
    
    def _time_derivative(self, values: np.ndarray,
                          timestamps: np.ndarray) -> np.ndarray:
        """沿时间轴的导数，直接使用时间戳序列"""
        if values.ndim == 1:
            result = np.gradient(values, timestamps)
            return np.nan_to_num(result, nan=0.0)
        result = np.stack([np.gradient(values[:, i], timestamps)
                         for i in range(values.shape[1])], axis=1)
        return np.nan_to_num(result, nan=0.0)
    
    def _derivative(self, values: np.ndarray, dt: np.ndarray) -> np.ndarray:
        """沿时间轴的导数（中心差分），处理NaN"""
        if values.ndim == 1:
            result = np.gradient(values, dt)
            return np.nan_to_num(result, nan=0.0)
        result = np.stack([np.gradient(values[:, i], dt)
                         for i in range(values.shape[1])], axis=1)
        return np.nan_to_num(result, nan=0.0)
    
    def _curvature(self, vel: np.ndarray, acc: np.ndarray) -> np.ndarray:
        """
        路径曲率 κ = |v × a| / |v|³
        """
        cross = np.cross(vel, acc)
        cross_norm = np.linalg.norm(cross, axis=1)
        speed_cubed = np.linalg.norm(vel, axis=1) ** 3
        # 避免除零
        safe_denom = np.maximum(speed_cubed, 1e-10)
        return cross_norm / safe_denom
    
    def _smooth(self, data: np.ndarray) -> np.ndarray:
        """简单移动平均平滑（Savitzky-Golay的简化替代）"""
        if len(data) < self.smooth_window:
            return data
        from numpy.lib.stride_tricks import sliding_window_view
        pad = self.smooth_window // 2
        if data.ndim == 1:
            padded = np.pad(data, pad, mode='edge')
            windows = sliding_window_view(padded, self.smooth_window)
            return windows.mean(axis=1)
        else:
            result = np.zeros_like(data)
            for i in range(data.shape[1]):
                padded = np.pad(data[:, i], pad, mode='edge')
                windows = sliding_window_view(padded, self.smooth_window)
                result[:, i] = windows.mean(axis=1)
            return result


# ─────────────────────────────────────────────
# 空间拓扑分析（梯度、散度、旋度的离散近似）
# ─────────────────────────────────────────────

class SpatialTopology:
    """
    基于六质心的离散矢量场空间拓扑分析。
    
    注意：六个质心不构成连续场，以下为离散近似：
    - 散度 ≈ 质心速度的径向分量之和（膨胀/收缩指标）
    - 旋度 ≈ 质心速度的切向分量之和（旋转指标）
    """
    
    @staticmethod
    def discrete_divergence(positions: dict[str, np.ndarray],
                            velocities: dict[str, np.ndarray],
                            centroid: np.ndarray) -> float:
        """
        离散散度：所有质心相对整体重心的径向速度分量之和。
        正值 = 膨胀（展开），负值 = 收缩（收拢）。
        
        Args:
            positions: 各质心位置
            velocities: 各质心速度
            centroid: 整体重心位置
        """
        div = 0.0
        for node_id in positions:
            r = positions[node_id] - centroid
            r_norm = np.linalg.norm(r)
            if r_norm < 1e-10:
                continue
            r_hat = r / r_norm
            v_radial = np.dot(velocities[node_id], r_hat)
            div += v_radial / r_norm
        return div
    
    @staticmethod
    def discrete_curl_magnitude(positions: dict[str, np.ndarray],
                                velocities: dict[str, np.ndarray],
                                centroid: np.ndarray) -> float:
        """
        离散旋度幅度：所有质心相对重心的切向速度分量之和。
        高值 = 旋转运动为主，低值 = 径向运动为主。
        """
        curl_z = 0.0  # 简化为二维旋度（z分量）
        for node_id in positions:
            r = positions[node_id] - centroid
            v = velocities[node_id]
            # 二维叉积 r × v 的 z 分量
            curl_z += r[0] * v[1] - r[1] * v[0]
        return curl_z
    
    @staticmethod
    def control_gradient(tensions: dict[str, float],
                         positions: dict[str, np.ndarray]) -> np.ndarray:
        """
        控制强度的空间梯度近似（最小二乘拟合线性场）。
        
        Args:
            tensions: 各质心的张力/控制强度标量
            positions: 各质心位置
        
        Returns:
            (3,) 梯度向量，指向张力增长最快的方向
        """
        nodes = list(tensions.keys())
        if len(nodes) < 2:
            return np.zeros(3)
        
        pos_arr = np.array([positions[n] for n in nodes])
        ten_arr = np.array([tensions[n] for n in nodes])
        
        # 去均值
        pos_centered = pos_arr - pos_arr.mean(axis=0)
        ten_centered = ten_arr - ten_arr.mean()
        
        # 最小二乘：gradient = (X^T X)^{-1} X^T y
        try:
            grad = np.linalg.lstsq(pos_centered, ten_centered, rcond=None)[0]
        except np.linalg.LinAlgError:
            grad = np.zeros(3)
        
        return grad
