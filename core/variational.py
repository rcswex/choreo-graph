"""
B5 · 变分求解器（简化版）
最小急动度（jerk）轨迹：给定起止位置/速度/加速度，
求解使 ∫‖d³r/dt³‖²dt 最小的5次多项式轨迹。
（Flash & Hogan 1985 的解析解）
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class TrajectoryBoundary:
    """轨迹边界条件"""
    position: np.ndarray    # (3,)
    velocity: np.ndarray    # (3,)
    acceleration: np.ndarray  # (3,)


class MinJerkSolver:
    """
    最小急动度轨迹求解器。
    
    Flash & Hogan (1985) 证明，最小化 J = ∫₀ᵀ ‖d³r/dt³‖² dt
    的解是一个关于归一化时间 τ=t/T 的5次多项式：
    
    r(τ) = a₀ + a₁τ + a₂τ² + a₃τ³ + a₄τ⁴ + a₅τ⁵
    
    其中系数由边界条件唯一确定。
    """
    
    def solve(self, start: TrajectoryBoundary,
              end: TrajectoryBoundary,
              duration: float,
              n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """
        求解最小急动度轨迹。
        
        Args:
            start: 起始边界条件 (位置, 速度, 加速度)
            end: 终止边界条件
            duration: 总时长 T (秒)
            n_points: 输出轨迹的采样点数
        
        Returns:
            (timestamps, positions) — (n_points,), (n_points, 3)
        """
        T = duration
        timestamps = np.linspace(0, T, n_points)
        positions = np.zeros((n_points, 3))
        
        for dim in range(3):
            # 边界条件
            x0 = start.position[dim]
            v0 = start.velocity[dim] * T       # 归一化
            a0 = start.acceleration[dim] * T**2
            xf = end.position[dim]
            vf = end.velocity[dim] * T
            af = end.acceleration[dim] * T**2
            
            # 5次多项式系数（Flash & Hogan 解析解）
            a_0 = x0
            a_1 = v0
            a_2 = a0 / 2.0
            
            # 从终止条件解 a3, a4, a5
            # r(1) = xf, r'(1) = vf, r''(1) = af
            # 线性系统 Ac = b
            A = np.array([
                [1, 1, 1],
                [3, 4, 5],
                [6, 12, 20],
            ], dtype=float)
            b = np.array([
                xf - a_0 - a_1 - a_2,
                vf - a_1 - 2 * a_2,
                af - 2 * a_2,
            ])
            
            coeffs_345 = np.linalg.solve(A, b)
            a_3, a_4, a_5 = coeffs_345
            
            coeffs = [a_0, a_1, a_2, a_3, a_4, a_5]
            
            # 采样
            for i, t in enumerate(timestamps):
                tau = t / T
                positions[i, dim] = sum(c * tau**k for k, c in enumerate(coeffs))
        
        return timestamps, positions
    
    def solve_multi_segment(self, waypoints: list[TrajectoryBoundary],
                             durations: list[float],
                             n_points_per_segment: int = 50
                             ) -> tuple[np.ndarray, np.ndarray]:
        """
        多段最小急动度轨迹（通过途经点）。
        
        Args:
            waypoints: N+1 个边界条件（N 段）
            durations: N 个段时长
            n_points_per_segment: 每段采样点数
        """
        all_ts = []
        all_pos = []
        t_offset = 0.0
        
        for i in range(len(durations)):
            ts, pos = self.solve(
                waypoints[i], waypoints[i + 1],
                durations[i], n_points_per_segment
            )
            all_ts.append(ts + t_offset)
            all_pos.append(pos)
            t_offset += durations[i]
        
        return np.concatenate(all_ts), np.concatenate(all_pos, axis=0)
    
    def optimal_score_curve(self, actual_positions: np.ndarray,
                             timestamps: np.ndarray,
                             window_sec: float = 1.0,
                             ) -> tuple[np.ndarray, np.ndarray]:
        """
        对实际轨迹逐窗口计算最小急动度参考，返回 s*(t)。
        
        在每个窗口内，以实际的起止条件求解最小急动度轨迹，
        然后比较实际急动度积分与最优急动度积分的比值。
        
        s*(t) = exp(-J_actual / J_optimal + 1)  ∈ (0, 1]
        
        Args:
            actual_positions: (T, 3)
            timestamps: (T,)
            window_sec: 滑动窗口宽度（秒）
        
        Returns:
            (timestamps_center, optimal_scores)
        """
        from core.kinematics import KinematicsCalculator
        calc = KinematicsCalculator(smooth_window=3)
        
        dt = np.median(np.diff(timestamps))
        window_frames = max(5, int(window_sec / dt))
        half_w = window_frames // 2
        
        n = len(timestamps)
        scores = []
        ts_center = []
        
        for center in range(half_w, n - half_w, max(1, half_w // 2)):
            i0 = center - half_w
            i1 = center + half_w
            
            seg_pos = actual_positions[i0:i1]
            seg_ts = timestamps[i0:i1]
            seg_dur = seg_ts[-1] - seg_ts[0]
            
            if seg_dur < 1e-6 or len(seg_pos) < 5:
                continue
            
            # 实际急动度积分
            trace = calc.compute("_", seg_ts, seg_pos)
            j_actual = calc.jerk_integral(trace.jerk, seg_ts)
            
            # 最优急动度积分
            vel = trace.velocity
            acc = trace.acceleration
            start_bc = TrajectoryBoundary(seg_pos[0], vel[0], acc[0])
            end_bc = TrajectoryBoundary(seg_pos[-1], vel[-1], acc[-1])
            
            opt_ts, opt_pos = self.solve(start_bc, end_bc, seg_dur, len(seg_pos))
            opt_trace = calc.compute("_", opt_ts, opt_pos)
            j_optimal = calc.jerk_integral(opt_trace.jerk, opt_ts)
            
            # 评分：实际越接近最优，得分越高
            if j_optimal < 1e-10:
                score = 1.0 if j_actual < 1e-10 else 0.5
            else:
                ratio = j_actual / j_optimal
                score = float(np.exp(-ratio + 1))
                score = min(score, 1.0)
            
            scores.append(score)
            ts_center.append(timestamps[center])
        
        return np.array(ts_center), np.array(scores)
