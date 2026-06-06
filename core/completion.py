"""
B4 · 美学完成度计算器
C ∈ [0,1]：子度量分解、聚合、校准
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from core.types import DecouplingEvent, CouplingGraphState


@dataclass
class SubMetrics:
    """单个解耦事件或耦合段的子度量"""
    c_catch: float = 1.0       # 空间精度（解耦事件专用）
    c_time: float = 1.0        # 时间精度（解耦事件专用）
    c_style: float = 1.0       # 风格符合度
    c_phrase: float = 1.0      # 短语完整度
    
    def total(self, weights: Optional[dict] = None) -> float:
        """
        加权乘积聚合。任一子度量为零则整体为零。
        
        Args:
            weights: {"catch": w1, "time": w2, "style": w3, "phrase": w4}
                     默认均为 1.0
        """
        w = weights or {"catch": 1.0, "time": 1.0, "style": 1.0, "phrase": 1.0}
        return (self.c_catch ** w["catch"] *
                self.c_time ** w["time"] *
                self.c_style ** w["style"] *
                self.c_phrase ** w["phrase"])


class CompletionCalculator:
    """
    美学完成度计算器。
    
    单次解耦事件 → SubMetrics → C^(k)
    全段视频 → 聚合所有事件 + 耦合段 → C_total
    """
    
    def __init__(self,
                 spatial_sigma: float = 0.05,
                 temporal_sigma: float = 0.033,
                 generalized_mean_p: float = -2.0,
                 coupled_weight_beta: float = 0.5,
                 sub_weights: Optional[dict] = None):
        """
        Args:
            spatial_sigma: 空间精度高斯核宽度 (米)
            temporal_sigma: 时间精度高斯核宽度 (秒)
            generalized_mean_p: 广义均值指数 p<0（越小越惩罚最差）
            coupled_weight_beta: 耦合段权重指数 β<1
            sub_weights: 子度量权重
        """
        self.spatial_sigma = spatial_sigma
        self.temporal_sigma = temporal_sigma
        self.p = generalized_mean_p
        self.beta = coupled_weight_beta
        self.sub_weights = sub_weights or {
            "catch": 1.0, "time": 1.0, "style": 1.0, "phrase": 1.0
        }
    
    # ── 子度量计算 ──
    
    def catch_score(self, hand_pos: np.ndarray,
                    prop_pos: np.ndarray) -> float:
        """
        C_catch（空间精度） = exp(-||r_hand - r_prop||² / σ²)
        用于已知道具位置的情形。
        """
        dist = np.linalg.norm(hand_pos - prop_pos)
        return float(np.exp(-(dist ** 2) / (self.spatial_sigma ** 2)))
    
    def catch_score_from_kinematics(self,
                                     hand_velocity_before: np.ndarray,
                                     hand_velocity_after: np.ndarray,
                                     hand_jerk_at_catch: float,
                                     jerk_baseline: float = 500.0) -> float:
        """
        C_catch（运动学版）：从接住前后的速度变化推断接住质量。
        
        成功的接住表现为：
        1. 速度方向突变（手改变运动方向去匹配道具）
        2. 接住后急动度迅速回落到基线
        
        C = direction_change_score × deceleration_score
        
        Args:
            hand_velocity_before: 接住前数帧的平均速度
            hand_velocity_after: 接住后数帧的平均速度
            hand_jerk_at_catch: 接住时刻的急动度幅度
            jerk_baseline: 急动度基线（非抛接段的平均值）
        """
        # 方向变化：接住前后速度方向余弦
        sb = np.linalg.norm(hand_velocity_before)
        sa = np.linalg.norm(hand_velocity_after)
        
        if sb < 1e-8 or sa < 1e-8:
            dir_score = 0.5
        else:
            cos_angle = np.dot(hand_velocity_before, hand_velocity_after) / (sb * sa)
            # 方向变化越大越好（说明手主动调整了方向去接）
            dir_score = (1.0 - cos_angle) / 2.0  # [-1,1] → [0,1]
        
        # 减速：接住后速率应低于接住前（吸收了冲量）
        if sb < 1e-8:
            decel_score = 0.5
        else:
            speed_ratio = sa / sb
            decel_score = float(np.exp(-speed_ratio))  # 比值越小越好
        
        return float(np.clip(dir_score * 0.5 + decel_score * 0.5, 0.0, 1.0))
    
    def time_score(self, actual_time: float,
                   target_time: float) -> float:
        """
        C_time = exp(-(t_actual - t_target)² / τ²)
        """
        dt = actual_time - target_time
        return float(np.exp(-(dt ** 2) / (self.temporal_sigma ** 2)))
    
    def style_score(self, jerk_cost: float, curvature_cost: float,
                    speed_var_cost: float,
                    jerk_max: float = 1e6,
                    curvature_max: float = 1e3,
                    speed_var_max: float = 1e4) -> float:
        """
        C_style = 1 - J_total / J_max
        其中 J_total = α·急动度代价 + β·curv_cost + γ·speed_cost
        
        归一化后映射到 [0,1]
        """
        # 归一化各项到 [0,1]
        j_norm = min(jerk_cost / jerk_max, 1.0)
        c_norm = min(curvature_cost / curvature_max, 1.0)
        s_norm = min(speed_var_cost / speed_var_max, 1.0)
        
        # 加权平均
        total = (j_norm + c_norm + s_norm) / 3.0
        return 1.0 - total
    
    def phrase_score(self, movement_duration: float,
                     expected_duration: float,
                     key_poses_hit: int,
                     key_poses_total: int) -> float:
        """
        C_phrase：动作短语完整度
        基于时长比和关键姿态命中率
        """
        if expected_duration <= 0 or key_poses_total <= 0:
            return 1.0
        
        duration_ratio = min(movement_duration / expected_duration, 1.0)
        pose_ratio = key_poses_hit / key_poses_total
        
        return duration_ratio * pose_ratio
    
    # ── 单事件完成度 ──
    
    def event_completion(self, sub: SubMetrics) -> float:
        """单个解耦事件的完成度 C^(k)"""
        return sub.total(self.sub_weights)
    
    # ── 全段聚合 ──
    
    def aggregate(self,
                  event_scores: list[float],
                  coupled_scores: list[float]) -> float:
        """
        全段完成度聚合。
        
        C_total = C_decouple · (mean(C_coupled))^β
        
        其中 C_decouple 使用广义均值偏向下界：
        C_decouple = (Σ C_k^p / K)^(1/p),  p < 0
        
        Args:
            event_scores: 各解耦事件的 C^(k) 列表
            coupled_scores: 各耦合段的风格完成度列表
        """
        if not event_scores and not coupled_scores:
            return 0.0
        
        # 解耦事件聚合
        if event_scores:
            c_decouple = self._generalized_mean(event_scores, self.p)
        else:
            c_decouple = 1.0  # 无解耦事件（如实例三）
        
        # 耦合段聚合
        if coupled_scores:
            c_coupled = np.mean(coupled_scores) ** self.beta
        else:
            c_coupled = 1.0
        
        return c_decouple * c_coupled
    
    def _generalized_mean(self, values: list[float], p: float) -> float:
        """
        广义均值 M_p = (Σ x_i^p / n)^(1/p)
        p → -∞ 退化为 min
        """
        values = np.array(values)
        values = np.clip(values, 1e-10, 1.0)  # 避免零值导致的数值问题
        
        if abs(p) > 100:
            return float(np.min(values))
        
        return float((np.mean(values ** p)) ** (1.0 / p))


# ─────────────────────────────────────────────
# 校准模块 (B6)
# ─────────────────────────────────────────────

class ScoreCalibrator:
    """
    B6 · 评分函数校准
    
    以单帧锚点 s(t₀) 校准评分函数的标度参数，
    然后对全视频逐帧评分。
    """
    
    def __init__(self, anchor_time: float, anchor_score: float):
        """
        Args:
            anchor_time: 锚点帧的时间 t₀
            anchor_score: 锚点帧的人工评分 s(t₀)
        """
        self.anchor_time = anchor_time
        self.anchor_score = anchor_score
        self._scale = None
        self._offset = None
    
    def calibrate(self, raw_score_at_anchor: float):
        """
        用锚点校准。假设线性标度 s = a·s_raw + b，
        额外约束 s(0分) = 0 → b = 0 → a = anchor_score / raw_score
        """
        if abs(raw_score_at_anchor) < 1e-10:
            self._scale = 1.0
        else:
            self._scale = self.anchor_score / raw_score_at_anchor
        self._offset = 0.0
    
    def apply(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        将校准应用到全视频的原始评分序列。
        
        Args:
            raw_scores: (T,) 未校准的评分序列
        
        Returns:
            (T,) 校准后的评分序列，裁剪到 [0, 1]
        """
        if self._scale is None:
            raise RuntimeError("请先调用 calibrate()")
        
        calibrated = raw_scores * self._scale + self._offset
        return np.clip(calibrated, 0.0, 1.0)
    
    def delta_s(self, actual_scores: np.ndarray,
                optimal_scores: np.ndarray) -> np.ndarray:
        """
        Δs(t) = s*(t) - s_actual(t)
        正值 = 可改进区间，负值 = 超预期区间
        """
        return optimal_scores - actual_scores


# ─────────────────────────────────────────────
# 静态帧评分（实例三用）
# ─────────────────────────────────────────────

class StaticFrameScorer:
    """
    从单帧的六质心构型计算静态美感评分的子度量。
    
    s = s_line · s_balance · s_tension
    """
    
    def line_score(self, positions: dict[str, np.ndarray]) -> float:
        """
        线条分：主要肢体线条的曲率连续性。
        以右臂-躯干-左臂构成的主线条为例。
        """
        # 取三点：左手-躯干-右手
        lh = positions.get("left_hand", np.zeros(3))
        t = positions.get("torso", np.zeros(3))
        rh = positions.get("right_hand", np.zeros(3))
        
        # 曲率 ≈ 2|sin(θ)| / |chord|
        v1 = lh - t
        v2 = rh - t
        
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.5
        
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # 理想角度范围：120°-180°（优美的弧线）
        # 映射到 [0,1]：180°→1.0, 90°→0.5, 0°→0.0
        return float(angle / np.pi)
    
    def balance_score(self, positions: dict[str, np.ndarray],
                      masses: dict[str, float]) -> float:
        """
        平衡分：重心投影是否在支撑面内。
        """
        # 计算加权重心
        total_mass = sum(masses.values())
        com = np.zeros(3)
        for node, pos in positions.items():
            if node in masses:
                com += pos * masses[node]
        com /= total_mass
        
        # 支撑点：脚的位置
        ll = positions.get("left_leg", com)
        rl = positions.get("right_leg", com)
        
        # 支撑面中心
        support_center = (ll + rl) / 2.0
        
        # 重心投影到支撑面的距离（仅水平分量）
        com_xy = com[:2]
        support_xy = support_center[:2]
        dist = np.linalg.norm(com_xy - support_xy)
        
        # 支撑面半径估计
        support_radius = np.linalg.norm(ll[:2] - rl[:2]) / 2.0 + 0.1
        
        # 距离越小越稳定
        ratio = dist / support_radius
        return float(np.exp(-ratio ** 2))
    
    def tension_score(self, positions: dict[str, np.ndarray]) -> float:
        """
        张力分：肢体伸展程度（作为力量投入的代理指标）。
        """
        # 各肢体到躯干的距离占最大可能距离的比例
        t = positions.get("torso", np.zeros(3))
        
        distances = []
        for node in ["left_hand", "right_hand", "left_leg", "right_leg", "head"]:
            if node in positions:
                d = np.linalg.norm(positions[node] - t)
                distances.append(d)
        
        if not distances:
            return 0.5
        
        # 归一化：假设最大伸展距离 ≈ 1.0m
        max_extension = 1.0
        mean_extension = np.mean(distances) / max_extension
        
        return float(np.clip(mean_extension, 0.0, 1.0))
    
    def composite_score(self, positions: dict[str, np.ndarray],
                        masses: dict[str, float]) -> tuple[float, dict]:
        """
        综合静态评分。
        
        Returns:
            (total_score, {"line": ..., "balance": ..., "tension": ...})
        """
        s_l = self.line_score(positions)
        s_b = self.balance_score(positions, masses)
        s_t = self.tension_score(positions)
        
        total = s_l * s_b * s_t
        
        return total, {"line": s_l, "balance": s_b, "tension": s_t}
