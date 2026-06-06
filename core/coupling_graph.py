"""
B3 · 耦合图模型
时变耦合图 G(t) 的状态管理、图转移检测、解耦谱构建
"""

from dataclasses import dataclass, field
from core.types import (
    Edge, CouplingGraphState, DecouplingEvent, DecouplingType,
    BodyNode, PropNode, BodyPropCoupling, IntraBodyCoupling,
    SymmetryCoupling, DancerProfile,
)
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────
# 耦合图定义
# ─────────────────────────────────────────────

class CouplingGraph:
    """
    时变耦合图 G(t)。
    管理节点集、边集、状态转移历史。
    """
    
    def __init__(self, nodes: list[str], initial_edges: list[Edge]):
        self.nodes = nodes
        self.history: list[CouplingGraphState] = []
        self.transitions: list[DecouplingEvent] = []
        
        # 初始状态
        self._current_edges = {e.key: e for e in initial_edges}
    
    @property
    def current_state(self) -> CouplingGraphState:
        t = self.history[-1].timestamp if self.history else 0.0
        return CouplingGraphState(
            timestamp=t,
            edges=list(self._current_edges.values())
        )
    
    @property
    def active_edge_count(self) -> int:
        return sum(1 for e in self._current_edges.values() if not e.is_free)
    
    def snapshot(self, timestamp: float) -> CouplingGraphState:
        """记录当前时刻的图状态"""
        state = CouplingGraphState(
            timestamp=timestamp,
            edges=[Edge(e.node_a, e.node_b, e.coupling_type)
                   for e in self._current_edges.values()]
        )
        self.history.append(state)
        return state
    
    def transition(self, timestamp_start: float, timestamp_end: float,
                   edge_key: tuple[str, str], new_type: 'Enum',
                   sigma: DecouplingType) -> DecouplingEvent:
        """
        执行一次图转移。
        
        Args:
            timestamp_start: τ⁻
            timestamp_end: τ⁺
            edge_key: (node_a, node_b) 排序后的边标识
            new_type: 新的耦合类型
            sigma: 解耦方式
        
        Returns:
            DecouplingEvent
        """
        sorted_key = tuple(sorted(edge_key))
        
        # 转移前状态
        graph_before = self.current_state
        graph_before.timestamp = timestamp_start
        
        # 执行转移
        old_edge = self._current_edges.get(sorted_key)
        if old_edge is None:
            # 新边生成
            old_edge = Edge(sorted_key[0], sorted_key[1], new_type)
        
        new_edge = Edge(sorted_key[0], sorted_key[1], new_type)
        self._current_edges[sorted_key] = new_edge
        
        # 转移后状态
        graph_after = CouplingGraphState(
            timestamp=timestamp_end,
            edges=[Edge(e.node_a, e.node_b, e.coupling_type)
                   for e in self._current_edges.values()]
        )
        
        event = DecouplingEvent(
            tau_minus=timestamp_start,
            tau_plus=timestamp_end,
            edge=old_edge,
            sigma=sigma,
            graph_before=graph_before,
            graph_after=graph_after,
        )
        self.transitions.append(event)
        return event
    
    def add_edge(self, node_a: str, node_b: str, coupling_type: 'Enum',
                 timestamp: float) -> DecouplingEvent:
        """新边生成（如头手耦合）"""
        return self.transition(
            timestamp, timestamp,
            (node_a, node_b), coupling_type,
            DecouplingType.NEW_COUPLING,
        )
    
    def edge_count_series(self) -> tuple[np.ndarray, np.ndarray]:
        """
        返回 |E(t)| 时间序列（活跃边数随时间变化）
        
        Returns:
            (timestamps, edge_counts)
        """
        if not self.history:
            return np.array([]), np.array([])
        
        ts = np.array([s.timestamp for s in self.history])
        counts = np.array([s.edge_count for s in self.history])
        return ts, counts


# ─────────────────────────────────────────────
# 预设耦合图工厂
# ─────────────────────────────────────────────

class GraphFactory:
    """预设耦合图模板"""
    
    @staticmethod
    def chair_dance(dancer: DancerProfile) -> CouplingGraph:
        """实例二：椅子舞六质心+椅子"""
        nodes = [n.value for n in BodyNode] + [PropNode.CHAIR.value]
        
        initial_edges = [
            # 身体-椅
            Edge(BodyNode.TORSO.value, PropNode.CHAIR.value,
                 BodyPropCoupling.SEATED),
            Edge(BodyNode.LEFT_HAND.value, PropNode.CHAIR.value,
                 BodyPropCoupling.GRIP),
            Edge(BodyNode.RIGHT_HAND.value, PropNode.CHAIR.value,
                 BodyPropCoupling.GRIP),
            Edge(BodyNode.LEFT_LEG.value, PropNode.CHAIR.value,
                 BodyPropCoupling.TOUCH),
            Edge(BodyNode.RIGHT_LEG.value, PropNode.CHAIR.value,
                 BodyPropCoupling.TOUCH),
            # 体内：初始全刚随
            Edge(BodyNode.HEAD.value, BodyNode.TORSO.value,
                 IntraBodyCoupling.RIGID),
            Edge(BodyNode.LEFT_HAND.value, BodyNode.TORSO.value,
                 IntraBodyCoupling.RIGID),
            Edge(BodyNode.RIGHT_HAND.value, BodyNode.TORSO.value,
                 IntraBodyCoupling.RIGID),
            Edge(BodyNode.LEFT_LEG.value, BodyNode.TORSO.value,
                 IntraBodyCoupling.RIGID),
            Edge(BodyNode.RIGHT_LEG.value, BodyNode.TORSO.value,
                 IntraBodyCoupling.RIGID),
            # 双侧对称
            Edge(BodyNode.LEFT_HAND.value, BodyNode.RIGHT_HAND.value,
                 SymmetryCoupling.SYMMETRIC),
            Edge(BodyNode.LEFT_LEG.value, BodyNode.RIGHT_LEG.value,
                 SymmetryCoupling.SYMMETRIC),
        ]
        
        return CouplingGraph(nodes, initial_edges)
    
    @staticmethod
    def fan_toss(dancer: DancerProfile) -> CouplingGraph:
        """实例一：扇（可扩展绸扇组合）"""
        nodes = [n.value for n in BodyNode] + [PropNode.FAN.value]
        
        initial_edges = [
            Edge(BodyNode.RIGHT_HAND.value, PropNode.FAN.value,
                 BodyPropCoupling.GRIP),
            # 体内边
            Edge(BodyNode.HEAD.value, BodyNode.TORSO.value,
                 IntraBodyCoupling.FLOWING),
            Edge(BodyNode.LEFT_HAND.value, BodyNode.TORSO.value,
                 IntraBodyCoupling.FLOWING),
            Edge(BodyNode.RIGHT_HAND.value, BodyNode.TORSO.value,
                 IntraBodyCoupling.FLOWING),
            Edge(BodyNode.LEFT_LEG.value, BodyNode.TORSO.value,
                 IntraBodyCoupling.FLOWING),
            Edge(BodyNode.RIGHT_LEG.value, BodyNode.TORSO.value,
                 IntraBodyCoupling.FLOWING),
        ]
        
        return CouplingGraph(nodes, initial_edges)
    
    @staticmethod
    def pure_body(dancer: DancerProfile) -> CouplingGraph:
        """实例三：无道具纯身体"""
        nodes = [n.value for n in BodyNode]
        
        initial_edges = [
            Edge(BodyNode.HEAD.value, BodyNode.TORSO.value,
                 IntraBodyCoupling.FLOWING),
            Edge(BodyNode.LEFT_HAND.value, BodyNode.TORSO.value,
                 IntraBodyCoupling.FLOWING),
            Edge(BodyNode.RIGHT_HAND.value, BodyNode.TORSO.value,
                 IntraBodyCoupling.FLOWING),
            Edge(BodyNode.LEFT_LEG.value, BodyNode.TORSO.value,
                 IntraBodyCoupling.FLOWING),
            Edge(BodyNode.RIGHT_LEG.value, BodyNode.TORSO.value,
                 IntraBodyCoupling.FLOWING),
            Edge(BodyNode.LEFT_HAND.value, BodyNode.RIGHT_HAND.value,
                 SymmetryCoupling.INDEPENDENT),
            Edge(BodyNode.LEFT_LEG.value, BodyNode.RIGHT_LEG.value,
                 SymmetryCoupling.INDEPENDENT),
        ]
        
        return CouplingGraph(nodes, initial_edges)


# ─────────────────────────────────────────────
# 耦合态自动判定（基于运动学阈值）
# ─────────────────────────────────────────────

class CouplingDetector:
    """
    D4 · 从运动学数据半自动判定体内耦合态。
    
    判据：
    - 刚随：两节点速度方向余弦 > θ_rigid，速率比 ∈ [1-ε, 1+ε]
    - 流随：速度方向余弦 > θ_flow，但速率比不满足刚随
    - 独立：速度方向余弦 < θ_flow
    - 反向：速度方向余弦 < -θ_contrary
    """
    
    def __init__(self,
                 theta_rigid: float = 0.95,
                 theta_flow: float = 0.5,
                 theta_contrary: float = 0.5,
                 speed_ratio_eps: float = 0.2):
        self.theta_rigid = theta_rigid
        self.theta_flow = theta_flow
        self.theta_contrary = theta_contrary
        self.speed_ratio_eps = speed_ratio_eps
    
    def classify(self, vel_a: np.ndarray, vel_b: np.ndarray) -> IntraBodyCoupling:
        """
        判定两节点的体内耦合态。
        
        Args:
            vel_a: (3,) 节点A速度
            vel_b: (3,) 节点B速度
        """
        sa = np.linalg.norm(vel_a)
        sb = np.linalg.norm(vel_b)
        
        if sa < 1e-6 or sb < 1e-6:
            return IntraBodyCoupling.RIGID  # 两者都静止 → 刚随
        
        cos_angle = np.dot(vel_a, vel_b) / (sa * sb)
        speed_ratio = min(sa, sb) / max(sa, sb)
        
        if cos_angle < -self.theta_contrary:
            return IntraBodyCoupling.CONTRARY
        elif cos_angle < self.theta_flow:
            return IntraBodyCoupling.INDEPENDENT
        elif (cos_angle >= self.theta_rigid and
              abs(speed_ratio - 1.0) < self.speed_ratio_eps):
            return IntraBodyCoupling.RIGID
        else:
            return IntraBodyCoupling.FLOWING
    
    def classify_symmetry(self, vel_left: np.ndarray,
                          vel_right: np.ndarray,
                          body_midplane_normal: np.ndarray
                          ) -> SymmetryCoupling:
        """
        判定双侧对称耦合态。
        
        镜像对称：左右速度关于中面对称
        反对称：左右速度相同
        独立：以上都不满足
        """
        # 镜像 = 反射 right 的法向分量
        reflected = vel_right.copy()
        n = body_midplane_normal / np.linalg.norm(body_midplane_normal)
        reflected = reflected - 2 * np.dot(reflected, n) * n
        
        sl = np.linalg.norm(vel_left)
        sr = np.linalg.norm(reflected)
        
        if sl < 1e-6 and sr < 1e-6:
            return SymmetryCoupling.SYMMETRIC
        
        if sl < 1e-6 or sr < 1e-6:
            return SymmetryCoupling.INDEPENDENT
        
        cos_mirror = np.dot(vel_left, reflected) / (sl * sr)
        cos_same = np.dot(vel_left, vel_right) / (sl * np.linalg.norm(vel_right))
        
        if cos_mirror > self.theta_rigid:
            return SymmetryCoupling.SYMMETRIC
        elif cos_same > self.theta_rigid:
            return SymmetryCoupling.ANTISYMMETRIC
        else:
            return SymmetryCoupling.INDEPENDENT
