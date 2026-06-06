"""
舞蹈美学介观分析框架 — 核心类型定义
Dance Aesthetics Mesoscopic Analysis Framework — Core Types
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────
# 节点：质心标识
# ─────────────────────────────────────────────

class BodyNode(Enum):
    """舞者六质心"""
    HEAD = "head"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    TORSO = "torso"
    LEFT_LEG = "left_leg"
    RIGHT_LEG = "right_leg"


class PropNode(Enum):
    """道具节点（可扩展）"""
    CHAIR = "chair"
    FAN = "fan"
    SILK = "silk"


# ─────────────────────────────────────────────
# 边：耦合类型
# ─────────────────────────────────────────────

class BodyPropCoupling(Enum):
    """身体-道具 耦合态"""
    GRIP = "grip"          # 握持：刚性约束
    LEAN = "lean"          # 倚靠：面接触
    HOOK = "hook"          # 勾挂：线接触
    PRESS = "press"        # 压：力约束
    TOUCH = "touch"        # 触：接触无力
    SEATED = "seated"      # 坐：骨盆-座面
    FREE = "free"          # 自由：无耦合


class IntraBodyCoupling(Enum):
    """体内 耦合态（肢体-躯干，或肢体-肢体）"""
    RIGID = "rigid"        # 刚随：作为整体运动
    FLOWING = "flowing"    # 流随：跟随但有相位差
    INDEPENDENT = "independent"  # 独立：各自运动
    CONTRARY = "contrary"  # 反向：刻意反向运动


class SymmetryCoupling(Enum):
    """双侧对称耦合态"""
    SYMMETRIC = "symmetric"        # 镜像对称
    ANTISYMMETRIC = "antisymmetric"  # 反对称
    INDEPENDENT = "independent"    # 独立


class PropInternalCoupling(Enum):
    """道具内部耦合态（如绸-扇）"""
    ADHESION = "adhesion"  # 粘连
    SLIDING = "sliding"    # 滑动
    FREE = "free"          # 自由


# ─────────────────────────────────────────────
# 边的统一表示
# ─────────────────────────────────────────────

@dataclass
class Edge:
    """耦合图中的一条边"""
    node_a: str          # 节点标识 (BodyNode.value 或 PropNode.value)
    node_b: str          # 节点标识
    coupling_type: Enum  # 上述四种耦合类型之一
    
    @property
    def is_free(self) -> bool:
        return self.coupling_type.value == "free"
    
    @property
    def key(self) -> tuple:
        """排序后的节点对，保证 (A,B) == (B,A)"""
        return tuple(sorted([self.node_a, self.node_b]))
    
    def __eq__(self, other):
        return self.key == other.key
    
    def __hash__(self):
        return hash(self.key)


# ─────────────────────────────────────────────
# 解耦事件
# ─────────────────────────────────────────────

class DecouplingType(Enum):
    """解耦方式分类"""
    TOSS_CATCH = "toss_catch"     # 抛接
    TOSS_DROP = "toss_drop"       # 抛落
    RELEASE_SLIDE = "release_slide"  # 滑释
    PEEL = "peel"                 # 剥离（渐进）
    SUSPEND = "suspend"           # 悬停
    INTRA_BODY = "intra_body"     # 体内解耦
    NEW_COUPLING = "new_coupling" # 新边生成


@dataclass
class DecouplingEvent:
    """
    解耦谱中的一个事件 τ^(k)。
    
    支持边迁移：释放边 e_release 与接住边 e_catch 可以不同。
    当两者相同时，退化为标准解耦-恢复。
    """
    tau_minus: float           # 解耦开始时刻 (秒)
    tau_plus: float            # 解耦结束时刻 (秒)
    e_release: Edge            # 释放边（断裂的边）
    e_catch: Optional[Edge] = None  # 接住边（恢复的边），None=未接住(抛落)
    sigma: DecouplingType = DecouplingType.TOSS_CATCH
    graph_before: Optional['CouplingGraphState'] = None
    graph_after: Optional['CouplingGraphState'] = None
    
    @property
    def duration(self) -> float:
        """滞空/解耦时长 |τ|"""
        return self.tau_plus - self.tau_minus
    
    @property
    def is_migration(self) -> bool:
        """是否为边迁移（释放边 ≠ 接住边）"""
        if self.e_catch is None:
            return False
        return self.e_release.key != self.e_catch.key
    
    @property
    def edge(self) -> Edge:
        """向后兼容：返回释放边"""
        return self.e_release


# ─────────────────────────────────────────────
# 耦合图状态（某一时刻的快照）
# ─────────────────────────────────────────────

@dataclass
class CouplingGraphState:
    """耦合图 G(t) 在某一时刻的状态"""
    timestamp: float
    edges: list[Edge] = field(default_factory=list)
    
    @property
    def active_edges(self) -> list[Edge]:
        """所有非自由边"""
        return [e for e in self.edges if not e.is_free]
    
    @property
    def edge_count(self) -> int:
        """|E(t)|：活跃边数"""
        return len(self.active_edges)
    
    def get_edge(self, node_a: str, node_b: str) -> Optional[Edge]:
        key = tuple(sorted([node_a, node_b]))
        for e in self.edges:
            if e.key == key:
                return e
        return None


# ─────────────────────────────────────────────
# 舞者体态参数（可估计）
# ─────────────────────────────────────────────

@dataclass
class DancerProfile:
    """舞者生物力学参数"""
    name: str
    age: int
    sex: str  # "F" or "M"
    height_cm: float = 165.0       # 默认值：中国女性平均
    arm_span_cm: float = 163.0     # 默认 ≈ 身高
    weight_kg: float = 52.0        # 默认值
    
    # 以下参数从身高/体重估计
    @property
    def torso_mass_ratio(self) -> float:
        """躯干质量占比（基于 Winter 1990 人体测量学数据）"""
        return 0.497
    
    @property
    def head_mass_ratio(self) -> float:
        return 0.081
    
    @property
    def arm_mass_ratio(self) -> float:
        """单臂"""
        return 0.050
    
    @property
    def leg_mass_ratio(self) -> float:
        """单腿"""
        return 0.161
    
    def mass_of(self, node: BodyNode) -> float:
        """各质心质量 (kg)"""
        ratios = {
            BodyNode.HEAD: self.head_mass_ratio,
            BodyNode.TORSO: self.torso_mass_ratio,
            BodyNode.LEFT_HAND: self.arm_mass_ratio,
            BodyNode.RIGHT_HAND: self.arm_mass_ratio,
            BodyNode.LEFT_LEG: self.leg_mass_ratio,
            BodyNode.RIGHT_LEG: self.leg_mass_ratio,
        }
        return self.weight_kg * ratios[node]


# ─────────────────────────────────────────────
# 帧数据
# ─────────────────────────────────────────────

@dataclass
class FrameData:
    """单帧数据"""
    timestamp: float                          # 秒
    positions: dict[str, np.ndarray]          # 节点 -> [x, y, z] 坐标
    coupling_state: Optional[CouplingGraphState] = None
    score: Optional[float] = None             # 美学评分（如有）
    
    def position_of(self, node: str) -> np.ndarray:
        return self.positions.get(node, np.zeros(3))
