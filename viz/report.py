"""
B7 · 可视化与报告生成
耦合图转移时间线、s(t)曲线、Δs(t)诊断图、|E(t)|波形、质心轨迹图
输出：Markdown 报告 + PNG 图表
"""

import numpy as np
from pathlib import Path
from typing import Optional
import json


class ReportGenerator:
    """
    生成 Markdown 格式的分析报告 + 配套图表数据（JSON）。
    图表由外部可视化工具（matplotlib等）渲染。
    """
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sections: list[str] = []
        self.figures: list[dict] = []
    
    def add_header(self, title: str, case_name: str, dancer_info: str):
        self.sections.append(f"""# {title}

**实例**: {case_name}
**舞者**: {dancer_info}
**生成时间**: {{timestamp}}

---
""")
    
    def add_coupling_timeline(self, timestamps: np.ndarray,
                               edge_counts: np.ndarray,
                               transitions: list[dict]):
        """添加耦合图转移时间线"""
        self.sections.append("## 耦合图转移时间线\n")
        
        # 转移事件表
        if transitions:
            self.sections.append("| 时刻 τ⁻ | 时刻 τ⁺ | 断裂/生成边 | 类型 σ | |E| 变化 |")
            self.sections.append("|---|---|---|---|---|")
            for t in transitions:
                self.sections.append(
                    f"| {t['tau_minus']:.3f}s | {t['tau_plus']:.3f}s | "
                    f"{t['edge']} | {t['sigma']} | "
                    f"{t['edge_count_before']} → {t['edge_count_after']} |"
                )
            self.sections.append("")
        
        # 保存 |E(t)| 数据
        self.figures.append({
            "type": "edge_count_waveform",
            "timestamps": timestamps.tolist(),
            "edge_counts": edge_counts.tolist(),
            "transitions": transitions,
        })
        self.sections.append("*图：|E(t)| 边数波形 → 见 edge_count_waveform.json*\n")
    
    def add_kinematics_summary(self, node_id: str,
                                jerk_integral: float,
                                curvature_mean: float,
                                speed_var_cost: float,
                                peak_jerk: float,
                                peak_jerk_time: float,
                                activity_threshold: float = 1e-6):
        """添加运动学摘要。低于噪声底限的量标注为无活动。"""
        
        is_active = jerk_integral > activity_threshold
        
        def fmt(val: float, spec: str = ".2f") -> str:
            if not is_active:
                return "—（无活动）"
            return f"{val:{spec}}"
        
        peak_str = (f"{peak_jerk:.2f} @ {peak_jerk_time:.3f}s"
                    if is_active else "—")
        
        self.sections.append(f"""### 运动学：{node_id}

| 指标 | 值 |
|---|---|
| 急动度积分 ∫‖j‖²dt | {fmt(jerk_integral)} |
| 平均曲率 κ̄ | {fmt(curvature_mean, '.4f')} |
| 速率变化代价 | {fmt(speed_var_cost)} |
| 峰值急动度 | {peak_str} |

""")
    
    def add_score_curve(self, timestamps: np.ndarray,
                        actual_scores: np.ndarray,
                        optimal_scores: Optional[np.ndarray] = None,
                        anchor_time: Optional[float] = None,
                        anchor_score: Optional[float] = None):
        """添加 s(t) 曲线"""
        self.sections.append("## 美学评分曲线 s(t)\n")
        
        data = {
            "type": "score_curve",
            "timestamps": timestamps.tolist(),
            "actual_scores": actual_scores.tolist(),
        }
        
        if optimal_scores is not None:
            data["optimal_scores"] = optimal_scores.tolist()
            delta = optimal_scores - actual_scores
            data["delta_scores"] = delta.tolist()
            
            # 统计
            improve_ratio = np.mean(delta > 0.01)
            exceed_ratio = np.mean(delta < -0.01)
            self.sections.append(
                f"- 可改进区间占比: {improve_ratio:.1%}\n"
                f"- 超预期区间占比: {exceed_ratio:.1%}\n"
                f"- 平均 Δs: {np.mean(delta):.4f}\n"
            )
        
        if anchor_time is not None:
            data["anchor"] = {"time": anchor_time, "score": anchor_score}
            self.sections.append(
                f"- 校准锚点: t₀={anchor_time:.3f}s, s(t₀)={anchor_score:.2f}\n"
            )
        
        self.figures.append(data)
        self.sections.append("*图：s(t) 曲线 → 见 score_curve.json*\n")
    
    def add_completion_summary(self, total_c: float,
                                event_scores: list[dict],
                                coupled_scores: list[dict]):
        """添加完成度总结"""
        self.sections.append(f"## 美学完成度 C = {total_c:.4f}\n")
        
        if event_scores:
            self.sections.append("### 解耦事件完成度\n")
            self.sections.append("| 事件 k | C_catch | C_time | C_style | C_phrase | C^(k) |")
            self.sections.append("|---|---|---|---|---|---|")
            for e in event_scores:
                self.sections.append(
                    f"| {e['name']} | {e['c_catch']:.3f} | {e['c_time']:.3f} | "
                    f"{e['c_style']:.3f} | {e['c_phrase']:.3f} | {e['total']:.3f} |"
                )
            self.sections.append("")
        
        if coupled_scores:
            self.sections.append("### 耦合段风格完成度\n")
            self.sections.append("| 段 | C_style |")
            self.sections.append("|---|---|")
            for c in coupled_scores:
                self.sections.append(f"| {c['name']} | {c['score']:.3f} |")
            self.sections.append("")
    
    def add_spatial_topology(self, timestamps: np.ndarray,
                              divergence: np.ndarray,
                              curl: np.ndarray):
        """添加空间拓扑分析"""
        self.sections.append("## 空间拓扑分析\n")
        
        self.figures.append({
            "type": "spatial_topology",
            "timestamps": timestamps.tolist(),
            "divergence": divergence.tolist(),
            "curl": curl.tolist(),
        })
        
        self.sections.append(
            f"- 散度范围: [{np.min(divergence):.3f}, {np.max(divergence):.3f}]\n"
            f"- 旋度范围: [{np.min(curl):.3f}, {np.max(curl):.3f}]\n"
            f"- 最大膨胀时刻: {timestamps[np.argmax(divergence)]:.3f}s\n"
            f"- 最大收缩时刻: {timestamps[np.argmin(divergence)]:.3f}s\n"
        )
        self.sections.append("*图：散度/旋度时间曲线 → 见 spatial_topology.json*\n")
    
    def add_subjective_kernel_estimate(self,
                                        config_dim: int,
                                        subjective_dim: int,
                                        anchor_sensitivity: dict):
        """添加主观核维度估计"""
        self.sections.append(f"""## 不可约主观核估计

| 维度 | 值 |
|---|---|
| 构型空间 dim(Q) | {config_dim} |
| 主观自由度 d | {subjective_dim} |
| 客观比例 | {(config_dim - subjective_dim) / config_dim:.1%} |

### 锚点敏感性分析

""")
        for param, sensitivity in anchor_sensitivity.items():
            self.sections.append(f"- {param}: Δs/Δparam = {sensitivity:.4f}")
        self.sections.append("")
    
    def generate(self, filename: str = "report.md") -> str:
        """生成完整报告"""
        import datetime
        
        content = "\n".join(self.sections)
        content = content.replace("{timestamp}",
                                   datetime.datetime.now().isoformat())
        
        # 写入 Markdown
        report_path = self.output_dir / filename
        report_path.write_text(content, encoding="utf-8")
        
        # 写入图表数据
        for fig in self.figures:
            fig_path = self.output_dir / f"{fig['type']}.json"
            fig_path.write_text(
                json.dumps(fig, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        
        return str(report_path)
