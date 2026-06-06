"""
合成数据生成器 + 全流水线测试
模拟实例二（椅子舞）的耦合图转移序列，验证整个计算流水线。
"""

import sys
sys.path.insert(0, "/home/claude/dance_aesthetics")

import numpy as np
from core.types import (
    BodyNode, PropNode, BodyPropCoupling, IntraBodyCoupling,
    SymmetryCoupling, DancerProfile, DecouplingType, Edge,
)
from core.coupling_graph import CouplingGraph, GraphFactory, CouplingDetector
from core.kinematics import KinematicsCalculator, SpatialTopology
from core.completion import (
    CompletionCalculator, SubMetrics, ScoreCalibrator, StaticFrameScorer,
)
from viz.report import ReportGenerator


# ─────────────────────────────────────────────
# 合成数据：模拟曹霞椅子舞 3秒片段
# ─────────────────────────────────────────────

def generate_chair_dance_synthetic(fps: int = 30, duration: float = 3.0):
    """
    生成椅子舞合成数据。
    模拟 Phase 0→5 的耦合图转移和质心运动。
    
    Returns:
        timestamps, positions_dict, dancer
    """
    n_frames = int(fps * duration)
    timestamps = np.linspace(0, duration, n_frames)
    
    dancer = DancerProfile(name="曹霞", age=22, sex="F",
                           height_cm=163, weight_kg=50)
    
    # 椅子位置（固定）
    chair_pos = np.array([0.0, 0.0, 0.45])  # 座面高45cm
    
    positions = {node.value: np.zeros((n_frames, 3)) for node in BodyNode}
    positions[PropNode.CHAIR.value] = np.tile(chair_pos, (n_frames, 1))
    
    for i, t in enumerate(timestamps):
        phase = t / duration  # 归一化进度 [0, 1]
        
        # 躯干：从坐姿到后仰再到收缩
        torso_y = 0.0
        torso_z = 0.45 + 0.15 * np.sin(phase * 2 * np.pi)
        torso_x = 0.1 * np.sin(phase * np.pi)
        positions["torso"][i] = [torso_x, torso_y, torso_z]
        
        # 头：跟随躯干但有延迟和振幅放大
        head_lag = 0.05
        head_phase = max(0, phase - head_lag)
        positions["head"][i] = [
            torso_x * 1.2,
            torso_y,
            torso_z + 0.25 + 0.05 * np.sin(head_phase * 3 * np.pi)
        ]
        
        # 双手：先握椅→释放→展开→收拢→头后
        if phase < 0.15:  # Phase 0: 握椅
            hand_spread = 0.15
            hand_z = 0.45
        elif phase < 0.4:  # Phase 1-2: 释放展开
            spread_progress = (phase - 0.15) / 0.25
            hand_spread = 0.15 + 0.55 * spread_progress
            hand_z = 0.5 + 0.2 * spread_progress
        elif phase < 0.6:  # Phase 3: 最大展开
            hand_spread = 0.7
            hand_z = 0.7 + 0.1 * np.sin((phase - 0.4) / 0.2 * np.pi)
        elif phase < 0.8:  # Phase 4: 收拢
            retract_progress = (phase - 0.6) / 0.2
            hand_spread = 0.7 - 0.55 * retract_progress
            hand_z = 0.7 - 0.1 * retract_progress
        else:  # Phase 5: 头后
            hand_spread = 0.1
            hand_z = torso_z + 0.25
        
        positions["left_hand"][i] = [
            torso_x - hand_spread, torso_y - 0.05, hand_z
        ]
        positions["right_hand"][i] = [
            torso_x + hand_spread, torso_y - 0.05, hand_z
        ]
        
        # 双腿：相对稳定
        positions["left_leg"][i] = [-0.15, 0.3, 0.0]
        positions["right_leg"][i] = [0.15, 0.3, 0.0]
    
    return timestamps, positions, dancer


# ─────────────────────────────────────────────
# 全流水线测试
# ─────────────────────────────────────────────

def run_full_pipeline():
    """运行完整分析流水线并生成报告"""
    
    print("=" * 60)
    print("舞蹈美学介观分析框架 · 全流水线测试")
    print("=" * 60)
    
    # 1. 生成数据
    print("\n[1/7] 生成合成数据...")
    timestamps, positions, dancer = generate_chair_dance_synthetic()
    print(f"  帧数: {len(timestamps)}, 时长: {timestamps[-1]:.1f}s")
    print(f"  舞者: {dancer.name}, {dancer.sex}, {dancer.age}岁")
    
    # 2. 构建耦合图
    print("\n[2/7] 构建耦合图...")
    graph = GraphFactory.chair_dance(dancer)
    graph.snapshot(0.0)
    print(f"  初始活跃边数: {graph.active_edge_count}")
    
    # 模拟图转移序列
    # τ(1): 释手 @ t=0.45s
    graph.transition(0.45, 0.45,
                     ("left_hand", "chair"), BodyPropCoupling.FREE,
                     DecouplingType.INTRA_BODY)
    graph.transition(0.45, 0.45,
                     ("right_hand", "chair"), BodyPropCoupling.FREE,
                     DecouplingType.INTRA_BODY)
    graph.snapshot(0.45)
    print(f"  τ(1) 释手 @ 0.45s → |E|={graph.active_edge_count}")
    
    # τ(2): 臂躯解耦 @ t=1.0s
    graph.transition(1.0, 1.0,
                     ("left_hand", "torso"), IntraBodyCoupling.INDEPENDENT,
                     DecouplingType.INTRA_BODY)
    graph.transition(1.0, 1.0,
                     ("right_hand", "torso"), IntraBodyCoupling.INDEPENDENT,
                     DecouplingType.INTRA_BODY)
    graph.snapshot(1.0)
    print(f"  τ(2) 臂躯解耦 @ 1.0s → |E|={graph.active_edge_count}")
    
    # τ(3): 头手耦合（新边）@ t=2.4s
    graph.add_edge("head", "left_hand", IntraBodyCoupling.RIGID, 2.4)
    graph.add_edge("head", "right_hand", IntraBodyCoupling.RIGID, 2.4)
    graph.snapshot(2.4)
    print(f"  τ(3) 头手新耦合 @ 2.4s → |E|={graph.active_edge_count}")
    
    graph.snapshot(3.0)
    
    # 3. 运动学计算
    print("\n[3/7] 运动学计算...")
    calc = KinematicsCalculator(smooth_window=5)
    traces = {}
    for node in BodyNode:
        trace = calc.compute(
            node.value, timestamps, positions[node.value]
        )
        traces[node.value] = trace
    
    # 汇总统计
    for node_id, trace in traces.items():
        ji = calc.jerk_integral(trace.jerk, timestamps)
        if ji < 1e-6:
            print(f"  {node_id:12s} | 无活动")
        else:
            peak_idx = np.argmax(trace.jerk_magnitude)
            print(f"  {node_id:12s} | 急动度={ji:10.2f} | "
                  f"峰值急动度={trace.jerk_magnitude[peak_idx]:8.2f} "
                  f"@ {timestamps[peak_idx]:.3f}s | "
                  f"平均曲率={np.mean(trace.curvature):8.4f}")
    
    # 4. 空间拓扑分析
    print("\n[4/7] 空间拓扑分析...")
    n_frames = len(timestamps)
    divergences = np.zeros(n_frames)
    curls = np.zeros(n_frames)
    
    for i in range(n_frames):
        pos_i = {n: positions[n][i] for n in positions if n != "chair"}
        vel_i = {n: traces[n].velocity[i] for n in traces}
        
        # 整体重心
        centroid = np.mean([pos_i[n] for n in pos_i], axis=0)
        
        divergences[i] = SpatialTopology.discrete_divergence(
            pos_i, vel_i, centroid)
        curls[i] = SpatialTopology.discrete_curl_magnitude(
            pos_i, vel_i, centroid)
    
    print(f"  散度范围: [{np.min(divergences):.3f}, {np.max(divergences):.3f}]")
    print(f"  旋度范围: [{np.min(curls):.3f}, {np.max(curls):.3f}]")
    print(f"  最大膨胀: {timestamps[np.argmax(divergences)]:.3f}s")
    print(f"  最大收缩: {timestamps[np.argmin(divergences)]:.3f}s")
    
    # 5. 完成度计算
    print("\n[5/7] 完成度计算...")
    comp = CompletionCalculator()
    
    # 为每个模拟的解耦事件计算子度量
    event_metrics = []
    
    # τ(1): 释手（无抛接，只是释放）
    sub1 = SubMetrics(c_catch=1.0, c_time=0.95, c_style=0.82, c_phrase=0.90)
    event_metrics.append({
        "name": "τ(1) 释手",
        "c_catch": sub1.c_catch, "c_time": sub1.c_time,
        "c_style": sub1.c_style, "c_phrase": sub1.c_phrase,
        "total": comp.event_completion(sub1),
    })
    
    # τ(2): 臂躯解耦
    sub2 = SubMetrics(c_catch=1.0, c_time=0.90, c_style=0.78, c_phrase=0.85)
    event_metrics.append({
        "name": "τ(2) 臂躯解耦",
        "c_catch": sub2.c_catch, "c_time": sub2.c_time,
        "c_style": sub2.c_style, "c_phrase": sub2.c_phrase,
        "total": comp.event_completion(sub2),
    })
    
    # τ(3): 头手新耦合
    sub3 = SubMetrics(c_catch=1.0, c_time=0.92, c_style=0.88, c_phrase=0.95)
    event_metrics.append({
        "name": "τ(3) 头手新耦合",
        "c_catch": sub3.c_catch, "c_time": sub3.c_time,
        "c_style": sub3.c_style, "c_phrase": sub3.c_phrase,
        "total": comp.event_completion(sub3),
    })
    
    event_scores = [e["total"] for e in event_metrics]
    coupled_scores = [0.85, 0.80, 0.83, 0.87]  # 模拟耦合段评分
    
    total_c = comp.aggregate(event_scores, coupled_scores)
    print(f"  事件完成度: {[f'{s:.3f}' for s in event_scores]}")
    print(f"  耦合段评分: {coupled_scores}")
    print(f"  总完成度 C = {total_c:.4f}")
    
    # 6. 校准测试（模拟实例三）
    print("\n[6/7] 校准模块测试（模拟实例三模态）...")
    
    # 生成模拟的原始评分
    raw_scores = 0.5 + 0.3 * np.sin(timestamps * 2) + 0.1 * np.random.randn(len(timestamps))
    raw_scores = np.clip(raw_scores, 0, 1)
    
    # 锚点校准
    anchor_idx = len(timestamps) // 3
    calibrator = ScoreCalibrator(
        anchor_time=timestamps[anchor_idx],
        anchor_score=0.85,
    )
    calibrator.calibrate(raw_scores[anchor_idx])
    calibrated = calibrator.apply(raw_scores)
    
    print(f"  锚点: t₀={timestamps[anchor_idx]:.3f}s, s(t₀)=0.85")
    print(f"  校准前均值: {np.mean(raw_scores):.4f}")
    print(f"  校准后均值: {np.mean(calibrated):.4f}")
    print(f"  校准后范围: [{np.min(calibrated):.4f}, {np.max(calibrated):.4f}]")
    
    # 7. 生成报告
    print("\n[7/7] 生成报告...")
    report = ReportGenerator(output_dir="/home/claude/dance_aesthetics/output")
    
    report.add_header(
        title="椅子舞介观分析报告（合成数据验证）",
        case_name="实例二 · 曹霞 · 椅子舞",
        dancer_info=f"{dancer.name}, {dancer.sex}, {dancer.age}岁, "
                    f"{dancer.height_cm}cm, {dancer.weight_kg}kg",
    )
    
    # 耦合图时间线
    ts_edge, counts_edge = graph.edge_count_series()
    transition_data = []
    for t in graph.transitions:
        transition_data.append({
            "tau_minus": t.tau_minus,
            "tau_plus": t.tau_plus,
            "edge": f"({t.edge.node_a}, {t.edge.node_b})",
            "sigma": t.sigma.value,
            "edge_count_before": t.graph_before.edge_count,
            "edge_count_after": t.graph_after.edge_count,
        })
    report.add_coupling_timeline(ts_edge, counts_edge, transition_data)
    
    # 运动学摘要
    for node_id, trace in traces.items():
        ji = calc.jerk_integral(trace.jerk, timestamps)
        report.add_kinematics_summary(
            node_id=node_id,
            jerk_integral=ji,
            curvature_mean=float(np.mean(trace.curvature)),
            speed_var_cost=calc.speed_variation_cost(trace.speed, timestamps),
            peak_jerk=float(np.max(trace.jerk_magnitude)),
            peak_jerk_time=float(timestamps[np.argmax(trace.jerk_magnitude)]),
        )
    
    # 空间拓扑
    report.add_spatial_topology(timestamps, divergences, curls)
    
    # 完成度
    coupled_data = [{"name": f"C_{i}", "score": s}
                    for i, s in enumerate(coupled_scores)]
    report.add_completion_summary(total_c, event_metrics, coupled_data)
    
    # 评分曲线（模拟）
    optimal_scores = np.clip(calibrated + 0.05, 0, 1)
    report.add_score_curve(
        timestamps, calibrated, optimal_scores,
        anchor_time=timestamps[anchor_idx],
        anchor_score=0.85,
    )
    
    # 主观核估计
    report.add_subjective_kernel_estimate(
        config_dim=18,  # 6质心 × 3维
        subjective_dim=7,  # 锚点(1) + 风格参数(6)
        anchor_sensitivity={
            "anchor_score (±0.05)": 0.047,
            "急动度权重 α": 0.023,
            "curvature_target κ₀": 0.031,
            "speed_var_weight γ": 0.018,
            "generalized_mean_p": 0.015,
            "coupled_beta": 0.012,
            "spatial_sigma": 0.009,
        }
    )
    
    report_path = report.generate("chair_dance_synthetic_report.md")
    print(f"  报告已生成: {report_path}")
    
    # 输出统计
    print(f"  图表数据文件: {len(report.figures)} 个 JSON")
    
    print("\n" + "=" * 60)
    print("流水线测试完成")
    print("=" * 60)
    
    return total_c


if __name__ == "__main__":
    c = run_full_pipeline()
