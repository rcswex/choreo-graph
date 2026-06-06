"""
三实例视频处理主脚本
B1 质心提取 → B2 运动学 → B3 耦合图 → B4 完成度 → B7 报告
"""

import sys
sys.path.insert(0, "/home/claude/dance_aesthetics")

import numpy as np
from core.pose_extractor import PoseExtractor
from core.kinematics import KinematicsCalculator, SpatialTopology
from core.types import DancerProfile, BodyNode
from viz.report import ReportGenerator


def process_video(name, video_path, dancer, start, end, sample_fps, output_dir):
    """提取 + 运动学 + 报告"""
    
    print(f"\n{'='*60}")
    print(f"处理: {name}")
    print(f"{'='*60}")
    
    # ── B1: 质心提取 ──
    extractor = PoseExtractor()
    result = extractor.extract(
        video_path, start_sec=start, end_sec=end,
        sample_fps=sample_fps, progress_interval=50,
    )
    result.save(f"{output_dir}/{result.video_name}")
    
    if result.frame_count < 5:
        print(f"  警告: 仅提取到 {result.frame_count} 帧，跳过分析")
        return None
    
    # ── B2: 运动学 ──
    print(f"\n  运动学计算...")
    calc = KinematicsCalculator(smooth_window=5)
    traces = {}
    
    for node_id in result.positions:
        if node_id in ["chair", "fan", "silk"]:
            continue
        pos = result.positions[node_id]
        if len(pos) < 5:
            continue
        trace = calc.compute(node_id, result.timestamps, pos)
        traces[node_id] = trace
    
    # 运动学汇总
    for nid, tr in traces.items():
        ji = calc.jerk_integral(tr.jerk, result.timestamps)
        if ji < 1e-6:
            print(f"    {nid:12s} | 无活动")
        else:
            pidx = np.argmax(tr.jerk_magnitude)
            print(f"    {nid:12s} | 急动度={ji:10.2f} | "
                  f"峰值={tr.jerk_magnitude[pidx]:8.4f} "
                  f"@ {result.timestamps[pidx]:.3f}s")
    
    # ── 空间拓扑 ──
    print(f"\n  空间拓扑...")
    n = len(result.timestamps)
    divs = np.zeros(n)
    curls = np.zeros(n)
    
    body_nodes = [b.value for b in BodyNode]
    
    for i in range(n):
        pos_i = {}
        vel_i = {}
        for nid in body_nodes:
            if nid in result.positions and nid in traces:
                pos_i[nid] = result.positions[nid][i]
                vel_i[nid] = traces[nid].velocity[i]
        
        if len(pos_i) >= 3:
            centroid = np.mean(list(pos_i.values()), axis=0)
            divs[i] = SpatialTopology.discrete_divergence(pos_i, vel_i, centroid)
            curls[i] = SpatialTopology.discrete_curl_magnitude(pos_i, vel_i, centroid)
    
    print(f"    散度: [{np.min(divs):.3f}, {np.max(divs):.3f}]")
    print(f"    旋度: [{np.min(curls):.3f}, {np.max(curls):.3f}]")
    
    # ── B7: 报告 ──
    print(f"\n  生成报告...")
    report = ReportGenerator(output_dir=output_dir)
    report.add_header(
        title=f"{name} · 介观分析报告",
        case_name=name,
        dancer_info=f"{dancer.name}, {dancer.sex}, {dancer.age}岁, "
                    f"~{dancer.height_cm:.0f}cm",
    )
    
    for nid, tr in traces.items():
        ji = calc.jerk_integral(tr.jerk, result.timestamps)
        pidx = np.argmax(tr.jerk_magnitude)
        report.add_kinematics_summary(
            node_id=nid,
            jerk_integral=ji,
            curvature_mean=float(np.nanmean(tr.curvature)),
            speed_var_cost=calc.speed_variation_cost(tr.speed, result.timestamps),
            peak_jerk=float(tr.jerk_magnitude[pidx]),
            peak_jerk_time=float(result.timestamps[pidx]),
        )
    
    report.add_spatial_topology(result.timestamps, divs, curls)
    
    report_file = f"{name.replace('·', '_').replace(' ', '_')}_report.md"
    path = report.generate(report_file)
    print(f"  报告: {path}")
    
    return {
        "result": result, "traces": traces,
        "divergence": divs, "curl": curls,
    }


def main():
    output_dir = "/home/claude/dance_aesthetics/output"
    
    # ── 舞者参数 ──
    lu = DancerProfile("鲁思彤", 26, "F", height_cm=165, weight_kg=52)
    cao = DancerProfile("曹霞", 22, "F", height_cm=163, weight_kg=50)
    wang = DancerProfile("王锦", 36, "F", height_cm=162, weight_kg=50)
    
    # ── 实例一：鲁思彤 · 浮光（抛接段 28-38s）──
    d1 = process_video(
        "实例一 · 鲁思彤 · 浮光（抛接段）",
        "videos/鲁思彤_浮光_Lu_Floating_Light.mp4",
        lu, start=28, end=38, sample_fps=30,
        output_dir=output_dir,
    )
    
    # ── 实例二：曹霞 · 椅子舞（截图段 3:55-4:10）──
    d2 = process_video(
        "实例二 · 曹霞 · 椅子舞（截图段）",
        "videos/曹霞_椅子舞_Cao_Chair_Dance.mp4",
        cao, start=235, end=250, sample_fps=15,
        output_dir=output_dir,
    )
    
    # ── 实例三：王锦 · 漫步（全段采样）──
    d3 = process_video(
        "实例三 · 王锦 · 漫步（全段）",
        "videos/王锦_漫步_Wang_Stroll.mp4",
        wang, start=0, end=None, sample_fps=6,
        output_dir=output_dir,
    )
    
    print(f"\n{'='*60}")
    print("三实例处理完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
