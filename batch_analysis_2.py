"""
批量执行：边迁移验证 + C_catch改进 + B5变分求解 + 关键帧聚合（图6a·图7a·图8a）
"""
import sys
sys.path.insert(0, "/home/claude/dance_aesthetics")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

from core.kinematics import KinematicsCalculator
from core.completion import CompletionCalculator, SubMetrics
from core.variational import MinJerkSolver, TrajectoryBoundary
from core.types import (
    DecouplingEvent, DecouplingType, Edge,
    BodyPropCoupling, BodyNode,
)

OUT = Path("/home/claude/dance_aesthetics/output")
calc = KinematicsCalculator(smooth_window=5)
nodes_body = [b.value for b in BodyNode]


# ══════════════════════════════════════════════
# 1. 边迁移实例化（鲁思彤）
# ══════════════════════════════════════════════

def verify_edge_migration():
    print("=" * 60)
    print("边迁移验证")
    print("=" * 60)
    
    event = DecouplingEvent(
        tau_minus=32.833,
        tau_plus=33.733,
        e_release=Edge("right_hand", "fan", BodyPropCoupling.GRIP),
        e_catch=Edge("left_hand", "fan", BodyPropCoupling.GRIP),
        sigma=DecouplingType.TOSS_CATCH,
    )
    
    print(f"  释放边: ({event.e_release.node_a}, {event.e_release.node_b})")
    print(f"  接住边: ({event.e_catch.node_a}, {event.e_catch.node_b})")
    print(f"  边迁移: {event.is_migration}")
    print(f"  |τ| = {event.duration:.3f}s")
    print(f"  向后兼容 .edge: ({event.edge.node_a}, {event.edge.node_b})")
    return event


# ══════════════════════════════════════════════
# 2. C7改进：运动学版 C_catch
# ══════════════════════════════════════════════

def improved_c7():
    print("\n" + "=" * 60)
    print("C7改进：运动学版 C_catch")
    print("=" * 60)
    
    d = np.load(OUT / "鲁思彤_浮光_.npz")
    ts = d["timestamps"]
    
    lh_pos = d["pos_left_hand"]
    lh_trace = calc.compute("left_hand", ts, lh_pos)
    
    # τ⁺ ≈ 33.733s
    tau_plus = 33.733
    idx_catch = np.argmin(np.abs(ts - tau_plus))
    
    # 取接住前后各5帧
    w = 5
    i0 = max(0, idx_catch - w)
    i1 = min(len(ts) - 1, idx_catch + w)
    
    vel_before = np.mean(lh_trace.velocity[i0:idx_catch], axis=0)
    vel_after = np.mean(lh_trace.velocity[idx_catch:i1], axis=0)
    jerk_at_catch = lh_trace.jerk_magnitude[idx_catch]
    
    comp = CompletionCalculator()
    c_catch_new = comp.catch_score_from_kinematics(
        vel_before, vel_after, jerk_at_catch
    )
    
    # 完整子度量
    rh_trace = calc.compute("right_hand", ts, d["pos_right_hand"])
    style_jerk = calc.jerk_integral(rh_trace.jerk, ts)
    style_curv = calc.curvature_deviation_cost(rh_trace.curvature, ts, 5.0)
    style_speed = calc.speed_variation_cost(rh_trace.speed, ts)
    c_style = comp.style_score(style_jerk, style_curv, style_speed,
                                jerk_max=5e6, curvature_max=1e4, speed_var_max=1e5)
    
    sub = SubMetrics(
        c_catch=c_catch_new,
        c_time=0.92,
        c_style=c_style,
        c_phrase=0.88,
    )
    c_total = comp.event_completion(sub)
    
    print(f"  C_catch (改进) = {c_catch_new:.3f}")
    print(f"  C_time         = {sub.c_time:.3f}")
    print(f"  C_style        = {sub.c_style:.3f}")
    print(f"  C_phrase       = {sub.c_phrase:.3f}")
    print(f"  C^(1) (改进)   = {c_total:.3f}")
    
    return sub, c_total


# ══════════════════════════════════════════════
# 3. B5：三实例最优急动度评分曲线
# ══════════════════════════════════════════════

def run_b5():
    print("\n" + "=" * 60)
    print("B5：最小急动度参考曲线")
    print("=" * 60)
    
    solver = MinJerkSolver()
    
    cases = [
        ("鲁思彤_浮光_", "torso", 0.5),
        ("曹霞_椅子舞_", "torso", 1.0),
        ("王锦_漫步_", "torso", 2.0),
    ]
    
    results = {}
    
    for fname, node, window in cases:
        d = np.load(OUT / f"{fname}.npz")
        ts = d["timestamps"]
        pos = d[f"pos_{node}"]
        
        print(f"\n  {fname} ({node}, window={window}s)...")
        ts_c, scores = solver.optimal_score_curve(pos, ts, window_sec=window)
        results[fname] = (ts_c, scores)
        
        print(f"    评分点数: {len(scores)}")
        print(f"    s* 均值: {np.mean(scores):.3f}")
        print(f"    s* 范围: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
    
    # 保存
    for fname, (ts_c, scores) in results.items():
        np.savez(OUT / f"{fname}_optimal.npz",
                 timestamps=ts_c, optimal_scores=scores)
    
    return results


# ══════════════════════════════════════════════
# 4. 图6(a)：鲁思彤关键帧选帧排版
# ══════════════════════════════════════════════

def make_fig6a():
    print("\n" + "=" * 60)
    print("图6(a)：鲁思彤关键帧排版")
    print("=" * 60)
    
    # 选5张代表帧：蓄力→甩弧→释放→峰值→接住
    frames = [
        ("/mnt/user-data/uploads/鲁思彤_浮光__00_00_32__20260605-211449_.png",
         "32.0s\n蓄力"),
        ("/mnt/user-data/uploads/鲁思彤_浮光__00_00_32__20260605-211513_.png",
         "32.5s\n甩弧"),
        ("/mnt/user-data/uploads/鲁思彤_浮光__00_00_33__20260605-211526_.png",
         "33.3s τ⁻\n释放"),
        ("/mnt/user-data/uploads/鲁思彤_浮光__00_00_33__20260605-211535_.png",
         "33.5s\n峰值"),
        ("/mnt/user-data/uploads/鲁思彤_浮光__00_00_34__20260605-211613_.png",
         "34.1s τ⁺\n接住"),
    ]
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    
    border_colors = ['#3498db', '#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    
    for i, (path, label) in enumerate(frames):
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 裁剪为中心区域
            h, w = img.shape[:2]
            crop_t = int(h * 0.1)
            crop_b = int(h * 0.95)
            img = img[crop_t:crop_b, :, :]
        else:
            img = np.zeros((400, 300, 3), dtype=np.uint8)
        
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(label, fontsize=9, fontweight='bold',
                     color=border_colors[i])
        ax.axis('off')
        
        # 边框颜色
        for spine in ax.spines.values():
            spine.set_edgecolor(border_colors[i])
            spine.set_linewidth(3)
            spine.set_visible(True)
        ax.set_frame_on(True)
    
    plt.suptitle('[Fig.6a] Lu Sitong — fan+silk toss sequence',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    
    path = OUT / "fig6a_lu_keyframes.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  图6(a) -> {path}")


# ══════════════════════════════════════════════
# 5. 图7(a)：曹霞关键帧排版
# ══════════════════════════════════════════════

def make_fig7a():
    print("\n" + "=" * 60)
    print("图7(a)：曹霞关键帧排版")
    print("=" * 60)
    
    frames = [
        ("/mnt/user-data/uploads/曹霞椅子舞-起始.png", "start\ninitial pose"),
        ("/mnt/user-data/uploads/_00_04_00__20260605-201114_.png", "4:00\nback lean"),
        ("/mnt/user-data/uploads/_00_04_02__20260605-201240_.png", "4:02\narms spread"),
        ("/mnt/user-data/uploads/_00_04_02__20260605-201255_.png", "4:02\ntip-over apex"),
        ("/mnt/user-data/uploads/_00_04_03__20260605-201327_.png", "4:03\nhands behind head"),
    ]
    colors = ['#3498db', '#f39c12', '#e74c3c', '#e74c3c', '#2ecc71']
    
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for i, (path, label) in enumerate(frames):
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            img = img[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95), :]
        else:
            img = np.zeros((300, 400, 3), dtype=np.uint8)
        ax = axes[i]
        ax.imshow(img); ax.set_title(label, fontsize=9, fontweight='bold', color=colors[i])
        ax.axis('off')
        for sp in ax.spines.values():
            sp.set_edgecolor(colors[i]); sp.set_linewidth(3); sp.set_visible(True)
        ax.set_frame_on(True)
    plt.suptitle('[Fig.7a] chair dance sequence (Cao Xia)', fontsize=12, y=1.01)
    plt.tight_layout()
    path = OUT / "fig7a_cao_keyframes.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  图7(a) -> {path}")


# ══════════════════════════════════════════════
# 6. 图8(a)：王锦关键帧排版
# ══════════════════════════════════════════════

def make_fig8a():
    print("\n" + "=" * 60)
    print("图8(a)：王锦关键帧排版")
    print("=" * 60)
    
    video_path = "videos/王锦_漫步_Wang_Stroll.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    times = [10, 60, 81, 150, 250]
    labels = ["10s\nopening", "60s\npre-anchor",
              r"81s $\star$" + "\nanchor (s=0.40)", "150s\nmid-piece", "250s\nclosing"]
    colors = ['#3498db', '#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    
    imgs = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        imgs.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret
                    else np.zeros((240, 320, 3), dtype=np.uint8))
    cap.release()
    
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for i, (img, label) in enumerate(zip(imgs, labels)):
        ax = axes[i]
        ax.imshow(img); ax.set_title(label, fontsize=9, fontweight='bold', color=colors[i])
        ax.axis('off')
        for sp in ax.spines.values():
            sp.set_edgecolor(colors[i]); sp.set_linewidth(3); sp.set_visible(True)
        ax.set_frame_on(True)
    plt.suptitle('[Fig.8a] "Stroll" key frames (Wang Jin)', fontsize=12, y=1.01)
    plt.tight_layout()
    path = OUT / "fig8a_wang_keyframes.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  图8(a) -> {path}")


# ══════════════════════════════════════════════
# 执行
# ══════════════════════════════════════════════

if __name__ == "__main__":
    event = verify_edge_migration()
    sub, c_total = improved_c7()
    b5_results = run_b5()
    make_fig6a()
    make_fig7a()
    make_fig8a()
    
    print("\n" + "=" * 60)
    print("批量执行完成")
    print("=" * 60)
    for f in sorted(OUT.glob("fig*")):
        print(f"  {f.name} ({f.stat().st_size // 1024}KB)")
    for f in sorted(OUT.glob("*_optimal.npz")):
        print(f"  {f.name} ({f.stat().st_size // 1024}KB)")
