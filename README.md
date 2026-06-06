# choreo-graph

**舞蹈动态美学的介观理论** · 用图论分析编舞

论文发表于知乎：https://zhuanlan.zhihu.com/p/2046546227622318506

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()

> 舞蹈的美能不能算？能算大部分——但有一个不可消灭的小核算不了。精确定位这个核，就是本项目的核心贡献。

## 这是什么

choreo-graph 是一套在**介观层级**上分析舞蹈运动美学的数学框架和计算工具。它不在像素级（太细，丢失意义）也不在印象级（太粗，不可操作）工作，而是在两者之间的**耦合图**层级：以身体各部位质心和道具为节点、以耦合关系为边的时变图论结构。

在这个层级上，“抛扇”是一次图转移事件，“接住”是一条边的恢复，“椅子舞”是一个特定的初始图拓扑。

## 核心概念

**物质导数与急动度（jerk）阶梯** · 用矢量场的物质导数刻画运动精度的时间阶次。3阶及以上（急动度及其推广）对应神经末梢层级的精细控制——舞蹈的美学价值恰在于对最小急动度原则的有意偏离。

**耦合图 G(t)** · 六质心（头、双手、躯干、双腿）+ 道具组成的时变图。边携带类型标签（握持/流随/对称/粘连……）。解耦事件 = 图的拓扑转移。支持边迁移（释放边 ≠ 接住边）和道具内部拓扑（绸扇粘连结构）。

**完成度 C ∈ [0,1]** · 子度量乘积（空间精度 × 时间精度 × 风格符合 × 短语完整），多事件用广义均值偏向下界聚合——最差的一次失误主导整体印象。

**不可约主观核** · 美学评价中的主观自由度 d ≈ 7（锚点评分 + 风格参数），远低于构型空间维度 dim(Q) = 18。审美的绝大部分是客观推演，主观性是低维的——但不可消去。

## 项目结构

```
choreo-graph/
├── core/
│   ├── types.py              # 节点·边·耦合态·解耦事件·舞者参数
│   ├── pose_extractor.py     # B1: MediaPipe 姿态提取 → 六质心
│   ├── kinematics.py         # B2: 物质导数·急动度·曲率·散度·旋度
│   ├── coupling_graph.py     # B3: 耦合图·图转移·解耦谱·自动判定
│   ├── completion.py         # B4+B6: 完成度·校准·静态帧评分
│   └── variational.py        # B5: 最小急动度轨迹（Flash & Hogan 解析解）
├── viz/
│   └── report.py             # B7: Markdown 报告 + JSON 图表数据
├── docs/
│   ├── paper.md              # 论文正文（§1–§8）
│   ├── abstracts.md          # 三版摘要（网络/理科/学位）
│   ├── references.md         # 参考文献（20篇）
│   └── figure_prompts.md     # 概念图 GPT-image-2 Prompt 集
├── videos/                   # 三实例视频素材
│   ├── 鲁思彤_浮光_Lu_Floating_Light.mp4
│   ├── 曹霞_椅子舞_Cao_Chair_Dance.mp4
│   └── 王锦_漫步_Wang_Stroll.mp4
├── tests/
│   └── test_pipeline.py      # 合成数据全流水线测试
├── output/                   # 生成的报告·图表·提取数据（已随仓库附带）
├── pose_landmarker_lite.task # MediaPipe 姿态估计模型（已随仓库附带）
├── process_all.py            # 三视频批处理主脚本
├── batch_analysis.py         # 数据图生成（图6bc·图7bc·图9）
└── batch_analysis_2.py       # 边迁移验证 + B5 + 关键帧聚合（图6a·图7a·图8a）
```

## 三个实例

| 实例 | 舞者 | 道具 | 解耦类型 | 分析模态 |
|---|---|---|---|---|
| 鲁思彤·《浮光》 | 女, 26 | 绸扇 | 抛接（边迁移 Rh→Lh） | 解耦正分析 |
| 曹霞·椅子舞 | 女, 22 | 椅子 | 释握 + 体内解耦 + 新边生成 | 耦合图状态追踪 |
| 王锦·《漫步》 | 女, 36 | 无 | 无（纯耦合） | 校准正问题 |

## 安装与使用

```bash
pip install mediapipe opencv-python-headless numpy matplotlib

# 合成数据测试
python tests/test_pipeline.py

# 处理视频（已随仓库附带于 videos/ 目录）
python process_all.py

# 生成数据图表
python batch_analysis.py
python batch_analysis_2.py
```

姿态估计模型（`pose_landmarker_lite.task`）、三实例视频（`videos/`）、提取数据（`output/*.npz`）和生成图表（`output/*.png`）已随仓库附带。安装依赖后即可直接运行全部脚本。

## 引用

```bibtex
@article{chen2026choreograph,
  title={舞蹈美学的可计算结构与不可约主观核——基于矢量场物质导数与耦合图转移的介观分析理论},
  author={陈文戈},
  year={2026}
}
```

## 许可

[MIT License](LICENSE)

---

# choreo-graph

**Mesoscopic Theory of Dance Aesthetics** · Analyzing choreography with graph theory

> Can the beauty of dance be computed? Mostly yes — but there exists a small, irreducible kernel that cannot. Precisely locating this kernel is the core contribution of this project.

## What is this

choreo-graph is a mathematical framework and computational toolkit for analyzing dance movement aesthetics at the **mesoscopic** level. It operates neither at the pixel level (too fine — meaning dissolves) nor at the impression level (too coarse — not actionable), but at the **coupling graph** level in between: a time-varying graph-theoretic structure with body segment centers of mass and props as nodes, and coupling relationships as edges.

At this level, “tossing a fan” is a graph transition event, “catching” is edge restoration, and “chair dance” is a specific initial graph topology.

## Core Concepts

**Material derivative & jerk hierarchy** · Motion precision is characterized by the temporal order of the material derivative. Order ≥ 3 (jerk and beyond) corresponds to nerve-ending-level fine control. The aesthetic value of dance lies precisely in the *intentional deviation* from the minimum-jerk principle [Flash & Hogan, 1985].

**Coupling graph G(t)** · Six body centers of mass (head, hands, torso, legs) plus props form a time-varying graph. Edges carry typed labels (grip / flowing / symmetric / adhesion …). Decoupling events = topological transitions of the graph. Supports edge migration (release edge ≠ catch edge) and internal prop topology (silk-fan adhesion).

**Completion metric C ∈ [0,1]** · Product of sub-metrics (spatial precision × temporal precision × style conformity × phrase completeness), aggregated across multiple events via generalized mean biased toward the lower bound — the worst single failure dominates overall impression.

**Irreducible subjective kernel** · The subjective degrees of freedom in aesthetic evaluation d ≈ 7 (anchor score + style parameters), far below the configuration space dimension dim(Q) = 18. Most of aesthetics is objective computation; subjectivity is low-dimensional — but irreducible.

## Three Case Studies

| Case | Dancer | Prop | Decoupling | Analysis Mode |
|---|---|---|---|---|
| Lu Sitong, *Floating Light* | F, 26 | Silk fan | Toss-catch (edge migration Rh→Lh) | Decoupling analysis |
| Cao Xia, chair dance | F, 22 | Chair | Release + intra-body decoupling + new edge | Coupling graph tracking |
| Wang Jin, *Stroll* | F, 36 | None | None (pure coupling) | Calibrated scoring |

## Installation

```bash
pip install mediapipe opencv-python-headless numpy matplotlib

python tests/test_pipeline.py        # synthetic data test
python process_all.py                # process videos
python batch_analysis.py             # generate data figures
```

The pose estimation model (`pose_landmarker_lite.task`), source videos (`videos/`), pre-extracted data (`output/*.npz`), and generated figures (`output/*.png`) are included in the repository. All scripts can be run directly after installing dependencies.

## Citation

```bibtex
@article{chen2026choreograph,
  title={The Computable Structure and Irreducible Subjective Kernel of Dance Aesthetics:
         A Mesoscopic Theory via Material Derivatives and Coupling Graph Transitions},
  author={Chen, Wenge},
  year={2026}
}
```

## License

[MIT License](LICENSE)
