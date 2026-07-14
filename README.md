# VIDVRD 全自动视频关系标注

本项目面向大创项目"Agent 赋能的视频关系理解"，目标是用 OpenClaw/Agent 调用一条稳定、可复现、可恢复的全自动标注主链，把输入视频转换为可审计的结构化结果：

- 物体轨迹：逐帧 `track_id`、类别、bbox。
- 视频关系：主体、谓词、客体、起止帧、置信度、来源。
- 质检报告：轨迹风险、关系冲突、强模型复核建议。
- 可选评测报告：自动结果与 Gold 标注的 Presence P/R/F1（含分谓词分析）。

## 项目结构

```text
VIDVRD/
├── src/vidvrd_auto/          # 唯一主工程包
│   ├── cli.py                # CLI 入口
│   ├── config/               # 配置加载与合并
│   ├── pipeline/             # 节点编排、manifest、runner
│   ├── nodes/                # 12 个流程节点
│   ├── models/               # VL 模型客户端
│   ├── prompts/              # 中文 Prompt 模板
│   ├── relations/            # 规则关系、候选生成、合并、复核
│   ├── detection/            # 检测适配层（调旧 Step1）
│   ├── tracking/             # 追踪适配层（调旧 Step2）
│   ├── evaluation/           # Presence 评测入口
│   └── utils/                # IO、路径、hash、进程
├── configs/                  # 配置文件
│   ├── default.json          # 完整默认配置
│   ├── semantic_relations.json # 语义关系实验配置
│   ├── predicate_taxonomy.json # 谓词定义（25 个谓词）
│   ├── dry_run.json          # dry-run 覆盖配置
│   └── production.json       # 生产覆盖配置
├── scripts/                  # 薄入口脚本
├── tests/                    # unittest 测试
├── docs/                     # 架构、Schema、Agent 工作流文档
├── tools/                    # 评测、人工标注辅助工具
├── my_scripts/               # 旧脚本适配层（检测、追踪、关系分类）
├── plan/                     # 会议规划与改进计划
├── skills/vidvrd-full-auto/  # OpenClaw Skill
└── data/                     # 样本数据
```

## 快速开始

所有运行使用 `vidvrd` conda 环境：

```bash
conda activate vidvrd
pip install -e .
```

dry-run（不调用多模态模型）：

```bash
python -m vidvrd_auto.cli --videos data/videos.txt --run_dir runs/debug001 --config configs/dry_run.json --resume --dry_run_relations --skip_eval
```

语义关系实验：

```bash
python -m vidvrd_auto.cli --videos data/videos_semantic.txt --run_dir runs/semantic_v1 --config configs/semantic_relations.json --resume --api_key YOUR_DASHSCOPE_KEY
```

## 主链节点

```text
video_ingest   视频读入
  → audio_prior   音频先验
  → step1_detect  目标检测（Rex-Omni / DINO-X）
  → keyframe_screen  关键帧粗筛
  → step2_track   轨迹生成（OC-SORT）
  → track_qc      轨迹质检
  → relation_rule  规则关系 + Object-Aware 候选
  → relation_llm   片段关系（Storyboard + Qwen-VL）
  → relation_merge 关系合并 + 耦合补全
  → global_relation 跨窗口聚合
  → relation_verify 复核 + 冲突消解 + 类别约束过滤
  → export         导出
```

每个节点写 `status.json`，同一视频同一配置可用 `--resume` 断点续跑。

## 谓词体系

当前支持 25 个谓词，分四层：

| 层级 | 谓词示例 |
|------|----------|
| 空间关系 | left, right, above, below, near, overlap, on, under |
| 运动关系 | toward, away, follow, chase, moving_together |
| 动作关系 | ride, sit_on, hold, carry, wear, kick, push |
| 交互关系 | hug, talk_to, look_at, walk_with, play_with, sing_with |

详见 `configs/predicate_taxonomy.json`。

## 输出

所有运行产物写入 `runs/<run_id>/`：

- `run_manifest.json`
- `pred/relations_pred.json`
- `videos/<video_id>/export/trajectories_pred.json`
- `videos/<video_id>/export/relation_qc.json`
- `reports/presence_report.md`（含分视频和分谓词评测表）

## 验证

```bash
python -m vidvrd_auto.cli --help
python -m compileall -q src scripts tests
python -m unittest discover -s tests
```

## 文档

- `docs/ARCHITECTURE.md` — 包结构与节点说明
- `docs/SCHEMAS.md` — 输出 JSON Schema
- `docs/WORKFLOW_AGENT.md` — OpenClaw Agent 工作流
- `plan/semantic_relation_plan.md` — 语义关系改进计划
