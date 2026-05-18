# VIDVRD 全自动视频关系标注

本项目面向大创项目“Agent 赋能的视频关系理解”，目标是用 OpenClaw/Agent 调用一条稳定、可复现、可恢复的全自动标注主链，把输入视频转换为可审计的结构化结果：

- 物体轨迹：逐帧 `track_id`、类别、bbox。
- 视频关系：主体、谓词、客体、起止帧、置信度、来源。
- 质检报告：轨迹风险、关系冲突、强模型复核建议。
- 可选评测报告：自动结果与 Gold 标注的 Presence P/R/F1。

## 当前架构

```text
VIDVRD/
├── src/vidvrd_auto/          # 唯一主工程包，新增能力都放这里
├── configs/                  # 唯一主配置目录
├── skills/vidvrd-full-auto/  # OpenClaw Skill 调用说明
├── scripts/                  # 薄入口脚本
├── tests/                    # unittest / smoke 测试
├── docs/                     # 中文架构、schema、Agent 工作流文档
├── my_scripts/               # 旧脚本兼容层，迁移完成后逐步废弃
├── auto_label/               # 旧入口兼容包装
├── tools/                    # 人工标注、评测等辅助工具
└── plan/                     # 会议规划与任务说明
```

新开发默认进入 `src/vidvrd_auto/`，不要再新增平行主链。

## 快速开始

所有真实运行和测试都使用 `vidvrd` conda 环境。

```bash
conda activate vidvrd
```

首次使用可安装为 editable 包：

```bash
conda run -n vidvrd python -m pip install -e .
```

不调用多模态模型的 dry-run：

```bash
conda run -n vidvrd python -m vidvrd_auto.cli --videos data/videos.txt --run_dir runs/debug001 --config configs/dry_run.json --resume --dry_run_relations --skip_eval
```

正式运行：

```bash
conda run -n vidvrd python -m vidvrd_auto.cli --videos data/videos.txt --run_dir runs/exp001 --config configs/default.json --resume --api_key YOUR_DASHSCOPE_KEY
```

不安装包时可使用薄入口：

```bash
conda run -n vidvrd python scripts/run_vidvrd_auto.py --videos data/videos.txt --run_dir runs/exp001 --config configs/default.json --resume
```

## 主链节点

```text
视频读入 video_ingest
  -> 音频先验 audio_prior
  -> 关键帧/逐帧检测 step1_detect
  -> 关键帧粗筛 keyframe_screen
  -> 轨迹生成 step2_track
  -> 轨迹质检 track_qc
  -> 规则关系 relation_rule
  -> 片段关系 relation_llm
  -> 关系合并 relation_merge
  -> 全局关系 global_relation
  -> 关系复核 relation_verify
  -> 导出 export
```

每个节点都会写 `status.json`，同一视频和同一配置可用 `--resume` 断点续跑。

## 输出

所有运行产物都写入 `runs/<run_id>/`：

- `run_manifest.json`
- `pred/relations_pred.json`
- `videos/<video_id>/export/trajectories_pred.json`
- `videos/<video_id>/export/relation_qc.json`
- `reports/presence_report.md`，当 Gold 文件存在且启用评测时生成

## OpenClaw

项目级 Skill：

`skills/vidvrd-full-auto/SKILL.md`

OpenClaw/Agent 的职责是调用稳定 CLI、检查 `run_manifest.json`、定位失败节点，并用相同命令加 `--resume` 恢复运行。业务逻辑不写进 Skill，统一保留在 `src/vidvrd_auto/`。

## 重要文档

- `docs/ARCHITECTURE.md`
- `docs/SCHEMAS.md`
- `docs/WORKFLOW_AGENT.md`
- `plan/大创会议.md`
- `plan/plan.md`

## 旧架构说明

`my_scripts/` 仍作为过渡兼容层被检测、追踪、片段关系节点调用，但不再新增主流程编排逻辑。后续迁移完成后，旧脚本只保留为反向调用新包的薄包装器。

`auto_label/vidvrd_auto_label.py` 是旧入口兼容包装，实际委托给 `vidvrd_auto.cli`。

## 验证

```bash
$env:PYTHONPATH="src"
conda run -n vidvrd python -m vidvrd_auto.cli --help
conda run -n vidvrd python -m compileall -q src scripts tests
conda run -n vidvrd python -m unittest discover -s tests
```
