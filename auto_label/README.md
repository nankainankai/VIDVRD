# auto_label：全自动标注编排层

`auto_label/` 是面向 OpenClaw 全自动标注的第一阶段实现。它不重写现有检测、追踪和关系脚本，而是把根目录 `my_scripts/` 中已经稳定的 Step 脚本串成可批量、可复现、可断点续跑的流程。

## 当前 MVP 流程

```text
视频列表
  -> video_ingest 本地/URL 输入物化
  -> Step1 目标检测
  -> keyframe_screen 关键帧粗筛
  -> Step2 追踪 + 切窗
  -> relation_rule 规则关系候选
  -> semi_auto 关系候选
  -> relation_merge 规则 + 大模型合并
  -> relation_verify 冲突与低置信度质检
  -> export 统一导出
  -> evaluate_presence 可选评测
```

第一版输出可直接用于 Phase-1 的 Pred 评测：

- `runs/<run_id>/pred/relations_pred.json`
- `runs/<run_id>/videos/<video_id>/inputs/source.json`
- `runs/<run_id>/videos/<video_id>/export/trajectories_pred.json`
- `runs/<run_id>/videos/<video_id>/export/relation_qc.json`
- `runs/<run_id>/run_manifest.json`
- `runs/<run_id>/reports/presence_report.md`（存在 Gold 时）

## 运行方式

单视频：

```bash
python auto_label/vidvrd_auto_label.py --video assets/test1_video.mp4 --run_dir runs/exp001 --resume --api_key YOUR_KEY
```

视频列表：

```bash
python auto_label/vidvrd_auto_label.py --videos data/videos.txt --run_dir runs/exp001 --resume --api_key YOUR_KEY
```

`data/videos.txt` 每行一个视频路径或 URL，支持相对仓库根目录的路径：

```text
assets/test1_video.mp4
assets/test2.mp4
https://example.com/video.mp4
```

仅检查抽帧和轨迹框，不调用多模态模型：

```bash
python auto_label/vidvrd_auto_label.py --videos data/videos.txt --run_dir runs/debug_storyboard --resume --dry_run_relations
```

## 断点续跑

每个视频的每个节点都会写：

```text
runs/<run_id>/videos/<video_id>/<node>/status.json
```

使用 `--resume` 时，如果节点状态为 `succeeded` 且输入 hash 没变化，会直接跳过该节点。常用控制参数：

- `--resume`：复用已完成节点。
- `--force`：忽略缓存，重跑选中节点。
- `--from_node step2_track`：从某个节点开始。
- `--to_node relation_llm`：跑到某个节点结束。

节点名称：

```text
video_ingest
step1_detect
keyframe_screen
step2_track
relation_rule
relation_llm
relation_merge
relation_verify
export
```

## 配置

默认配置在 `auto_label/default_config.json`。可以复制后按实验修改：

```bash
python auto_label/vidvrd_auto_label.py --videos data/videos.txt --run_dir runs/exp002 --config auto_label/default_config.json --resume
```

当前配置包含：

- `video_ingest`：URL 下载超时、是否覆盖已有下载。
- `detector`：检测后端、Rex-Omni 类别、关键帧间隔、是否保存检测框视频。
- `keyframe_screen`：关键帧粗筛阈值，默认至少一个采样帧有两个有效物体才继续。
- `tracking`：滑窗大小和步长。
- `relation_rule`：规则关系阈值，目前支持 `left/right`、`above/below`、`near`、`overlap`。
- `relations`：谓词列表、分组大小、最大窗口数、storyboard 抽帧数、dry-run 等。
- `relation_merge`：规则和 LLM 关系合并、耦合补全。
- `relation_verify`：低置信度和互斥关系质检。
- `evaluate`：是否评测，以及 Gold 路径。

## 后续阶段

下一阶段在当前骨架上新增：

1. 扩展 `keyframe_screen`：接入多模态模型判断“是否值得继续标注”和自动裁剪建议。
2. 扩展 `relation_rule`：加入 `toward/away/follow` 等动态关系规则。
3. 扩展 `relation_verify`：加入轨迹漂移、类别漂移和覆盖率风险。
4. 扩展 `skills/vidvrd-full-auto/SKILL.md`：加入真实 OpenClaw 部署后的环境检查命令。
