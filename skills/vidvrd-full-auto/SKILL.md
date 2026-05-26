---
name: vidvrd-full-auto
description: 运行 VIDVRD 全自动视频关系标注流程。用户要求基于 OpenClaw 自动标注视频、批量生成轨迹和关系、恢复失败运行或查看运行报告时使用。
---

# VIDVRD 全自动标注

## 用途

本 Skill 用于调用本仓库的稳定全自动标注主链。确定性 runner 是 `src/vidvrd_auto` 中的 `vidvrd_auto.cli`；Skill 只负责说明如何调用、如何查看输出、如何恢复失败。

## 前置检查

运行前执行环境检查（推荐）：

```bash
conda run -n vidvrd python scripts/check_openclaw_env.py
```

无 conda 时：

```bash
python scripts/check_openclaw_env.py
```

运行前确认：

1. 使用 `vidvrd` conda 环境（或已 `pip install -e .` 的 Python 3.10+）。
2. 输入是单个视频路径、URL，或每行一个视频/URL 的文本文件（示例列表：`data/videos.txt`）。
3. `configs/default.json` 存在，或用户提供了自定义 JSON 配置。
4. 首次 smoke test：生成测试视频 `python scripts/make_validation_dummy.py`。
5. 非 dry-run 的关系标注需要设置 `DASHSCOPE_API_KEY`，或传入 `--api_key`。
6. Rex-Omni 检测需要本地依赖和模型路径；DINO-X 检测需要 `DINOX_API_TOKEN`。
7. **无 API/无 GPU 验证主链**：使用 `configs/dry_run.json`（`detector.backend=mock`、`tracking.backend=mock`），不调用检测模型与 VL API。
8. 如果 `python -m vidvrd_auto.cli` 无法导入，先在仓库根目录运行 `conda run -n vidvrd python -m pip install -e .`，或使用 `python scripts/run_vidvrd_auto.py`。

## 主命令

单视频：

```bash
conda run -n vidvrd python -m vidvrd_auto.cli --video <video_path> --run_dir runs/<run_id> --config configs/default.json --resume --api_key <key>
```

视频列表：

```bash
conda run -n vidvrd python -m vidvrd_auto.cli --videos <videos_txt> --run_dir runs/<run_id> --config configs/default.json --resume --api_key <key>
```

不调用多模态模型与检测模型的 dry-run（mock 检测/追踪 + 规则关系）：

```bash
conda run -n vidvrd python scripts/make_validation_dummy.py
conda run -n vidvrd python -m vidvrd_auto.cli --video data/validation_dummy.mp4 --run_dir runs/smoke001 --config configs/dry_run.json --resume --dry_run_relations --skip_eval
```

批量列表：

```bash
conda run -n vidvrd python -m vidvrd_auto.cli --videos data/videos.txt --run_dir runs/<run_id> --config configs/dry_run.json --resume --dry_run_relations --skip_eval
```

mock 检测 + 全链路真实 VL（关键帧/轨迹/全局/片段关系均看图）：

```bash
conda run -n vidvrd python scripts/run_with_vl.ps1
# 或: python scripts/run_vidvrd_auto.py --video data/validation_dummy.mp4 --run_dir runs/live_vl --config configs/run_with_vl.json --resume --api_key $DASHSCOPE_API_KEY
```

## Agent 工作流

1. 新数据或新环境先用 `--dry_run_relations --skip_eval` 验证。
2. dry-run 成功后，去掉 `--dry_run_relations` 调用多模态模型。
3. 检查 `runs/<run_id>/run_manifest.json`。
4. 如果部分视频失败，用相同命令加 `--resume` 恢复。
5. 只恢复局部节点时使用 `--from_node` 和 `--to_node`。

说明：当使用 `--to_node` 停在 export 之前时，视频会在 manifest 中标记为 `state=partial`（不会生成 `export/relations_pred.json` / `trajectories_pred.json`，也不会计入最终 `pred/relations_pred.json` 聚合）。manifest 的 `args` 字段会记录本次运行的 `from_node/to_node` 等关键参数，便于审计与复现。

节点名：

```text
video_ingest
audio_prior
step1_detect
keyframe_screen
step2_track
track_qc
relation_rule
relation_llm
relation_merge
global_relation
relation_verify
export
```

## 输出

成功后向用户报告：

- `runs/<run_id>/pred/relations_pred.json`
- `runs/<run_id>/videos/<video_id>/inputs/source.json`
- `runs/<run_id>/videos/<video_id>/export/trajectories_pred.json`
- `runs/<run_id>/videos/<video_id>/export/relation_qc.json`
- `runs/<run_id>/run_manifest.json`
- `runs/<run_id>/reports/presence_report.md` if Gold evaluation ran

如果是区间运行（manifest 中出现 `state=partial`），则不应向用户承诺存在 export/pred 产物；应提示继续运行到 `export` 或用 `--resume` 完整跑通。

## 失败处理

命令失败时：

1. 读取 `runs/<run_id>/run_manifest.json`。
2. 找到 `state=failed` 的视频。
3. 查看失败节点的 `status.json` 和 `run.log`。
4. 优先修复环境、配置或数据问题。
5. 使用同一命令加 `--resume` 恢复。
6. 只有确认缓存过期时才使用 `--force --from_node <node>`。

## 中文诊断建议

- `missing DASHSCOPE api key`：设置 `DASHSCOPE_API_KEY`，或在命令中传入 `--api_key`。
- Rex-Omni 导入失败：检查 `configs/default.json` 的 `detector.rex_model_path` 和本地依赖；也可临时改用 DINO-X。
- DINO-X token 缺失：设置 `DINOX_API_TOKEN`。
- 视频下载失败：检查 URL、网络和 `video_ingest.download_timeout_sec`。
- `required output missing`：查看该节点 `run.log`，修复后用 `--force --from_node <node>` 重跑该节点及后续节点。
- VL 模型输出无法解析：优先检查该节点 Prompt、模型返回文本和 `dry_run` 配置。

## 汇报格式

完成后总结：

- 运行目录。
- 成功和失败视频数量。
- 最终关系文件路径。
- 如果有评测，给出评测报告路径。
- 主要失败原因和对应节点。
- 可参考 `docs/RUN_REPORT_TEMPLATE.md` 组织完整报告。
