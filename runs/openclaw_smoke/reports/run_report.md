# VIDVRD 运行报告

## 基本信息

- 运行目录：`runs/openclaw_smoke`
- 开始时间：2026-05-26 16:07:58
- 结束时间：2026-05-26 16:08:00
- 配置文件：`C:/Users/吴泽宇/.openclaw/workspace/configs/dry_run.json`
- 输入视频数量：1
- 成功：0，失败：1，跳过：0，部分完成：0
- 最终关系数：0
- 最终关系文件：`runs/openclaw_smoke/pred/relations_pred.json`
- resume：True，force：False，dry_run_relations：True
- api_key_present：True

## 评测

- 状态：skipped
- 原因：disabled

## 节点状态

| 节点 | 成功 | 失败 | 跳过/其他 |
|---|---:|---:|---:|
| video_ingest | 1 | 0 | 0 |
| audio_prior | 1 | 0 | 0 |
| step1_detect | 0 | 1 | 0 |
| keyframe_screen | 0 | 0 | 0 |
| step2_track | 0 | 0 | 0 |
| track_qc | 0 | 0 | 0 |
| relation_rule | 0 | 0 | 0 |
| relation_llm | 0 | 0 | 0 |
| relation_merge | 0 | 0 | 0 |
| global_relation | 0 | 0 | 0 |
| relation_verify | 0 | 0 | 0 |
| export | 0 | 0 | 0 |

## 视频明细

### validation_dummy
- state: failed
- error: required output missing: C:\Users\吴泽宇\Desktop\VIDVRD\runs\openclaw_smoke\videos\validation_dummy\step1_detect\detections_full.jsonl

## 失败与异常

- validation_dummy: required output missing: C:\Users\吴泽宇\Desktop\VIDVRD\runs\openclaw_smoke\videos\validation_dummy\step1_detect\detections_full.jsonl
- validation_dummy/step1_detect: required output missing: C:\Users\吴泽宇\Desktop\VIDVRD\runs\openclaw_smoke\videos\validation_dummy\step1_detect\detections_full.jsonl

## 下一步建议

- 若全部 SKIP 且需重跑：使用新 `--run_dir` 或 `--force --from_node <node>`
- 若部分失败：修复后原命令加 `--resume`
- 正式检测：确认 Rex-Omni 模型或 `DINOX_API_TOKEN`，使用 `configs/production_full.json`
