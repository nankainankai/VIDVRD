# VIDVRD 运行报告

## 基本信息

- 运行目录：`runs/rexomni_full`
- 开始时间：2026-05-27 09:14:54
- 结束时间：2026-05-27 09:15:33
- 配置文件：`configs/rexomni_full.json`
- 输入视频数量：1
- 成功：0，失败：0，跳过：1，部分完成：0
- 最终关系数：0
- 最终关系文件：`runs/rexomni_full/pred/relations_pred.json`
- resume：True，force：False，dry_run_relations：False
- api_key_present：True

## 评测

- 状态：skipped
- 原因：disabled

## 节点状态

| 节点 | 成功 | 失败 | 跳过/其他 |
|---|---:|---:|---:|
| video_ingest | 1 | 0 | 0 |
| audio_prior | 1 | 0 | 0 |
| step1_detect | 1 | 0 | 0 |
| keyframe_screen | 1 | 0 | 0 |
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
- state: skipped
- skip_reason: not_enough_valid_objects

## 下一步建议

- 若全部 SKIP 且需重跑：使用新 `--run_dir` 或 `--force --from_node <node>`
- 若部分失败：修复后原命令加 `--resume`
- 正式检测：确认 Rex-Omni 模型或 `DINOX_API_TOKEN`，使用 `configs/production_full.json`
