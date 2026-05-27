# VIDVRD 运行报告

## 基本信息

- 运行目录：`runs/smoke_eval`
- 开始时间：2026-05-26 14:12:54
- 结束时间：2026-05-26 14:12:58
- 配置文件：`configs/dry_run_eval.json`
- 输入视频数量：1
- 成功：1，失败：0，跳过：0，部分完成：0
- 最终关系数：2
- 最终关系文件：`runs/smoke_eval/pred/relations_pred.json`
- resume：False，force：True，dry_run_relations：True
- api_key_present：True

## 评测

- 状态：succeeded
- 报告：`runs/smoke_eval/reports/presence_report.md`

## 节点状态

| 节点 | 成功 | 失败 | 跳过/其他 |
|---|---:|---:|---:|
| video_ingest | 1 | 0 | 0 |
| audio_prior | 1 | 0 | 0 |
| step1_detect | 1 | 0 | 0 |
| keyframe_screen | 1 | 0 | 0 |
| step2_track | 1 | 0 | 0 |
| track_qc | 1 | 0 | 0 |
| relation_rule | 1 | 0 | 0 |
| relation_llm | 1 | 0 | 0 |
| relation_merge | 1 | 0 | 0 |
| global_relation | 1 | 0 | 0 |
| relation_verify | 1 | 0 | 0 |
| export | 1 | 0 | 0 |

## 视频明细

### validation_dummy
- state: succeeded
- outputs: `{'relations_pred_json': 'runs/smoke_eval/videos/validation_dummy/export/relations_pred.json', 'trajectories_pred_json': 'runs/smoke_eval/videos/validation_dummy/export/trajectories_pred.json'}`

## 下一步建议

- 若全部 SKIP 且需重跑：使用新 `--run_dir` 或 `--force --from_node <node>`
- 若部分失败：修复后原命令加 `--resume`
- 正式检测：确认 Rex-Omni 模型或 `DINOX_API_TOKEN`，使用 `configs/production_full.json`
