# Gold 标注样例

本目录存放与 `pred/relations_pred.json` schema 对齐的人工 Gold，供 Presence P/R/F1 评测。

## 文件

| 文件 | 说明 |
|------|------|
| `relations_gold.json` | 关系 Gold（主体/客体 track_id、谓词、起止帧） |
| `trajectories_gold.json` | 轨迹 Gold（可选，后续轨迹评测用） |

## 字段口径

与 `docs/SCHEMAS.md` 一致：使用 `subject_track_id` / `object_track_id`。

## 当前样例

`validation_dummy` 对应 `data/validation_dummy.mp4` smoke 视频，用于本地评测链路验证。

**「50 条 Gold」**：指约 50 个视频的完整人工标注（轨迹 + 时间段关系），见根目录 `plan.md` 与 `plan/plan.md` 第一阶段。当前仓库**不是** 50 条，仅为 smoke；标注组交付后替换本目录，并保证 `video_id` 与跑批视频一致。
