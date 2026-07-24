# 数据格式

关系时间段统一使用闭区间 `[start_frame, end_frame]`；轨迹字典的帧号使用十进制字符串。

## 视频词表

`vocabulary/objects.json`：

```json
{
  "mode": "open",
  "categories": ["person", "domestic_cat", "traffic light"],
  "label_map": {"cat": "domestic_cat"},
  "entries": [
    {"raw_label": "cat", "canonical_label": "domestic_cat", "ontology_source": "vidvrd"}
  ],
  "discovery": {"state": "succeeded"}
}
```

## 检测

`detect/detections.jsonl` 每帧一行：

```json
{"frame": 1, "timestamp": 0.04, "objects": [], "detection_batch": {"status": "skipped", "source": "tracker_propagation", "detection_interval": 5}}
```

检测对象包含 `bbox`、`raw_class_name`、标准化 `class_name`、`confidence` 和批次来源。

## 轨迹

`track/tracks.jsonl` 每帧一行：

```json
{"frame": 1, "tracks": [{"track_id": 1, "bbox": [5, 5, 25, 35], "bbox_observed": null, "box_source": "interpolated", "class_name": "person", "is_predicted": true}]}
```

`box_source`：

- `observed`：来自 Rex-Omni 且被跟踪器接收。
- `interpolated`：前后真实观测之间的短缺口插值。
- `predicted`：仅来自 OC-SORT 状态预测。

最终 `pred/trajectories.json`：

```json
{"video_id": [{"track_id": 1, "category": "person", "trajectory": {"0": [5, 5, 25, 35]}, "box_sources": {"0": "observed"}}]}
```

## 关系

`pred/relations.json`：

```json
{
  "video_id": [{
    "relation_id": "r000001",
    "subject_track_id": 1,
    "predicate": "walk_behind",
    "object_track_id": 2,
    "start_frame": 0,
    "end_frame": 44,
    "confidence": 0.82,
    "predicate_components": {"action": "walk", "spatial": "behind", "comparative": ""},
    "source": "cross_window_aggregate",
    "segment_count": 2
  }]
}
```

## Gold

- `gold/vidvrd_50_relations.json`：官方关系转成项目闭区间。
- `gold/vidvrd_50_trajectories.json`：官方 TID、类别和逐帧框。
- `gold/vidvrd_50_manifest.json`：来源路径、视频/轨迹/关系计数和谓词分布。

## 评测

`reports/metrics.json` 包含：

- `evaluated_videos`：本次预测文件中的视频作用域；部分试跑不会把其他 Gold 视频计为漏检。
- `tracks`：匹配数、Gold 召回、平均 vIoU 和 ID 映射。
- `overall`：TP/FP/FN、P/R/F1 和 micro AP。
- `mean_ap`：仅对 Gold 中出现的谓词求逐谓词 AP 均值。
- `splits`、`per_predicate`：base/novel 与逐谓词指标。
- `recall_at_50/100`、`tagging_precision_at_1/5/10`。
- `error_examples`：有限数量的 FP/FN 样例。

`run_manifest.json` 汇总配置哈希、输入、每个阶段状态和最终产物路径。
