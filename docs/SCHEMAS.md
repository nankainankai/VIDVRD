# 数据格式

轨迹字典的帧号使用十进制字符串。数据契约分为两层：

- Python 规范化 schema 使用半开区间 `[start_frame, end_frame)`；
- 既有 v1 JSON 产物和 Gold 仍使用闭区间 `[start_frame, end_frame]`，以免旧缓存静默错位。

`FrameSpan.from_values(..., convention="inclusive")` 和 `Relation.from_dict()` 会把闭区间转换为规范化半开区间。Python schema 输出包含 `"span_convention": "half_open"`；所有运行时关系 JSON 在写盘边界统一转换为闭区间并显式写入 `"span_convention": "inclusive"`。读取未标记的旧产物时仍按 legacy inclusive 解释。

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
{"frame": 1, "timestamp": 0.04, "objects": [], "detection_batch": {"status": "skipped", "source": "sparse_schedule", "detection_interval": 3}}
```

检测对象包含 `bbox`、`raw_class_name`、标准化 `class_name`、分数和批次来源。规范化 schema 同时读取旧 `confidence` 和新 `score` 字段；`score_kind` 标明其标度。没有原生检测分数时写 `null`/`unavailable`。旧 Rex 产物中的常量 `confidence=1.0` 只存在于归档结果中。

## 轨迹

`track/tracks.jsonl` 每帧一行：

```json
{"frame": 1, "tracks": [{"track_id": 1, "bbox": [5, 5, 25, 35], "bbox_observed": [5, 5, 25, 35], "box_source": "observed", "class_name": "person", "confidence": null, "track_status": "confirmed", "is_predicted": false}]}
```

`box_source`：

- `observed`：来自 Rex-Omni 且被跟踪器接收。
- `interpolated`：main 在两个真实观测之间补齐的不超过 8 帧的短缺口。
- `predicted`：旧产物或未来显式预测使用。

`track_status` 区分 `tentative` 与 `confirmed`；`track_quality` 保存真实观测数、距最近观测帧数和类别稳定度。旧产物缺少这些字段时由兼容层填充，不改变原 JSON。

当前正式路线直接使用 OC-SORT 经适配器映射后的 `track_id`。OC-SORT 的更新步按检测锚点计，输出帧号和插值间隔按真实视频帧计。`track/tracklets.json` 和 `track/stitch_links.json` 为兼容旧产物保留，但在正式路线中分别为空列表和 `enabled=false`。

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
    "end_frame": 45,
    "span_convention": "half_open",
    "rule_support": 0.76,
    "agent_score": 0.82,
    "ranking_score": 0.82,
    "score_kind": "mixed_ranking",
    "evidence_frames": [3, 11, 18, 26, 34],
    "predicate_components": {"action": "walk", "spatial": "behind", "comparative": ""},
    "source": "cross_window_aggregate",
    "segment_count": 2
  }]
}
```

`rule_support` 是几何规则在证据片段内的加权支持率，`agent_score` 是模型给出的排序分，两者不共享概率语义。`ranking_score` 仅供统一排序。以上是 Python 规范化 schema 示例；实际 `pred/relations.json` 会写成等价的闭区间，例如 `end_frame=44`，并显式标记 `span_convention=inclusive`。官方和诊断评测均按该字段读取，未标记旧产物仍按 inclusive 处理。

## Gold

- `gold/vidvrd_50_relations.json`：官方关系转成项目 v1 闭区间；规范化读取时转换回半开区间。
- `gold/vidvrd_50_trajectories.json`：官方 TID、类别和逐帧框。
- `gold/vidvrd_50_manifest.json`：来源路径、视频/轨迹/关系计数和谓词分布。

## 评测

`reports/official_vidvrd.json` 包含：

- `evaluator=imagenet_vidvrd_official_2017_compatible_v1` 和 tube vIoU 阈值；
- `relation_detection.mean_ap`、`recall_at.50/100`；
- `relation_tagging.precision_at.1/5/10`；
- `dataset_scope`：split、预期视频数、缺失/额外/显式空预测视频及是否完整覆盖官方测试集；
- `gold_adapter`、`prediction_adapter`：项目轨迹转连续 tube 时的数量记录。

`reports/diagnostic_track_aligned.json` 保留旧版全轨迹匈牙利对齐分析，所有指标使用 `diagnostic_` 前缀，不得作为官方 VidVRD mAP 报告。

`run_manifest.json` 汇总配置哈希、完整输入文件 SHA-256、运行模式、schema 版本、时间区间约定、模型与 prompt 版本、Git revision、当前源码树 fingerprint、每个阶段状态和最终产物路径。

## Agent-lite 证据与动作

`semantic/evidence_packets.json` 保存语义 Agent 实际看到的证据边界。每个 `EvidencePacket` 至少包含：视频与窗口标识、闭区间帧范围、已展示帧、当前轨迹对共同可见帧、两个轨迹的 ID 与类别、允许判断的有向候选谓词及谓词族、轨迹特征、框来源统计、`candidate_policy`、`evidence_mode`，以及允许补看的最大帧数。初始 packet 使用 `hierarchical_predicate_v1` 和 `event_burst_dual_view`。同一对象对的连续 packet 最多 6 个组成一次请求，模型以 `packet_results[].packet_id` 分别返回动作；若 Agent 申请补帧或扩候选，补充调用使用独立的 `:supplemental` packet 并一并落盘。

语义阶段只接受以下动作：

- `accept_relation`：必须给出 packet 内的有向轨迹对、候选谓词、区间、已展示证据帧、排序分数和理由；
- `reject_relation`、`defer_for_review`：不产生关系；
- `request_more_frames`：只能选择 packet 中尚未展示的共同可见帧，并受 `max_additional_frames` 限制；
- `request_candidate_expansion`：只能为 packet 内的一个有向目标对申请已登记的 `expandable_families`，最多一次，扩展后候选总数不超过 `expanded_candidate_limit`。

补帧与扩候选可以在同一初始响应中同时申请，但共享一次补充调用；补充 packet 的预算为零，不能形成循环。

最终复核阶段只接受 `accept_relation`、`reject_relation`、`change_predicate`、`refine_interval` 和 `defer_for_review`。动作必须引用既有 `relation_id`；谓词只能来自官方词表；区间只能收窄，证据帧只能取自该关系已有证据。Agent 不能触发重检测、重跟踪、外部写入、模型升级或循环调用。

语义调用批次与原始响应写入 `semantic/run.log.batch_audit`；逐 packet 校验结果和被拒动作写入 `agent_audit`。复核动作、被拒动作以及应用前后快照写入 `verify/qc.json`。互斥规则产生的内部 `delete` 是确定性程序动作，不属于 AgentAction。
