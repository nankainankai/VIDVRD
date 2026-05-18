# 输出 Schema

所有生产运行产物都写入 `runs/<run_id>/`。本文件只记录主链稳定字段，节点可以额外写审计字段。

## run_manifest.json

```json
{
  "run_dir": "runs/exp001",
  "started_at": "YYYY-MM-DD HH:MM:SS",
  "finished_at": "YYYY-MM-DD HH:MM:SS",
  "config_path": "configs/default.json",
  "config_hash": "...",
  "nodes": ["video_ingest", "audio_prior", "step1_detect"],
  "args": {
    "video": "...",
    "videos": "...",
    "from_node": "",
    "to_node": "",
    "resume": true,
    "force": false,
    "dry_run_relations": false,
    "skip_eval": false,
    "api_key_present": true
  },
  "videos": [],
  "pred_relations_json": "runs/exp001/pred/relations_pred.json",
  "video_state_counts": {"succeeded": 1, "failed": 0, "skipped": 0, "partial": 0},
  "evaluate": {}
}
```

`videos[]` 的每个 item 至少包含 `video_id/source/nodes/state`：

- `state=succeeded`：本次运行实际生成了 `export/relations_pred.json` 和 `export/trajectories_pred.json`。
- `state=partial`：本次为区间运行（例如 `--to_node` 停在 export 之前），或未执行到 export；不会把该视频计入最终导出聚合。
- `state=skipped`：例如 `keyframe_screen` 判定 drop。
- `state=failed`：节点抛错或 required outputs 缺失。

## status.json

每个节点写 `runs/<run_id>/videos/<video_id>/<node>/status.json`。`input_hash` 用于判断 `--resume` 时是否可以复用缓存。

```json
{
  "node": "relation_merge",
  "state": "succeeded",
  "input_hash": "...",
  "finished_at": "YYYY-MM-DD HH:MM:SS",
  "outputs": {}
}
```

## relations_pred.json

```json
{
  "video_id": [
    {
      "subject_track_id": 1,
      "predicate": "left",
      "object_track_id": 2,
      "start_frame": 0,
      "end_frame": 30,
      "confidence": 0.8,
      "source": "rule_geometry"
    }
  ]
}
```

内部统一使用 `subject_track_id/object_track_id`。旧评测脚本可能仍接受 `subject_id/object_id`，但新导出优先使用 track id 字段。

## audio_prior.json

```json
{
  "enabled": true,
  "video_id": "demo",
  "label": "singing",
  "confidence": 1.0,
  "source": "vggsound_csv",
  "matched_key": "demo"
}
```

## screen_result.json

```json
{
  "passed": true,
  "decision": "keep",
  "reason": "ok",
  "max_frame_index": 240,
  "crop_suggestion": [0.0, 0.0, 640.0, 480.0],
  "vl_screen": {
    "enabled": true,
    "state": "succeeded",
    "model": "qwen-vl-max",
    "model_reason": "关键帧中存在足够人物且画面质量可用"
  }
}
```

`decision` 可取 `keep`、`drop`、`crop`。`drop` 会让当前视频跳过后续节点；`crop` 表示建议裁剪后继续。

## track_qc.json

```json
{
  "track_count": 3,
  "short_track_count": 0,
  "large_jump_count": 1,
  "vl_review": {
    "enabled": true,
    "state": "succeeded",
    "items": []
  },
  "passed": true
}
```

## relation_qc.json

```json
{
  "relation_count": 12,
  "low_confidence_count": 1,
  "conflict_count": 0,
  "strong_model_review_enabled": true,
  "strong_model_review_count": 1,
  "final_actions": []
}
```

`final_actions` 用于记录强模型复核后的动作，例如 `keep`、`delete`、`change_predicate`、`adjust_span`、`add_coupling`。
