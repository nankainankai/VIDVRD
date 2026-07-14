# 语义关系标注改进计划

> 目标：让系统能可靠识别 ride、hold、follow、chase、sing_with 等真实语义关系，而非仅限于 left/right/near 等几何空间关系。
> 验收：在 30-50 条 Gold 小集上报告每类关系 Presence P/R/F1，总体 F1 ≥ 0.5。

---

## 现状问题

1. **检测只有 person**：`configs/default.json` 中 `rex_categories: "person"`，skateboard/microphone/horse 等关键物体从未被检测，关系无从谈起。
2. **谓词只有 13 个空间/运动类**：`configs/predicate_taxonomy.json` 没有 ride/hold/carry/hug/sing_with 等动作/交互谓词。
3. **Storyboard 全景缩略图**：VL 模型看的是 360px 高的 4 列拼图，小物体几个像素，看不清交互细节。
4. **Prompt 按谓词分组盲问**：每次只问 3 个谓词，模型不知道物体类别，只能从框位置猜。
5. **Verify 不执行修正**：检测出冲突和低置信度后只写报告，最终输出原封不动。
6. **video_id 不对齐**：`semi_auto_label_relations.py` 用 `video_path.stem`，pipeline 用 `video_id_for_source()`，merge 可能丢关系。
7. **resume 丢数据**：`semi_auto` 的 `--resume` 不加载已有 `output_json`，中断恢复时已完成窗口的关系丢失。

---

## 第 1 周：关系定义 + 检测类别 + Gold 小集

### 1.1 扩展 `configs/predicate_taxonomy.json`

新增三层谓词（在现有 13 个基础上增加到 ~25 个）：

```
接触/承载关系（category: "contact_action"）：
  ride       骑/乘    inverse=""     subject 约束: person  object 约束: vehicle/animal
  sit_on     坐在上面  inverse=""     subject: person  object: furniture/surface
  hold       握/拿    inverse=""     subject: person  object: object
  carry      携带/背   inverse=""     subject: person  object: object
  wear       穿戴     inverse=""     subject: person  object: object
  on         在上面    inverse="under" mutex_group="on_under"
  under      在下面    inverse="on"    mutex_group="on_under"

人物互动关系（category: "interaction"）：
  chase      追       inverse=""     subject: person  object: person/animal
  hug        拥抱     inverse="hug"  subject: person  object: person
  kick       踢       inverse=""     subject: person  object: object
  push       推       inverse=""     subject: person  object: person/object
  talk_to    对话     inverse="talk_to" subject: person  object: person
  look_at    注视     inverse=""     subject: person  object: any
  walk_with  同行     inverse="walk_with" subject: person  object: person
  play_with  玩耍     inverse="play_with" subject: person  object: any

音频辅助关系（category: "audio_assisted"）：
  sing_with  对唱/合唱 inverse="sing_with" subject: person  object: person
```

每个谓词新增字段：
- `subject_categories`: 允许的主体类别列表（用于候选过滤）
- `object_categories`: 允许的客体类别列表
- `candidate_triggers`: 什么条件自动生成候选（见第 3 周）

**改动文件**：`configs/predicate_taxonomy.json`

### 1.2 扩展检测类别

**改动文件**：`configs/default.json`

```diff
- "rex_categories": "person",
+ "rex_categories": "person,skateboard,bicycle,horse,dog,cat,ball,guitar,microphone,chair,table,car,bag,cup,phone,surfboard",
```

同时新建 `configs/semantic_relations.json`（覆盖 default，用于语义关系实验）：

```json
{
  "detector": {
    "rex_categories": "person,skateboard,bicycle,horse,dog,cat,ball,guitar,microphone,chair,table,car,bag,cup,phone,surfboard"
  },
  "relations": {
    "api_model": "qwen-vl-max",
    "group_size": 0,
    "max_frames_per_window": 8
  },
  "relation_rule": {
    "enabled": true,
    "object_aware_candidates": true
  },
  "relation_verify": {
    "low_confidence_threshold": 0.35,
    "apply_actions": true,
    "category_constraints_enabled": true
  }
}
```

**运行命令**：
```bash
conda run -n vidvrd python -m vidvrd_auto.cli \
  --videos data/videos_semantic.txt \
  --run_dir runs/semantic_v1 \
  --config configs/semantic_relations.json \
  --resume
```

### 1.3 构建 Gold 小集

**来源**：
- `data/manual_samples/add_VidVRD/` 已有 2 个视频的空间关系标注，可复用
- 新增 10-15 个包含目标场景的视频（骑车、追逐、唱歌、滑板等）
- 每个视频标注 2-5 条关系

**Gold 格式**（兼容现有 `evaluate_presence.py`）：

```json
{
  "video_id_1": [
    {
      "subject_track_id": 0,
      "subject_category": "person",
      "object_track_id": 1,
      "object_category": "skateboard",
      "predicate": "ride",
      "start_frame": 30,
      "end_frame": 120,
      "audio_assisted": false
    }
  ]
}
```

**输出文件**：`gold/relations_gold_semantic.json`

**验收标准**：
- Gold 文件存在，≥ 30 条关系标注
- 覆盖至少 8 种谓词（含 3+ 种非空间谓词）
- `evaluate_presence.py` 能正常读取并计算 P/R/F1

---

## 第 2 周：改善 VL 模型的视觉输入

### 2.1 Pair 裁剪 Storyboard

**核心思路**：不改 VL 调用方式（还是发一张图），但图片内容从全景缩略图改为 pair 局部放大图。

**改动文件**：`my_scripts/semi_auto_label_relations.py`

新增函数 `_make_pair_storyboard()`：

```python
def _make_pair_storyboard(frames, frame_indices, track_a_bboxes, track_b_bboxes,
                          tid_a, tid_b, tile_h=360):
    """为一对 track 生成裁剪放大的 storyboard。
    
    对每帧：计算包含 A、B 的最小矩形 → 外扩 40% → 裁剪 → 在裁剪图上标注 A(红)、B(蓝)。
    拼成 2×4 网格，每帧标注时间戳。
    """
    crops = []
    for frame, fi, bbox_a, bbox_b in zip(frames, frame_indices, 
                                          track_a_bboxes, track_b_bboxes):
        if bbox_a is None or bbox_b is None:
            continue
        # 计算包含两个 bbox 的 union，外扩 40%
        union = [min(bbox_a[0], bbox_b[0]), min(bbox_a[1], bbox_b[1]),
                 max(bbox_a[2], bbox_b[2]), max(bbox_a[3], bbox_b[3])]
        w = union[2] - union[0]
        h = union[3] - union[1]
        expand = 0.4
        union[0] = max(0, union[0] - w * expand)
        union[1] = max(0, union[1] - h * expand)
        union[2] = min(frame.shape[1], union[2] + w * expand)
        union[3] = min(frame.shape[0], union[3] + h * expand)
        
        crop = frame[int(union[1]):int(union[3]), int(union[0]):int(union[2])]
        # 在 crop 坐标系下画 A(红) B(蓝)
        # 标注 "A: person" / "B: skateboard" + 时间戳 "t=1.2s"
        draw_track_on_crop(crop, bbox_a, union, f"A(ID{tid_a})", (0,0,255))
        draw_track_on_crop(crop, bbox_b, union, f"B(ID{tid_b})", (255,128,0))
        draw_timestamp(crop, fi, fps)
        crops.append(crop)
    return _make_storyboard(crops, tile_h)
```

**改造调用逻辑**：从"全景 storyboard × N 组谓词"改为"pair storyboard × 每对一次全量查询"。

当前逻辑：
```
for window in windows:
    storyboard = 全景(all tracks)
    for group in chunk(predicates, 3):
        ask VL(storyboard, group)    # N 次调用
```

改为：
```
for window in windows:
    for (track_a, track_b) in pairs_in_window:
        pair_sb = pair_storyboard(track_a, track_b)
        candidate_preds = get_candidate_predicates(track_a.class, track_b.class)
        ask VL(pair_sb, candidate_preds)    # 每 pair 一次调用
```

**API 调用量变化**：
- 旧：每窗口 3-4 次（按谓词分组）× 窗口数
- 新：每窗口的 pair 数 × 1 次
- 窗口内通常 2-6 对有效 pair，所以调用量相近但每次查询更精准

### 2.2 修复 video_id 对齐

**改动文件**：`src/vidvrd_auto/relations/clip_classifier.py`

```diff
  cmd = [
      python_executable(),
      str(scripts / "semi_auto_label_relations.py"),
      "--windows_json", str(windows_json),
      "--tracks_jsonl", str(tracks_jsonl),
      "--output_json", str(out_json),
+     "--video_id", video_id,  # 显式传入主链 video_id
      ...
  ]
```

**改动文件**：`my_scripts/semi_auto_label_relations.py`

```diff
+ ap.add_argument("--video_id", type=str, default="", help="Override video_id for output JSON key")
  ...
- video_id = video_path.stem
+ video_id = (args.video_id or "").strip() or video_path.stem
```

### 2.3 修复 resume 丢数据

**改动文件**：`my_scripts/semi_auto_label_relations.py`

在 `main()` 开头，resume 模式下加载已有输出：

```python
all_relations: List[Dict[str, Any]] = []
if args.resume and out_path.exists():
    try:
        existing = _safe_read_json(out_path)
        if isinstance(existing, dict):
            existing_rels = existing.get(video_id, [])
            if isinstance(existing_rels, list):
                all_relations = [r for r in existing_rels if isinstance(r, dict)]
                print(f"RESUME: loaded {len(all_relations)} existing relations from {out_path}")
    except Exception:
        pass
```

### 2.4 给 verify/global_relation 传真实图片

当前 `relation_verify_prompt` 和 `global_relation_prompt` 只发 JSON 文本摘要给 VL 模型，没有图片。

**改动文件**：`src/vidvrd_auto/nodes/global_relation.py`、`src/vidvrd_auto/relations/ops.py`

在调用 `VLClient.call()` 时，传入该窗口对应的 storyboard 图片路径：

```diff
- vl_result = VLClient(client_cfg).call(prompt=..., dry_run=vl_dry_run)
+ storyboard_paths = list(storyboards_dir.glob("seg_*.jpg")) if storyboards_dir else []
+ vl_result = VLClient(client_cfg).call(
+     prompt=..., 
+     image_paths=storyboard_paths[:4],  # 最多 4 张代表性证据图
+     dry_run=vl_dry_run
+ )
```

**验收**：VL 请求日志中包含 `image` 字段，不再是纯文本。

---

## 第 3 周：Object-Aware 候选生成

### 3.1 新增候选规则引擎

**新增文件**：`src/vidvrd_auto/relations/object_candidates.py`

```python
"""基于物体类别对的候选谓词生成。

不判定关系是否成立，只生成"值得让 VL 验证"的候选列表。
"""

# 类别对 → 候选谓词
PAIR_CANDIDATES = {
    ("person", "skateboard"): ["ride", "on", "hold", "push", "near"],
    ("person", "bicycle"):    ["ride", "on", "push", "near"],
    ("person", "horse"):      ["ride", "on", "near", "follow"],
    ("person", "dog"):        ["walk_with", "hold", "follow", "chase", "near", "play_with"],
    ("person", "cat"):        ["hold", "carry", "near", "play_with"],
    ("person", "ball"):       ["hold", "kick", "throw", "carry", "near"],
    ("person", "guitar"):     ["hold", "play_with", "carry"],
    ("person", "microphone"): ["hold", "near"],
    ("person", "chair"):      ["sit_on", "near", "push"],
    ("person", "car"):        ["ride", "near", "push"],
    ("person", "surfboard"):  ["ride", "on", "hold", "near"],
    ("person", "bag"):        ["hold", "carry", "wear"],
    ("person", "cup"):        ["hold", "near"],
    ("person", "phone"):      ["hold", "look_at", "near"],
    ("person", "person"):     [
        "near", "follow", "chase", "hug", "push", "kick",
        "talk_to", "look_at", "walk_with", "sing_with", "play_with"
    ],
}

# 对称处理：(A, B) 和 (B, A) 可能需要不同谓词
SYMMETRIC_PREDICATES = {"near", "hug", "talk_to", "walk_with", "sing_with", "play_with"}

def get_candidate_predicates(subject_class: str, object_class: str,
                             audio_label: str = "") -> list[str]:
    """返回该物体对值得验证的候选谓词列表。"""
    s = subject_class.lower().strip()
    o = object_class.lower().strip()
    
    candidates = list(PAIR_CANDIDATES.get((s, o), []))
    
    # 反向查找
    if not candidates:
        reverse = PAIR_CANDIDATES.get((o, s), [])
        candidates = [p for p in reverse if p in SYMMETRIC_PREDICATES]
    
    # 兜底：任意 pair 至少查空间关系
    if not candidates:
        candidates = ["near", "overlap"]
    
    # 音频辅助：如果音频提示唱歌且是 person-person
    if audio_label and s == "person" and o == "person":
        al = audio_label.lower()
        if "sing" in al and "sing_with" not in candidates:
            candidates.insert(0, "sing_with")
        if ("speech" in al or "talk" in al) and "talk_to" not in candidates:
            candidates.insert(0, "talk_to")
    
    return candidates
```

### 3.2 改造 relation_rule 加入 object-aware 候选

**改动文件**：`src/vidvrd_auto/relations/ops.py` 的 `generate_rule_relations()`

在现有几何规则之后，新增一段：

```python
# ---- Object-aware 候选 ----
if bool(config.get("object_aware_candidates", False)):
    from vidvrd_auto.relations.object_candidates import get_candidate_predicates
    
    # 收集每个 track 的主类别
    track_class = {}  # track_id -> most_common_class
    for frame_tracks in tracks_by_frame.values():
        for tid, t in frame_tracks.items():
            cls = str(t.get("class_name", "unknown")).lower()
            track_class.setdefault(tid, Counter())[cls] += 1
    track_main_class = {tid: counter.most_common(1)[0][0] 
                        for tid, counter in track_class.items()}
    
    # 对每个窗口的每对 track，生成类别候选
    for wi, w in enumerate(windows):
        # ... 取 track_ids, start, end ...
        for sid, oid in combinations(track_ids, 2):
            s_cls = track_main_class.get(sid, "unknown")
            o_cls = track_main_class.get(oid, "unknown")
            cands = get_candidate_predicates(s_cls, o_cls, audio_label)
            for pred in cands:
                if pred in {"near", "overlap", "left", "right", ...}:
                    continue  # 几何关系已由上面的规则处理
                relations.append({
                    "subject_track_id": sid,
                    "object_track_id": oid,
                    "predicate": pred,
                    "start_frame": start,
                    "end_frame": end,
                    "confidence": 0.15,  # 低置信度候选，待 VL 验证
                    "source": "candidate_object_aware",
                    "segment_id": segment_id,
                    "subject_category": s_cls,
                    "object_category": o_cls,
                    "evidence": f"candidate from {s_cls}-{o_cls} pair",
                })
```

### 3.3 改造 relation_llm Prompt

**改动文件**：`my_scripts/semi_auto_label_relations.py`

将 prompt 从"给谓词列表让模型泛选"改为"给具体 pair + 候选谓词让模型验证"：

```python
prompt = f"""你是视频关系标注专家。

## 画面说明
这是物体 A（红框，{class_a}）和物体 B（蓝框，{class_b}）在 {len(frames_idx)} 帧中的裁剪放大图。
帧按时间从左到右、从上到下排列，每帧标注了时间戳。
{audio_hint}

## 候选关系
以下是根据物体类别和轨迹生成的候选关系，请判断每个是否成立：
{chr(10).join(f'- {p}' for p in candidate_preds)}

## 输出格式（JSON）
{{
  "relations": [
    {{"predicate": "ride", "confidence": 0.85, "evidence": "A 站在 B 上且同步移动"}}
  ],
  "scene": "一句话描述 A 和 B 之间发生了什么"
}}

要求：
- 只输出有把握的关系，不确定的不要输出
- confidence 范围 0~1，0.7 以上表示较有把握
- evidence 要说明具体视觉证据（如"A 的框包含在 B 框内""两者距离持续缩小"）
- predicate 必须从上面的候选列表中选择
"""
```

**验收**：
- `person + skateboard` pair 必定收到 `ride/on/hold` 候选
- prompt 中包含物体类别名称
- VL 模型可以通过 pair 裁剪图看清两个物体的交互

---

## 第 4 周：让 Verify 真正影响输出

### 4.1 实现 actions 执行

**改动文件**：`src/vidvrd_auto/relations/ops.py` 的 `verify_relations()`

在第 416 行之前插入 action 执行逻辑：

```python
# ---- 应用 final_actions ----
if bool(config.get("apply_actions", False)) and final_actions:
    to_delete = set()
    for act in final_actions:
        idx = act.get("index")
        action = str(act.get("action", "")).strip().lower()
        if not isinstance(idx, int) or idx < 0 or idx >= len(items):
            continue
        if action == "delete":
            to_delete.add(idx)
        elif action == "change_predicate":
            new_pred = str(act.get("new_predicate", "")).strip()
            if new_pred:
                items[idx]["predicate"] = new_pred
                items[idx]["source"] = "verify_corrected"
        elif action == "adjust_span":
            if act.get("start_frame") is not None:
                items[idx]["start_frame"] = int(act["start_frame"])
            if act.get("end_frame") is not None:
                items[idx]["end_frame"] = int(act["end_frame"])
    if to_delete:
        items = [item for i, item in enumerate(items) if i not in to_delete]

# ---- 互斥冲突自动消解 ----
items = _resolve_mutex_conflicts(items)

# ---- 最低置信度过滤 ----
min_export_conf = float(config.get("min_export_confidence", 0.0))
if min_export_conf > 0:
    items = [item for item in items 
             if float(item.get("confidence", 0.0) or 0.0) >= min_export_conf]
```

### 4.2 互斥冲突自动消解

```python
def _resolve_mutex_conflicts(items):
    """同一 pair+span 上有互斥谓词时，保留置信度更高的一方。"""
    by_pair_span = {}
    for i, item in enumerate(items):
        key = _rel_key(item)
        if key is None:
            continue
        sid, pred, oid, start, end = key
        pair_key = (sid, oid, start, end)
        by_pair_span.setdefault(pair_key, []).append((i, pred, float(item.get("confidence", 0.0))))
    
    to_delete = set()
    for pair_key, entries in by_pair_span.items():
        preds = {pred for _, pred, _ in entries}
        for mutex_pair in MUTEX_PAIRS:
            if mutex_pair.issubset(preds):
                p1, p2 = sorted(mutex_pair)
                c1 = max((c for _, p, c in entries if p == p1), default=0.0)
                c2 = max((c for _, p, c in entries if p == p2), default=0.0)
                drop_pred = p1 if c1 <= c2 else p2
                for idx, pred, _ in entries:
                    if pred == drop_pred:
                        to_delete.add(idx)
    
    return [item for i, item in enumerate(items) if i not in to_delete]
```

### 4.3 类别约束过滤

**改动文件**：`src/vidvrd_auto/relations/ops.py`

```python
def _category_constraint_check(item, track_classes, taxonomy):
    """检查关系的主/客体类别是否满足谓词约束。"""
    pred = str(item.get("predicate", "")).strip()
    pred_def = taxonomy.get(pred, {})
    
    subj_cats = pred_def.get("subject_categories", [])
    obj_cats = pred_def.get("object_categories", [])
    
    if not subj_cats and not obj_cats:
        return True  # 无约束
    
    sid = item.get("subject_track_id")
    oid = item.get("object_track_id")
    s_cls = track_classes.get(sid, "unknown").lower()
    o_cls = track_classes.get(oid, "unknown").lower()
    
    if subj_cats and s_cls not in [c.lower() for c in subj_cats]:
        return False  # ride 的 subject 必须是 person
    if obj_cats and o_cls not in [c.lower() for c in obj_cats]:
        return False  # ride 的 object 必须是 vehicle/animal
    
    return True
```

在 verify 末尾加入过滤：

```python
if bool(config.get("category_constraints_enabled", False)):
    preds_def = predicate_defs()
    items = [item for item in items 
             if _category_constraint_check(item, track_main_classes, preds_def)]
```

**验收**：
- 互斥关系（如同时有 left 和 right）只保留一个
- confidence < 0.35 的关系不进入最终输出
- `ride(person, person)` 会被类别约束过滤掉

---

## 第 5 周：端到端评测与性能调参

### 5.1 运行全量评测

```bash
conda run -n vidvrd python -m vidvrd_auto.cli \
  --videos data/videos_semantic.txt \
  --run_dir runs/semantic_eval \
  --config configs/semantic_relations.json \
  --resume

# 评测
conda run -n vidvrd python tools/evaluate_presence.py \
  --gold gold/relations_gold_semantic.json \
  --pred runs/semantic_eval/pred/relations_pred.json \
  --report runs/semantic_eval/reports/semantic_report.md
```

### 5.2 分谓词评测报告

**改动文件**：`tools/evaluate_presence.py`

在现有 per-video 表之后，新增 per-predicate 分析：

```python
# Per Predicate
pred_tp, pred_fp, pred_fn = Counter(), Counter(), Counter()
for vid in video_ids:
    g = gold_map.get(vid, {})
    p = pred_map.get(vid, {})
    for key in g.keys() & p.keys():
        pred_tp[key[1]] += 1       # key = (sid, pred, oid)
    for key in p.keys() - g.keys():
        pred_fp[key[1]] += 1
    for key in g.keys() - p.keys():
        pred_fn[key[1]] += 1
```

输出表：
```
| predicate | TP | FP | FN | Precision | Recall | F1 |
|-----------|---:|---:|---:|----------:|-------:|---:|
| ride      |  3 |  1 |  2 |     0.750 |  0.600 | 0.667 |
| follow    |  2 |  3 |  1 |     0.400 |  0.667 | 0.500 |
| left      |  8 |  2 |  3 |     0.800 |  0.727 | 0.762 |
| sing_with |  0 |  0 |  2 |     0.000 |  0.000 | 0.000 |
```

### 5.3 错误分析模板

对每个 FN（漏检），分析原因：

```
## 失败案例分析

### FN: video_001, (0, ride, 1) 
- 根因：检测器未检测到 skateboard（rex_categories 中没有）
- 修复：扩展检测类别

### FN: video_003, (0, sing_with, 1)
- 根因：audio_prior 返回空标签，VL 未收到音频提示
- 修复：确保 VGGSound CSV 覆盖该视频

### FP: video_005, (0, ride, 2)
- 根因：person 和 bicycle 框重叠 > 0.5，候选规则生成了 ride，但实际人在自行车旁边站着
- 修复：提高 ride 的 VL 验证门槛
```

### 5.4 性能调参

```json
{
  "tracking": {
    "export_pair_viz_videos": false,
    "enable_llm_qc": false
  },
  "relations": {
    "max_windows": 20,
    "sleep_sec": 0.2
  }
}
```

关闭非必要的 pair 视频导出和 Step2 LLM QC，减少 ~40% 运行时间。

---

## 改动文件汇总

| 文件 | 改动内容 |
|------|----------|
| `configs/predicate_taxonomy.json` | 新增 ~12 个语义谓词 + subject/object 类别约束 |
| `configs/default.json` | 扩展 rex_categories |
| `configs/semantic_relations.json` | 新建：语义关系实验配置 |
| `gold/relations_gold_semantic.json` | 新建：30-50 条 Gold 标注 |
| **`src/vidvrd_auto/relations/object_candidates.py`** | **新建：类别对候选谓词引擎** |
| `src/vidvrd_auto/relations/ops.py` | verify 执行 actions + 互斥消解 + 类别约束 + 最低置信度 |
| `src/vidvrd_auto/relations/taxonomy.py` | 新增 subject/object_categories 读取 |
| `src/vidvrd_auto/relations/clip_classifier.py` | 传 video_id 给旧脚本 |
| `src/vidvrd_auto/prompts/templates.py` | 新增 pair-centric prompt 模板 |
| `my_scripts/semi_auto_label_relations.py` | pair storyboard + 新 prompt + resume 修复 + video_id 参数 |
| `tools/evaluate_presence.py` | 新增 per-predicate 评测表 |
| `data/videos_semantic.txt` | 新建：目标场景视频列表 |

---

## 不做什么（明确排除）

1. **不做视频片段输入**：API 成本和延迟不可控，storyboard 优化足够
2. **不训练自定义模型**：依赖 VL 零样本能力 + 规则引擎，不引入训练流程
3. **不做全谓词覆盖**：先聚焦 25 个高价值谓词，跑通再扩
4. **不重构 subprocess 调用**：旧脚本迁移是独立技术债，不在本计划内
5. **不做实时/在线推理**：保持批处理模式

---

## 里程碑

| 周次 | 交付物 | 验收标准 |
|------|--------|----------|
| W1 | taxonomy + 检测配置 + Gold | Gold ≥ 30 条，评测脚本能跑 |
| W2 | pair storyboard + video_id 修复 + resume 修复 | VL 请求中包含 pair 图片，不再丢关系 |
| W3 | object_candidates + 新 prompt | person-skateboard 必产 ride 候选 |
| W4 | verify 执行 + 冲突消解 + 类别约束 | 输出不含互斥冲突和非法类别关系 |
| W5 | 端到端评测报告 | 每类谓词 P/R/F1，总体 F1 ≥ 0.5 |
