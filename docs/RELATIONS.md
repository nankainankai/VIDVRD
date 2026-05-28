# 关系检测技术文档

本文档描述 VIDVRD Auto 中**视频物体关系（Video Relation Detection, VRD）**的谓词体系、检测流水线、配置项与已知局限。面向需要维护关系节点、扩展谓词或对接 Gold 评测的开发者。

相关文档：

- 系统总览：[`TECHNICAL.md`](TECHNICAL.md)
- 架构专篇：[`ARCHITECTURE.md`](ARCHITECTURE.md)
- JSON 产物字段：[`SCHEMAS.md`](SCHEMAS.md)
- 配置分层说明：[`../configs/CONFIGS.md`](../configs/CONFIGS.md)

---

## 1. 概述

关系检测是主链后半段的核心能力，输入为**轨迹窗口**（`windows.json` + `tracks_full.jsonl`），输出为**时序三元组**列表，写入 `export/relations_pred.json` 并汇总到 `runs/<id>/pred/relations_pred.json`。

每条关系记录表示：在帧区间 `[start_frame, end_frame]` 内，主体轨迹 `subject_track_id` 与客体轨迹 `object_track_id` 之间成立谓词 `predicate`，并附带置信度与来源。

检测采用**多源融合**策略：

| 来源 | 节点 | 特点 |
|------|------|------|
| 规则几何 | `relation_rule` | 确定性、可解释、覆盖空间/运动类 |
| 多模态 LLM | `relation_llm` | 语义理解、可补深度/交互类 |
| 耦合补全 | `relation_merge` | 由 inverse 定义自动补反向关系 |
| 跨窗聚合 | `global_relation` | 合并动态关系碎片 |
| 质检复核 | `relation_verify` | 互斥冲突、低置信度、可选强模型 |

---

## 2. 谓词分类体系

### 2.1 权威定义

**单一真相源**：[`configs/predicate_taxonomy.json`](../configs/predicate_taxonomy.json)

每个谓词包含以下元数据：

| 字段 | 含义 |
|------|------|
| `zh` | 中文释义 |
| `category` | 大类：`spatial` / `depth` / `motion` / `interaction` |
| `inverse` | 反向耦合谓词（如 `left` ↔ `right`）；空表示无定向反向 |
| `mutex_group` | 互斥组名；同组谓词不能在同一 (subject, object, 帧段) 上同时成立 |
| `requires_contact` | 是否强调接触/重叠 |
| `uses_motion` | 是否运动/时序类谓词 |

### 2.2 当前谓词表（16 个）

| 类别 | 谓词 | 中文 | 反向 | 互斥组 |
|------|------|------|------|--------|
| spatial | `left` | 在左侧 | `right` | horizontal |
| spatial | `right` | 在右侧 | `left` | horizontal |
| spatial | `above` | 在上方 | `below` | vertical |
| spatial | `below` | 在下方 | `above` | vertical |
| spatial | `near` | 靠近 | — | — |
| spatial | `overlap` | 重叠 | — | — |
| depth | `front` | 在前方 | `behind` | depth |
| depth | `behind` | 在后方 | `front` | depth |
| motion | `toward` | 朝向/接近 | `away` | relative_motion |
| motion | `away` | 远离 | `toward` | relative_motion |
| motion | `follow` | 跟随 | — | — |
| motion | `chase` | 追逐 | — | — |
| motion | `moving_together` | 共同移动 | — | — |
| interaction | `contact` | 接触 | — | — |
| interaction | `sing_with` | 与…一起唱 | — | sing |
| interaction | `sing_to` | 向…唱 | — | sing |

代码读取入口：[`src/vidvrd_auto/relations/taxonomy.py`](../src/vidvrd_auto/relations/taxonomy.py)

```python
load_taxonomy()           # 读 JSON
predicate_defs()          # 谓词 → 元数据
coupling_inverse()        # 反向表（merge / clip_relation 耦合用）
mutex_pairs()             # 互斥对集合（verify 冲突检测用）
prompt_predicate_summary() # 拼 Prompt 摘要（当前 VL 主 Prompt 未直接引用）
```

### 2.3 别名与规范化

[`src/vidvrd_auto/relations/predicate_aliases.py`](../src/vidvrd_auto/relations/predicate_aliases.py) 负责：

- **`canonical_predicate()`**：中文/口语/变体 → 英文 canonical（如「左侧」→ `left`）
- **`DEFAULT_RELATION_PREDICATES`**：VL 默认候选列表（10 个，见下节）
- **`audio_predicates_from_label()`**：VggSound 音频标签 → 额外候选（如含 `sing` → `sing_with`）

**最终导出**中 `predicate` 字段统一为英文 canonical 字符串。

### 2.4 各层使用的谓词范围（当前不一致点）

| 模块 | 使用的谓词 |
|------|-----------|
| `predicate_taxonomy.json` | 全部 16 个 |
| `DEFAULT_RELATION_PREDICATES`（VL 默认） | `left, right, front, behind, above, below, overlap, near, follow, toward` |
| `relation_rule` 规则产出 | `left, right, above, below, near, overlap, contact, toward, away, follow, chase, moving_together`（**不含** `front/behind/sing_*`） |
| `global_relation` 动态聚合 | 配置 `dynamic_predicates`，默认 `toward, away, follow, chase` |
| `relation_viz` 位置类过滤 | 硬编码 `POSITIONAL_PREDICATES` 子集 |

> **说明**：taxonomy 是完整词表；VL 默认只问 10 个谓词。要让 VL 覆盖全部 16 个，需在配置 `relations.relations` 中显式列出（见 §5）。

---

## 3. 检测流水线

主链中关系相关节点顺序（见 [`pipeline/constants.py`](../src/vidvrd_auto/pipeline/constants.py)）：

```text
track_qc
  └─ relation_rule      → relations_rule.json
  └─ relation_llm       → relations_llm.json (+ storyboards/)
  └─ relation_merge     → relations_merged.json
  └─ global_relation    → relations_global.json
  └─ relation_verify    → relations_verified.json, relation_qc.json
  └─ export             → relations_pred.json（最终导出）
```

### 3.1 relation_rule — 规则几何关系

**实现**：[`relations/ops.py`](../src/vidvrd_auto/relations/ops.py) → `generate_rule_relations()`

**输入**：每个 tracking window 内的轨迹对 `(subject, object)`，逐帧读取 bbox 中心与面积。

**算法概要**：

1. 对每个 window、每对轨迹，在 `[start_frame, end_frame]` 内逐帧统计谓词「投票」。
2. 空间关系：比较两框中心差 `dx, dy` 与尺度相关的 margin（`axis_margin_ratio × sqrt(area)`）。
   - `dx < -margin` → subject 在 object **left**
   - `dx > margin` → **right**
   - `dy` 同理 → **above** / **below**
3. 距离/接触：
   - 中心距 ≤ `near_distance_ratio × scale` → **near**
   - IoU ≥ `overlap_iou_threshold` → **overlap** 且 **contact**
4. 运动（需跨帧）：
   - 相对距离变化率 → **toward** / **away**
   - 速度向量对齐 + 距离约束 → **moving_together** / **follow** / **chase**
5. 某谓词在 window 内投票比例 ≥ `min_vote_ratio` 时输出，`confidence = 投票比例`，`source = rule_geometry`。

**不产出**：`front`、`behind`（无深度估计）、`sing_with`、`sing_to`（需语义/音频）。

### 3.2 relation_llm — 片段多模态关系

**实现**：[`relations/clip_relation.py`](../src/vidvrd_auto/relations/clip_relation.py) → `run_clip_relation()`

**流程**：

1. 按 window 从视频抽帧（约 1fps），在帧上绘制轨迹框与 ID，拼成 **storyboard** 图。
2. 构建谓词候选列表 `_build_predicate_list(config)`：
   - 若 `relations.relations` 非空 → 逗号分隔列表
   - 否则 → `DEFAULT_RELATION_PREDICATES`
   - 再合并 `vggsound_label` 触发的音频扩展谓词
3. 将候选按 `group_size`（默认 3）分组，每组调用 DashScope VL（默认 `qwen-vl-max`），Prompt 要求只从当前组选谓词、输出 JSON triples。
4. 输出经 `canonical_predicate()` 规范化；segment 级 progress 支持 `--resume`。

**产物**：`relations_llm.json`，`source` 通常为 `semi_auto` 或类似 LLM 标记；storyboard 保存在 `relation_llm/storyboards/`。

### 3.3 relation_merge — 多源合并与耦合

**实现**：[`relations/ops.py`](../src/vidvrd_auto/relations/ops.py) → `merge_relations()`

**合并键**：`(subject_track_id, predicate, object_track_id, start_frame, end_frame)`

- 同源重复：取较高 `confidence`，sources 合并，confidence 略增 (+0.05)。
- **`apply_coupling=true`**（默认）：若存在 `A -left-> B`，且不存在 `B -right-> A`，则自动补全反向关系，`source=coupling`。

反向表来自 `taxonomy.coupling_inverse()`，与 `ops.py` 内硬编码的 left/right 等合并。

### 3.4 global_relation — 跨窗口聚合

**实现**：[`nodes/global_relation.py`](../src/vidvrd_auto/nodes/global_relation.py)

- 按 `(subject, predicate, object)` 分组。
- 对 **dynamic_predicates**（默认 `toward, away, follow, chase`）：若同一三元组出现在 ≥ `min_segments` 个 window，合并为一条长 span，`source=global_relation`。
- 其余关系原样 passthrough。
- 可选 `vl_enabled`：全片均匀抽帧 storyboard + VL 做视频级复核（默认关闭）。

### 3.5 relation_verify — 质检与强模型复核

**实现**：[`relations/ops.py`](../src/vidvrd_auto/relations/ops.py) → `verify_relations()`

**检查项**：

| 检查 | 说明 |
|------|------|
| 低置信度 | `confidence < low_confidence_threshold`（默认 0.45） |
| 互斥冲突 | 同一 pair+span 上同时出现 mutex 组内谓词（如 left + right） |
| 强模型复核 | `strong_model_review_enabled=true` 时对低置信/冲突项调用 VL，产出 `final_actions` |

**产物**：

- `relations_verified.json`：关系列表（当前主要为 passthrough + QC 元数据）
- `relation_qc.json`：统计、冲突、复核结果；`passed = (conflict_count == 0)`

### 3.6 export — 最终导出

将 verify 后结果写入 `export/relations_pred.json`，并汇总至多视频 `pred/relations_pred.json`。字段规范见 [`SCHEMAS.md`](SCHEMAS.md)。

---

## 4. 关系记录 Schema

单条关系（导出统一格式）：

```json
{
  "subject_track_id": 1,
  "object_track_id": 2,
  "predicate": "left",
  "start_frame": 0,
  "end_frame": 30,
  "confidence": 0.82,
  "source": "rule_geometry",
  "segment_id": 1,
  "evidence": "geometry vote 18/24 in window 1"
}
```

| 字段 | 说明 |
|------|------|
| `subject_track_id` / `object_track_id` | OC-SORT 轨迹 ID；方向有语义（subject 相对 object 成立 predicate） |
| `predicate` | 英文 canonical，来自 taxonomy |
| `start_frame` / `end_frame` | 闭区间帧号 |
| `confidence` | 0~1；规则层为投票比例，LLM 为模型自报 |
| `source` | `rule_geometry` / `semi_auto` / `coupling` / `global_relation` 等 |
| `sources` | merge 后多源列表（可选） |
| `segment_id` | 来源 window（可选） |
| `evidence` | 人类可读依据（可选） |

---

## 5. 配置项

[`configs/default.json`](../configs/default.json) 中关系相关段落：

### relation_rule

| 键 | 默认 | 含义 |
|----|------|------|
| `enabled` | true | 是否运行规则节点 |
| `min_vote_ratio` | 0.6 | window 内谓词最低投票比例 |
| `axis_margin_ratio` | 0.08 | 左右/上下判定 margin（相对 bbox 尺度） |
| `near_distance_ratio` | 0.35 | near 判定距离阈值 |
| `overlap_iou_threshold` | 0.05 | overlap/contact IoU 阈值 |
| `max_pairs_per_window` | 0 | 每 window 最多轨迹对数（0=不限制） |
| `motion_align_ratio` | 0.35 | 共同运动速度对齐余弦阈值 |
| `motion_distance_eps_ratio` | 0.02 | toward/away 相对距离变化阈值 |

### relations（relation_llm）

| 键 | 默认 | 含义 |
|----|------|------|
| `api_model` | qwen-vl-max | DashScope VL 模型 |
| `relations` | `""` | 逗号分隔谓词列表；空则用 DEFAULT 10 个 |
| `group_size` | 3 | 每次 VL 询问的谓词数 |
| `max_windows` | 0 | 最多处理 window 数（0=全部） |
| `max_frames_per_window` | 8 | storyboard 最多帧数 |
| `vggsound_label` | `""` | 音频先验标签，可扩展 LLM 候选谓词 |
| `dry_run` | false | 只生成 storyboard，不调 API |

### relation_merge / global_relation / relation_verify

见 `default.json` 中 `apply_coupling`、`dynamic_predicates`、`low_confidence_threshold`、`strong_model_review_enabled` 等；完整说明见 [`configs/CONFIGS.md`](../configs/CONFIGS.md)。

### export.relation_viz_*（可视化，非检测）

渲染关系叠加视频时的过滤参数（`min_confidence`、`max_confidence_spatial` 等），见 [`utils/relation_viz.py`](../src/vidvrd_auto/utils/relation_viz.py)。

---

## 6. 代码模块索引

```text
src/vidvrd_auto/relations/
├── taxonomy.py          # 谓词 taxonomy 加载
├── predicate_aliases.py # 别名、VL 默认列表、音频扩展
├── ops.py               # 规则生成、merge、verify 核心逻辑
├── clip_relation.py     # relation_llm 主实现
├── merge.py             # merge 薄封装
└── verify.py            # verify 薄封装

src/vidvrd_auto/nodes/
├── relation_rule.py
├── relation_llm.py
├── global_relation.py
└── ...

scripts/render_relation_video.py   # 关系可视化渲染 CLI
tools/evaluate_presence.py         # Presence P/R/F1 评测（含谓词别名）
```

---

## 7. 与数据集标注的对齐

VIDVRD 原始样本标注（如 `anno_mm_vidvrd/`、`predicate2id.json`）使用**数据集侧 schema**，不会自动并入本仓库 `predicate_taxonomy.json`。

对接 Gold（`gold/relations_gold.json`）与 Presence 评测时需：

1. 将数据集 predicate id / 中文标签映射到 canonical 英文谓词。
2. 统一使用 `subject_track_id` / `object_track_id`（或评测脚本支持的 legacy 字段）。
3. 帧 span 与 track id 需与 pipeline 产出同一坐标系（同一检测/追踪 run）。

评测入口：`tools/evaluate_presence.py`，配置 `evaluate.gold_json`。

---

## 8. 可视化

[`relation_viz.py`](../src/vidvrd_auto/utils/relation_viz.py) 在轨迹框之间绘制关系箭头与 `subject->object:predicate conf` 标签。

过滤规则（可配置）：

- 置信度低于阈值的关系不画
- 位置类谓词：过高置信或中心距过大时可隐藏（减少 clutter）

生成命令：

```bash
python scripts/render_relation_video.py --run_dir runs/<run_id>
```

---

## 9. 已知局限

1. **词表三处不同步**：taxonomy（16）、VL 默认（10）、规则产出（12+）未完全统一；改谓词需多处手工维护。
2. **深度谓词**：`front` / `behind` 仅 LLM 可产出，规则层无深度估计。
3. **交互/音频谓词**：`sing_with` / `sing_to` 依赖 LLM + 音频先验扩展，规则层不覆盖。
4. **taxonomy → Prompt**：`prompt_predicate_summary()` 尚未接入 VL 主 Prompt，LLM 仅看到英文谓词名列表。
5. **verify 阶段**：当前以 QC 报告为主，强模型 `final_actions` 默认不自动改写 relations 列表。
6. **数据集 Gold**：仓库内仍为 `validation_dummy` 样例；批量 50 条样本需转换脚本入仓。

---

## 10. 扩展谓词操作清单

| 步骤 | 操作 |
|------|------|
| 1 | 在 `predicate_taxonomy.json` 增加谓词及 category / inverse / mutex_group |
| 2 | 在 `predicate_aliases.py` 补充中英文别名 |
| 3 | 若需 VL 识别：配置 `relations.relations` 或更新 `DEFAULT_RELATION_PREDICATES` |
| 4 | 若需规则产出：在 `ops.generate_rule_relations()` 增加几何/时序判定 |
| 5 | 若参与动态聚合：更新 `global_relation.dynamic_predicates` |
| 6 | 若可视化需特殊处理：更新 `relation_viz.POSITIONAL_PREDICATES` 等 |
| 7 | 跑 `--from_node relation_rule` 或全链 smoke，检查 merge 耦合与 verify 互斥 |

---

## 11. 后续优化方向（规划，非当前任务）

以下为团队已识别、**优先级较高但暂不实施**的改进项，供后续迭代参考：

1. **统一谓词来源**：从 `predicate_taxonomy.json` 自动生成 VL 候选列表、`DEFAULT_RELATION_PREDICATES` 与可视化位置类集合，消除三处手工不一致。
2. **关系检测质量**：
   - 规则层引入更稳健的深度/尺度归一化；减少 near/overlap 与 left/right 冗余。
   - LLM Prompt 接入 taxonomy 中文释义与互斥说明，降低 hallucination。
   - merge 阶段按 source 加权而非简单 max confidence。
3. **global_relation**：扩大聚合谓词范围；默认开启轻量 VL 复核或基于时序连贯性的 span 合并。
4. **verify 闭环**：强模型 `final_actions` 自动 apply 到 `relations_verified.json`。
5. **评测闭环**：`anno_mm_vidvrd` → `gold/relations_gold.json` 转换器 + 批量 Presence 报告。
6. **性能**：relation_llm 的 window/帧数上限、storyboard 缓存策略；长视频与 Rex 检测的 interval 协同调参。

---

## 12. 调试与重跑

关系节点依赖上游 `step2_track` 产物。局部重跑示例：

```bash
python -m vidvrd_auto.cli \
  --videos data/videos.txt \
  --run_dir runs/exp001 \
  --config configs/default.json \
  --from_node relation_rule \
  --to_node export \
  --force
```

注意：正在运行的进程不会加载新代码；修改实现后需新 `run_dir` 或 `--force` 重跑相关节点。

查看单视频 QC：

```text
runs/<run_id>/videos/<video_id>/relation_verify/relation_qc.json
```
