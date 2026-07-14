# VIDVRD 变更记录

> 记录 2026-05-29 前后围绕「项目审查、语义关系改进、废弃清理、代码重整」的改动与结论。

---

## 1. 背景与目标

项目面向大创「Agent 赋能的视频关系理解」，主链为 OpenClaw-first 的全自动视频关系标注流水线。

本轮工作的核心诉求：

1. **全面审查**项目完成度、正确性、逻辑完整性、性能。
2. **明确能力边界**：当前系统擅长几何/窗口级关系，不能稳定识别「对唱、滑滑板、追人」等高层语义关系。
3. **按可执行计划改进**：扩展检测类别、谓词体系、object-aware 候选、pair storyboard、verify 真正改输出。
4. **删除废弃代码、统一命名、减少重复层**。

---

## 2. 项目审查结论（审查阶段）

### 2.1 已完成的部分

| 模块 | 状态 |
|------|------|
| 12 节点主链（ingest → export） | 可跑通 |
| CLI + `--resume` / `--from_node` / `--to_node` | 可用 |
| 配置合并（`configs/default.json` + 覆盖配置） | 可用 |
| Presence 评测（`tools/evaluate_presence.py`） | 可用 |
| OpenClaw Skill（`skills/vidvrd-full-auto/`） | 有文档 |

### 2.2 主要问题（审查时发现）

| 类别 | 问题 |
|------|------|
| 检测 | 默认 `rex_categories: "person"`，skateboard/microphone 等未检出则无语义关系 |
| 关系 | 谓词仅约 13 个，偏 left/right/near/overlap |
| VL 输入 | 全景 storyboard 分辨率低，小物体看不清 |
| Prompt | 按谓词分组盲问，模型不知物体类别 |
| Verify | 检测冲突/低置信度后**不执行**修正，输出原样复制 |
| 架构 | step1/step2/relation_llm 仍 subprocess 调 `my_scripts/` |
| 数据 | `semi_auto` resume 不加载已有 `output_json`，中断恢复丢关系 |
| 对齐 | `video_id` 可能用 `video_path.stem`，与 pipeline 不一致导致 merge 丢关系 |
| 性能 | 全量读 JSONL、多节点重复解析、串行 VL、默认 export pair 视频 |

### 2.3 能力边界（共识）

当前主链本质是：

```text
检测框（多类别后可扩展）→ 轨迹（OC-SORT）→ 几何规则 + VL 看 storyboard 猜关系 → 合并/复核/导出
```

**不能稳定保证**：对唱、滑滑板、拥抱、传球等需「物体类别 + 动作语义 + 时序 + 可选音频」联合理解的关系；无专门训练的视频关系模型。

改进方向（不做视频片段 API 输入）：扩检测类别、Gold 小集、pair 裁剪 storyboard、object-aware 候选、verify 生效。

详细计划见：`plan/semantic_relation_plan.md`。

---

## 3. 语义关系改进（功能改动）

### 3.1 配置

| 文件 | 改动 |
|------|------|
| `configs/default.json` | `rex_categories` 扩展为 16 类：`person,skateboard,bicycle,horse,dog,cat,ball,guitar,microphone,chair,table,car,bag,cup,phone,surfboard` |
| `configs/semantic_relations.json` | **新增** 语义关系实验配置：`object_aware_candidates`、`pair_storyboard`、`apply_actions`、`category_constraints_enabled`、`min_export_confidence` 等 |

### 3.2 谓词体系

| 文件 | 改动 |
|------|------|
| `configs/predicate_taxonomy.json` | 由约 13 个扩至约 **25 个**，新增 `subject_categories` / `object_categories` |

新增层级示例：

- **接触/承载**：`ride`, `sit_on`, `hold`, `carry`, `wear`, `on`, `under`
- **交互**：`hug`, `kick`, `push`, `talk_to`, `look_at`, `walk_with`, `play_with`
- **音频辅助**：`sing_with`
- 保留原空间/运动：`left`, `right`, `follow`, `chase`, `toward`, `away` 等

### 3.3 新增：Object-Aware 候选

| 文件 | 说明 |
|------|------|
| `src/vidvrd_auto/relations/object_candidates.py` | **新增**。类别对 → 候选谓词表，如 `(person, skateboard) → [ride, on, hold, ...]`；支持音频触发 `sing_with` / `talk_to` |

`relation_rule` 在 `object_aware_candidates: true` 时生成低置信度候选（`source: candidate_object_aware`），供 VL 验证。

### 3.4 关系核心逻辑 `ops.py`

| 能力 | 说明 |
|------|------|
| `generate_rule_relations` | 几何规则 + object-aware 候选；`audio_label` 由 runner 注入 |
| `merge_relations` | 去重、耦合补全（left↔right 等） |
| `verify_relations` | **执行** `delete` / `change_predicate` / `adjust_span`；互斥自动消解；类别约束过滤；`min_export_confidence` 过滤 |

`runner.py` 将 `audio_prior` 标签传入 `relation_rule` 配置。

**规则节点参数修正**：`max_pairs_per_window` 仅限制 object-aware **候选条数**；若需限制窗口内参与几何/候选的轨迹数量，使用 `max_track_ids_per_window`（避免旧逻辑误把「截断 track_ids」当成「限制 pair 数」）。

### 3.5 片段关系 `semi_auto_label_relations.py`

| 能力 | 说明 |
|------|------|
| `--video_id` | 与主链 video_id 对齐 |
| `--pair_storyboard` | 按 track 对裁剪放大 storyboard，标注 A/B + 类别 + 时间戳 |
| Pair prompt | 给定候选谓词列表，VL 验证是否成立（非全谓词分组盲问） |
| `--resume` | 启动时加载已有 `output_json`，避免丢已完成窗口的关系 |
| Resume 边界 | 若仅有 `.progress.json` 而 `output_json` 缺失，会告警并**重跑 LLM 段**（避免空输出却整段 SKIP） |
| 谓词别名 | `PREDICATE_ALIASES` 扩展中文/英文 → canonical（含 ride、sing_with、talk_to 等） |

`clip_classifier.py` 增加 `--video_id`、`--pair_storyboard`、`--max_pairs_per_window` 等参数传递。

### 3.6 VL 与全局/复核

| 文件 | 改动 |
|------|------|
| `prompts/templates.py` | verify prompt 说明可使用 storyboard 图片、`index` 下标 |
| `nodes/global_relation.py` | VL 调用可传 `image_paths`；`_global_review` 写入独立 `global_review.json`，不混入关系 JSON |
| `relations/verify` → `ops.verify_relations` | 支持 `storyboards_dir`，强模型复核可看图 |

### 3.7 评测

| 文件 | 改动 |
|------|------|
| `tools/evaluate_presence.py` | 新增 **Per Predicate** 表（TP/FP/FN/P/R/F1）；Gold 可读 `predicate` 或旧字段 `relationship_type`（兼容 `data/manual_samples/add_VidVRD/anno_*.json`） |

---

## 4. 删除的废弃文件

以下文件已无活跃引用或已被新主链替代，**可安全删除**（部分已在工作区删除）。

### 4.1 `auto_label/`（整目录）

| 文件 | 原因 |
|------|------|
| `auto_label/vidvrd_auto_label.py` | 仅转发 `vidvrd_auto.cli`，与 `scripts/run_vidvrd_auto.py` 重复 |
| `auto_label/relation_ops.py` | 与 `src/vidvrd_auto/relations/ops.py` 重复，零 import |
| `auto_label/default_config.json` | 已被 `configs/default.json` 取代 |
| `auto_label/__init__.py` | 无实际导出 |
| `auto_label/README.md` | 内容过时 |

### 4.2 `my_scripts/` 旧主链

| 文件 | 原因 |
|------|------|
| `step3_window_relation_classification.py` | 由 `semi_auto_label_relations.py` + `relation_llm` 取代 |
| `step4_video_relation_event_aggregation.py` | 由 `nodes/global_relation.py` 取代 |
| `step5_video_relation_natural_language_qwen.py` | 新 pipeline 不做 NL 描述；仅旧 `run_all` 使用 |
| `run_all.py` | 旧 Step1–5 编排，已被 `vidvrd_auto.cli` 取代 |
| `run_phase1.py` | 旧 Phase1 快捷入口 |
| `modules/visualizer.py` | 全仓库零 import |
| `prepare_anno_platform_inputs.py` | 仅 README 提及 |
| `requirements.txt` | 无引用；依赖见 `pyproject.toml` / 文档 |
| `modules/【需求文档】....pdf` | 非代码资产 |

### 4.3 仍须保留的 `my_scripts/`（subprocess 依赖）

| 文件 | 调用方 |
|------|--------|
| `step1_full_video_box_detection_dinox.py` | `detection/legacy_step1.py` |
| `step2_full_video_tracking_ocsort_qc_pairviz.py` | `tracking/legacy_step2.py` |
| `semi_auto_label_relations.py` | `relations/clip_classifier.py` |
| `config.py`, `utils_io.py`, `modules/*` | 上述脚本依赖 |
| `install_rexomni_deps.py` | step1 `--auto_install_rexomni` 时 |

---

## 5. 代码重整（结构清理）

### 5.1 消除重复 IO

- `relations/ops.py` 删除自实现的 `read_json` / `write_json` / `_iter_jsonl`
- 统一使用 `vidvrd_auto.utils.io`

### 5.2 删除薄包装模块

| 删除文件 | 替代方式 |
|----------|----------|
| `relations/rules.py` | `runner` 直接从 `relations.ops` 导入 `generate_rule_relations` |
| `relations/merge.py` | 同上，`merge_relations` |
| `relations/verify.py` | 同上，`verify_relations`（`storyboards_dir` 已在 `ops` 中） |

`relations/__init__.py` 作为公共 API re-export 上述三函数。

**对外导入约定**（删除薄包装后）：

```python
from vidvrd_auto.relations import generate_rule_relations, merge_relations, verify_relations
# 或
from vidvrd_auto.relations.ops import ...
```

`tests/test_relations.py` 已改为从 `vidvrd_auto.relations` 导入（勿再 `from vidvrd_auto.relations.merge import ...`）。

### 5.3 删除未使用代码

| 文件 | 原因 |
|------|------|
| `models/detector.py` | `DetectorConfig` / `load_detector_config` 全仓库无引用 |

### 5.4 文档整理

| 操作 | 文件 |
|------|------|
| 删除 | `FULL_FLOW_INTERFACE.md`（与 ARCHITECTURE + SCHEMAS 重复） |
| 删除 | `docs/CLEANUP_AUDIT.md`（清理任务已完成） |
| 删除 | 根目录 `plan.md`（与 `plan/plan.md` 重复） |
| 重写 | `README.md` — 当前结构、25 谓词、semantic 配置、验证命令 |
| 重写 | `docs/ARCHITECTURE.md` — 包结构、关系流、适配层说明 |
| 重写 | `my_scripts/README.md` — 仅描述 3 个仍被 subprocess 调用的 legacy 脚本 + 调试命令；删除对已移除的 `run_all` / Step3–5 的引用 |

### 5.5 整理后的 `relations/` 结构

```text
src/vidvrd_auto/relations/
├── __init__.py           # 公共 API
├── ops.py                # 规则 / 合并 / 复核（唯一核心实现）
├── clip_classifier.py    # subprocess → semi_auto
├── object_candidates.py  # 类别对候选
└── taxonomy.py           # 读取 predicate_taxonomy.json
```

---

## 6. 当前推荐运行方式

```bash
conda activate vidvrd
pip install -e .

# 语义关系实验
python -m vidvrd_auto.cli \
  --videos data/videos_semantic.txt \
  --run_dir runs/semantic_v1 \
  --config configs/semantic_relations.json \
  --resume \
  --api_key YOUR_DASHSCOPE_API_KEY

# 评测（需准备 gold/relations_gold_semantic.json）
python tools/evaluate_presence.py \
  --gold gold/relations_gold_semantic.json \
  --pred runs/semantic_v1/pred/relations_pred.json \
  --report runs/semantic_v1/reports/presence_report.md
```

---

## 7. 尚未完成 / 下一步

| 项 | 说明 |
|----|------|
| Gold 小集 | `gold/relations_gold_semantic.json`，≥30 条，覆盖 8+ 非空间谓词 |
| 视频列表 | `data/videos_semantic.txt`，含骑车/追逐/唱歌等场景 |
| 端到端实测 | 用 `semantic_relations.json` 跑通并看 per-predicate F1 |
| 代码迁移 | step1/step2/semi_auto 迁入 `src/vidvrd_auto`（去掉 subprocess） |
| 运动规则 | 用 tracker 速度/距离序列补 toward/away/follow（未做） |
| `pyproject.toml` | 可补充 runtime 依赖列表（原 `my_scripts/requirements.txt` 已删） |

验收目标（计划）：Gold 小集上 Overall F1 ≥ 0.5，且能按谓词汇报失败原因。

### 7.1 架构是否继续重构（审查结论）

| 判断 | 说明 |
|------|------|
| **当前是否更清晰** | 是。主入口统一到 `vidvrd_auto.cli`，`relations/` 单实现（`ops.py`），文档与删除废弃脚本方向一致。 |
| **是否建议继续大重构** | **否**。先跑通 `semantic_relations.json` 端到端 + Gold 评测，再考虑迁移 subprocess。 |
| **可选小步** | ① `ops.py` 拆文件（rules/merge/verify 逻辑分模块，仍经 `ops` 或 `__init__` 导出）；② 将 `semi_auto` 核心迁入 `src`；③ `pyproject.toml` 补全依赖。 |
| **暂勿动** | `my_scripts/semi_auto_label_relations.py` 体量大但仍是唯一 relation_llm 算子，大挪移易引入回归。 |

### 7.2 已知限制（改完仍存在）

| 项 | 说明 |
|----|------|
| 质检不阻断 | `track_qc` / `keyframe_screen` 只记风险或 skip，不因短轨/大跳变自动停链 |
| VL 质检默认关 | `keyframe_screen.vl_enabled`、`global_relation.vl_enabled`、`relation_verify.strong_model_review_enabled` 在 default 中多为 false；开启后多数节点才传 storyboard |
| 检测成本 | `default.json` 已扩 16 类 + Rex `detection_interval: 1` 时，长视频 Step1 仍是最重环节 |
| Presence 与 track_id | 评测按 `(subject_track_id, predicate, object_track_id)`；Gold 若与某次自动追踪的 id 不一致，需固定「同一次 run 的 tracks」再标 Gold |
| 语义能力 | 链路改进后仍依赖 VL 零样本 + 规则候选，不保证 sing_with/ride 稳定正确 |

---

## 8. 本地验证（重整后）

在 `PYTHONPATH=src` 下已执行：

```bash
python -m compileall -q src scripts tests tools my_scripts
python -m unittest discover -s tests
python -m vidvrd_auto.cli --help
```

结果：**6/6 测试通过**，CLI 可加载；**无当前 linter 报错**。

重整过程中曾暴露的问题（已修）：

- 删除 `relations/merge.py` 后，`tests/test_relations.py` 仍 `from vidvrd_auto.relations.merge import merge_relations` → 已改为 `from vidvrd_auto.relations import merge_relations`。

**未在本轮执行**：真实视频 + API 的全链跑通、Gold 对比实验（依赖 `videos_semantic.txt` 与 `gold/relations_gold_semantic.json`）。

---

## 9. 变更文件清单（git 视角）

### 新增

- `configs/semantic_relations.json`
- `plan/semantic_relation_plan.md`
- `src/vidvrd_auto/relations/object_candidates.py`
- `change.md`（本文件）

### 修改

- `configs/default.json`
- `configs/predicate_taxonomy.json`
- `my_scripts/semi_auto_label_relations.py`
- `src/vidvrd_auto/relations/ops.py`
- `src/vidvrd_auto/relations/__init__.py`
- `src/vidvrd_auto/relations/clip_classifier.py`
- `src/vidvrd_auto/pipeline/runner.py`
- `src/vidvrd_auto/nodes/global_relation.py`
- `src/vidvrd_auto/prompts/templates.py`
- `tools/evaluate_presence.py`
- `README.md`
- `docs/ARCHITECTURE.md`
- `my_scripts/README.md`
- `tests/test_relations.py`

### 删除

- `auto_label/`（全部）
- `my_scripts/step3*.py`, `step4*.py`, `step5*.py`, `run_all.py`, `run_phase1.py`
- `my_scripts/visualizer.py`, `prepare_anno_platform_inputs.py`, `requirements.txt`
- `src/vidvrd_auto/relations/rules.py`, `merge.py`, `verify.py`
- `src/vidvrd_auto/models/detector.py`
- `FULL_FLOW_INTERFACE.md`, `docs/CLEANUP_AUDIT.md`, `plan.md`（根目录）

---

## 10. 一句话总结

**完成了项目审查与语义关系链路改造（多类别检测、25 谓词、object-aware 候选、pair storyboard、verify 真过滤），并删除废弃脚本与重复包装层、统一 relations 实现与文档；结构已够清晰，宜先端到端实验而非继续大重构；要验证「对唱/滑滑板」类效果，仍需 Gold 与真实跑数。**
