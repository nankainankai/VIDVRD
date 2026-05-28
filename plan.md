# 全自动视频关系标注项目分工计划

## 当前项目状态（截至 2026-05-28）

项目已形成 **OpenClaw-first** 全自动标注主链 MVP：统一 CLI（`vidvrd_auto.cli`）、12 节点编排、配置合并、节点缓存、`run_manifest.json`、自动生成 `reports/run_report.md`、unittest 与多种运行配置。

### 主链（已实现）

```text
video_ingest
  -> audio_prior
  -> step1_detect
  -> keyframe_screen
  -> step2_track
  -> track_qc
  -> relation_rule
  -> relation_llm
  -> relation_merge
  -> global_relation
  -> relation_verify
  -> export
```

### 已验收能力

| 能力 | 说明 |
|------|------|
| Mock 端到端 | `configs/dry_run.json`：`detector/tracking.backend=mock`，无需 GPU/API |
| VL 传图 | `utils/vl_frames.py` + `VLClient.call_bgr`；`keyframe_screen` / `track_qc` / `global_relation` / `relation_verify` |
| 片段关系主包化 | `relations/clip_relation.py`（`relation_llm` 节点）；`semi_auto_label_relations.py` 为薄 CLI |
| Rex-Omni-AWQ | 本地 `D:/Rex-Omni-AWQ`（仓库外）+ `rexomni_detector.py`；`configs/rexomni_full.json` 在 `test1_video` 上 **12/12 节点 succeeded** |
| 规则 motion | `relations/ops.py`：toward/away/follow 等 |
| 关系可视化 | `export.relation_viz_video` → `relation_box_vis.mp4`（`utils/relation_viz.py`） |
| Presence 评测链路 | `tools/evaluate_presence.py` + `gold/relations_gold.json` **样例 1 视频** |

### 未完成 / 进行中

- **50 条真 Gold**（见下文）：标注组交付前无法做有意义的批量 P/R/F1。
- Step1/Step2 仍经 `detection/legacy_step1.py`、`tracking/legacy_step2.py` 调 `my_scripts`，未完全迁入 `src/vidvrd_auto`。
- `keyframe_screen` 在 Rex 生产配置中常关闭；VL 粗筛与 Rex 链路的组合策略待产品化。
- `relation_verify.strong_model_review_enabled` 默认 false；强模型终审未常态化。
- Prompt / 谓词边界需结合 Gold 与失败 case 迭代。
- `default.json` 中 `relations.max_windows: 0` 会对所有窗口调 LLM，长视频极慢——生产跑批需显式限制 `max_windows`。

### 工程里程碑日志

- **2026-05-26**：mock dry-run；`check_openclaw_env.py`；`pipeline/report.py`；`clip_relation` 迁入；VL 节点传图；`configs/CONFIGS.md`；motion 规则扩展。
- **2026-05-27**：`configs/rexomni_full.json`；Rex 全链验收（`runs/rexomni_full_test1`）；`run_with_vl.json` / 一键脚本。
- **2026-05-28**：`rexomni_full_kf2.json`（`keyframe_interval: 2`）；`test1_kf2` 全链 + 关系可视化。

---

## 「50 条 Gold」是什么？

定义来源：**[`plan/plan.md`](plan/plan.md) 第一阶段**（人工标注 + 半自动 + 可复现评测）。

| 项 | 说明 |
|----|------|
| **是什么** | 约 **50 个视频** 的人工标准标注，与自动产出 **同一 schema**，用于对比质量 |
| **轨迹 Gold** | `gold/trajectories_gold.json`：关键帧标注 → 插值/传播 → 少量修正，得到逐帧框与 `track_id` |
| **关系 Gold** | `gold/relations_gold.json`：按 **时间段** 标注关系（`start_frame`~`end_frame`），不是逐帧打点 |
| **工程用途** | 自动跑批得到 `pred/relations_pred.json` 后，跑 **Presence P/R/F1**（`tools/evaluate_presence.py`），输出 `reports/presence_report.md`，按失败类型改规则/Prompt |
| **仓库现状** | 仅 `validation_dummy` **2 条关系** smoke 样例，用于验证评测脚本；**不能**当作项目真实精度 |

标注组交付后：替换 `gold/` 内容，配置 `evaluate.enabled: true`（或 `configs/production.json`），对同一批 `video_id` 跑主链并对比。

---

## 分工原则

- 新代码默认进入 `src/vidvrd_auto/`。
- 配置默认进入 `configs/`。
- 说明与运行文档使用中文。
- 真实测试与 Rex 推理优先在 **`vidvrd` conda** 环境。
- `my_scripts/` 为过渡层，检测/追踪迁移完成后仅保留薄包装。

---

## 成员 1：工程主链与实验闭环（吴、李）

**方向**：系统可跑、可恢复、可比较、可汇报。

**任务**：

- 维护 `cli`、`pipeline/runner.py`、`run_manifest.json`、节点缓存与 `--resume`。
- OpenClaw Skill：环境检查、运行、失败诊断、resume、结果汇报。
- 维护配置模板与 [`configs/CONFIGS.md`](configs/CONFIGS.md)；批量跑批与成本/失败统计。
- 维护 `README.md`、`docs/WORKFLOW_AGENT.md`。

**模块**：`src/vidvrd_auto/cli.py`、`pipeline/`、`config/`、`skills/vidvrd-full-auto/`、`docs/`、`tests/`。

---

## 成员 2：检测、关键帧筛选与轨迹质量（张）

**方向**：会议方案第 2–4 步——稳定框与轨迹。

**任务**：

- ~~下载部署 Rex-Omni-AWQ~~ → **已完成本地路径与主链接入**；维护 `rexomni_detector.py`、Windows/CUDA 问题。
- 可选 DINO-X（`DINOX_API_TOKEN`、`production_full.json`）作备选检测。
- 关键帧策略：`keyframe_interval`、框可视化；对比 `rexomni_full` vs `rexomni_full_kf2`。
- `keyframe_screen`：VL `keep/drop/crop`（Rex 链路上是否启用待决）。
- OC-SORT、窗口切分、pair 可视化；`track_qc` 规则 + VL。
- **迁移** Step1/Step2 至 `src/vidvrd_auto/detection`、`tracking`。

**模块**：`detection/`、`tracking/`、`nodes/detect.py`、`screen.py`、`track.py`、`track_qc.py`；`my_scripts/modules/rexomni_detector.py`。

---

## 成员 3：谓词体系、规则关系与片段关系（李、吴）

**方向**：会议方案第 5–6 步——片段可靠关系候选。

**任务**：

- `configs/predicate_taxonomy.json`：层级、互斥、反向耦合、接触/运动约束。
- 规则关系与 motion；storyboard、音频先验、分组 VL Prompt。
- ~~迁入 `clip_classifier`~~ → **核心已在 `clip_relation.py`**；持续调解析与异常处理。
- 结合 **50 条 Gold** 与 Presence 失败 case 迭代。

**模块**：`relations/taxonomy.py`、`ops.py`、`clip_relation.py`、`storyboard.py`、`nodes/relation_llm.py`、`prompts/`。

---

## 成员 4：全局关系、复核与评测交付（黄）

**方向**：会议方案第 7–8 步 + 与 Gold 对齐。

**任务**：

- `global_relation` 跨窗聚合；动态谓词 toward/follow 等。
- `relation_verify`：冲突、低置信、强模型复核（当前默认关闭，待开启评估）。
- 导出 schema；**主导 50 条 Gold 规范与 Presence 报告**、失败案例分析。
- `tools/evaluate_presence.py`、`gold/`、启用 `evaluate` 的跑批流程。

**模块**：`nodes/global_relation.py`、`relations/verify.py`、`nodes/export.py`、`evaluation/`、`tools/`、`gold/`。

---

## 协作节奏

1. **第一轮（已完成）**：dry-run / mock 主链不断。
2. **第二轮（进行中）**：Rex + VL 真模型单视频验收；录失败 case。
3. **第三轮（待 Gold）**：50 视频 Gold 齐套 → 每日/每版跑评测 → 指标表与汇报材料。

合并前检查：

```bash
conda run -n vidvrd python -m vidvrd_auto.cli --help
conda run -n vidvrd python -m compileall -q src scripts tests
conda run -n vidvrd python -m unittest discover -s tests
```

---

## 最终验收目标

| 目标 | 状态 |
|------|------|
| 给定视频一键产出轨迹、关系、报告 | 单视频已达成（mock / Rex+VL） |
| 节点失败可 `--resume` | 已达成 |
| 大模型环节有真实现 + dry-run/mock | 已达成 |
| 旧 `my_scripts` 不被主链依赖 | **未达成**（Step1/2 仍 legacy） |
| 新成员读 README 即可运行 | 已更新 README（含 Rex/Gold 说明） |
| 50 条 Gold 对齐 Presence 评测 | **待标注组交付** |

---

## 相关文档

- 大创阶段目标与 50 条交付清单：[`plan/plan.md`](plan/plan.md)
- 会议记录：[`plan/大创会议.md`](plan/大创会议.md)
- Gold 字段说明：[`gold/README.md`](gold/README.md)
