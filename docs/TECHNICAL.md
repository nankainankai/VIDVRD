# VIDVRD Auto 技术总览

本文档是 **VIDVRD 全自动视频关系标注** 项目的总技术说明，面向开发者与维护者：说明系统做什么、怎么跑、产物是什么、各模块如何衔接。

> **文档分层**
>
> | 文档 | 定位 |
> |------|------|
> | **本文 `TECHNICAL.md`** | 总览：目标、主链、数据流、配置、产物、能力边界 |
> | [`ARCHITECTURE.md`](ARCHITECTURE.md) | 架构专篇：包结构、可恢复性、验证命令 |
> | [`RELATIONS.md`](RELATIONS.md) | 关系检测专篇：谓词体系、规则/VL/merge/verify 算法 |
> | [`SCHEMAS.md`](SCHEMAS.md) | 产物 JSON 字段规范 |
> | [`WORKFLOW_AGENT.md`](WORKFLOW_AGENT.md) | OpenClaw/Agent 调度与故障诊断 |
> | [`../configs/CONFIGS.md`](../configs/CONFIGS.md) | 配置文件选择与合并规则 |
> | [`../README.md`](../README.md) | 快速上手、命令示例、环境安装 |
> | [`../plan.md`](../plan.md) | 团队分工与工程里程碑 |

---

## 1. 项目目标

**输入**：一个或多个视频（本地路径或 URL 列表）。

**输出**（结构化、可审计）：

| 产物 | 说明 |
|------|------|
| `trajectories_pred.json` | 逐帧物体轨迹：`track_id`、类别、bbox |
| `relations_pred.json` | 时序关系三元组：主体/客体 track、谓词、起止帧、置信度、来源 |
| `relation_qc.json` | 关系质检：低置信、互斥冲突、强模型复核建议 |
| `run_report.md` | 本次 run 汇总报告 |
| `presence_report.md` | 相对 Gold 的 Presence P/R/F1（可选） |

**设计原则**：

- **OpenClaw-first**：Agent 只负责调用 CLI、读 manifest、失败恢复；业务逻辑在 `src/vidvrd_auto/`。
- **可恢复**：每节点写 `status.json` + 输入 hash；`--resume` 跳过已成功步骤。
- **配置驱动**：检测后端、VL 开关、关系参数均通过 JSON 配置，不硬编码环境差异。

大创两阶段目标（人工 Gold + 全自动 pred + Presence 迭代）见 [`plan/plan.md`](../plan/plan.md)。

---

## 2. 系统总览

```text
                    ┌─────────────────────────────────────────┐
                    │           vidvrd_auto.cli               │
                    │     pipeline/runner + run_manifest      │
                    └─────────────────┬───────────────────────┘
                                      │
     ┌────────────────────────────────┼────────────────────────────────┐
     │                                │                                │
     ▼                                ▼                                ▼
┌─────────┐                    ┌─────────────┐                  ┌─────────────┐
│ 感知层   │                    │  关系层      │                  │  交付层      │
│ ingest  │                    │ rule + LLM  │                  │ export      │
│ detect  │ ── windows/tracks ─▶│ merge       │ ── relations ──▶│ pred 汇总   │
│ track   │                    │ global      │                  │ QC / 报告   │
│ track_qc│                    │ verify      │                  │ evaluate    │
└─────────┘                    └─────────────┘                  └─────────────┘
     │                                │
     ▼                                ▼
 Rex-Omni / DINO-X / mock      规则几何 + DashScope VL
 OC-SORT / mock                 谓词 taxonomy + 耦合/互斥
```

**12 节点主链**（顺序固定）：

```text
video_ingest → audio_prior → step1_detect → keyframe_screen → step2_track
  → track_qc → relation_rule → relation_llm → relation_merge
  → global_relation → relation_verify → export
```

各节点职责一句话：

| 节点 | 职责 |
|------|------|
| `video_ingest` | 视频落盘、元信息 |
| `audio_prior` | VGGSound 音频标签先验（可选） |
| `step1_detect` | 关键帧目标检测（Rex-Omni / DINO-X / mock） |
| `keyframe_screen` | 关键帧粗筛（规则 + 可选 VL，可 drop 视频） |
| `step2_track` | OC-SORT 轨迹（legacy 适配） |
| `track_qc` | 轨迹质量检查（规则 + 可选 VL） |
| `relation_rule` | 几何/运动规则关系 |
| `relation_llm` | storyboard + 分组 VL 关系 |
| `relation_merge` | 多源合并 + inverse 耦合补全 |
| `global_relation` | 跨窗口动态关系聚合 |
| `relation_verify` | 冲突/低置信 QC + 可选强模型复核 |
| `export` | 导出 pred、可选关系可视化视频 |

节点细节与包结构见 [`ARCHITECTURE.md`](ARCHITECTURE.md)；关系算法见 [`RELATIONS.md`](RELATIONS.md)。

---

## 3. 端到端数据流

以单视频 `runs/<run_id>/videos/<video_id>/` 为粒度：

```text
inputs/source.json
    │
    ├─▶ step1_detect/detections_full.jsonl, video_meta.json
    ├─▶ step2_track/tracks_full.jsonl, windows.json
    │
    ├─▶ relation_rule/relations_rule.json
    ├─▶ relation_llm/relations_llm.json (+ storyboards/)
    │
    ├─▶ relation_merge/relations_merged.json
    ├─▶ global_relation/relations_global.json
    ├─▶ relation_verify/relations_verified.json, relation_qc.json
    │
    └─▶ export/relations_pred.json, trajectories_pred.json, relation_qc.json
              │
              └─▶ runs/<run_id>/pred/relations_pred.json（多视频汇总）
```

**关键中间产物**：

- **`windows.json`**：追踪窗口列表，关系节点按 window 分段处理。
- **`tracks_full.jsonl`**：逐帧轨迹，规则关系与 storyboard 绘制均依赖此文件。
- **`run_manifest.json`**：全 run 状态、配置 hash、每视频每节点 succeeded/failed。

字段定义见 [`SCHEMAS.md`](SCHEMAS.md)。

---

## 4. 仓库结构

```text
VIDVRD/
├── src/vidvrd_auto/       # 主工程包（新代码默认放这里）
│   ├── cli.py             # CLI 入口
│   ├── pipeline/          # runner、manifest、report
│   ├── nodes/             # 12 节点入口
│   ├── relations/         # 规则/VL/merge/verify/taxonomy
│   ├── detection/         # 检测迁移层（含 legacy 适配）
│   ├── tracking/          # 追踪迁移层
│   ├── models/            # VLClient 等
│   ├── evaluation/        # 评测钩子
│   └── utils/             # IO、relation_viz、vl_frames 等
├── configs/               # 运行配置 + predicate_taxonomy.json
├── gold/                  # 人工 Gold（评测标准答案）
├── runs/                  # 运行产物（通常不入库）
├── scripts/               # 薄 CLI、环境检查、一键脚本
├── tests/                 # unittest
├── tools/                 # evaluate_presence 等
├── my_scripts/            # Step1/Step2 过渡实现（Rex-Omni 等）
├── skills/vidvrd-full-auto/  # OpenClaw Skill
└── docs/                  # 技术文档（本文及专篇）
```

**迁移状态**：Step1/Step2 仍经 `detection/legacy_step1.py`、`tracking/legacy_step2.py` 调用 `my_scripts/`；关系链已主包化（`relations/clip_relation.py` 等）。

---

## 5. 环境与依赖

| 项 | 说明 |
|----|------|
| 推荐环境 | `vidvrd` conda（torch、opencv、Rex 依赖） |
| 安装 | `python -m pip install -e .` |
| 检查 | `python scripts/check_openclaw_env.py` |

**外部服务与密钥**（勿写入仓库）：

| 变量 / 资源 | 用途 |
|-------------|------|
| `DASHSCOPE_API_KEY` | DashScope VL：`relation_llm`、`track_qc`、`global_relation`、`relation_verify` 等 |
| `DINOX_API_TOKEN` | 仅 `detector.backend=dinox` 时 |
| `D:/Rex-Omni-AWQ/` | 本地 Rex-Omni 权重（仓库外，勿提交 Git） |

Mock 模式（`configs/dry_run.json`）无需 GPU 与 API，用于 CI 与冒烟。

---

## 6. 运行方式

### 6.1 主命令

```bash
python -m vidvrd_auto.cli \
  --videos data/videos.txt \
  --run_dir runs/exp001 \
  --config configs/default.json \
  --resume
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--video` / `--videos` | 单视频或列表 txt（每行一路径） |
| `--config` | JSON 配置（深度覆盖 `default.json`） |
| `--resume` | 跳过 input_hash 一致且已成功的节点 |
| `--force` | 强制重跑当前区间节点 |
| `--from_node` / `--to_node` | 局部重跑 |
| `--dry_run_relations` | 关系 LLM 不调 API |
| `--skip_eval` | 跳过 Presence 评测 |
| `--api_key` | DashScope 密钥（或环境变量） |

更多命令示例见 [`README.md`](../README.md)。

### 6.2 配置选型（摘要）

| 场景 | 推荐配置 |
|------|----------|
| 无 GPU/API 冒烟 | `dry_run.json` |
| 仅验关系 VL | `run_with_api.json` |
| Rex 真检测 + VL 验收 | `rexomni_full.json` |
| 批量生产（注意 LLM 成本） | `default.json` 或定制；**长视频需限制 `relations.max_windows`** |
| Presence 评测链路 | `dry_run_eval.json` 或 `production.json` |

完整说明见 [`configs/CONFIGS.md`](../configs/CONFIGS.md)。

---

## 7. 模型与算法分工

| 阶段 | 实现 | 后端选项 |
|------|------|----------|
| 检测 | `step1_detect` | `mock` / `rexomni` / `dinox`（经 legacy 适配） |
| 追踪 | `step2_track` | `mock` / `legacy`（OC-SORT） |
| 规则关系 | `relation_rule` | 纯几何/时序投票，见 [`RELATIONS.md`](RELATIONS.md) §3.1 |
| 片段关系 | `relation_llm` | DashScope VL + storyboard 分组询问 |
| 全局/复核 | `global_relation` / `relation_verify` | 规则聚合 + 可选 VL |

**谓词体系**：16 个 canonical 谓词定义于 `configs/predicate_taxonomy.json`；VL 默认候选 10 个，规则与 taxonomy 覆盖范围不完全一致——详见 [`RELATIONS.md`](RELATIONS.md) §2。

---

## 8. 产物与评测

### 8.1 最终导出

| 路径 | 内容 |
|------|------|
| `pred/relations_pred.json` | 全 run 关系汇总 |
| `pred/trajectories_pred.json` | 全 run 轨迹汇总（若 export 启用） |
| `videos/<id>/export/relation_box_vis.mp4` | 关系可视化（`export.relation_viz_video=true`） |
| `reports/run_report.md` | 自动生成 run 报告 |

补生成可视化：

```bash
python scripts/render_relation_video.py --run_dir runs/<run_id>
```

### 8.2 Gold 与 Presence

- **Gold**：`gold/relations_gold.json`（及可选 `trajectories_gold.json`），schema 与 pred 对齐。
- **「50 条 Gold」**：大创第一阶段约 50 视频人工标注，用于批量 P/R/F1；仓库内目前仅有 `validation_dummy` smoke 样例。
- **Presence 指标**：按 `(subject_track_id, predicate, object_track_id)` 是否出现计 TP/FP/FN，不比较起止帧细粒度。
- **工具**：`tools/evaluate_presence.py`；配置 `evaluate.enabled: true` 且提供 `--skip_eval` 未禁用时自动跑。

Gold 说明见 [`gold/README.md`](../gold/README.md)。

---

## 9. 可恢复性与调试

每个节点写入：

```text
runs/<run_id>/videos/<video_id>/<node>/status.json
runs/<run_id>/videos/<video_id>/<node>/run.log        # 部分节点
runs/<run_id>/run_manifest.json
```

**恢复失败任务**：同一命令加 `--resume`。

**局部重跑**（例：只重跑关系到导出）：

```bash
python -m vidvrd_auto.cli \
  --videos data/videos.txt \
  --run_dir runs/exp001 \
  --config configs/default.json \
  --from_node relation_rule \
  --to_node export \
  --force
```

**注意**：正在运行的进程不会加载新代码；改实现后需新 `run_dir` 或 `--force` 重跑相关节点。

Agent 侧故障诊断流程见 [`WORKFLOW_AGENT.md`](WORKFLOW_AGENT.md)。

---

## 10. 当前能力边界（2026-05）

### 已验收

- 12 节点主链 + `--resume` + `run_manifest.json`
- Mock 全流程（`dry_run.json`）
- Rex-Omni-AWQ 真检测 + legacy 追踪 + VL 关系（`rexomni_full.json`，单视频 12/12）
- `relation_llm` 主包化（`clip_relation.py`，非 subprocess）
- Presence 评测链路（Gold 样例级）
- 关系可视化（`relation_viz.py`）

### 未完成 / 限制

| 项 | 说明 |
|----|------|
| 真 Gold 批量 | 约 50 条人工标注待入仓，无法做有意义批量 P/R/F1 |
| Step1/Step2 迁入 | 仍经 `my_scripts` legacy 适配 |
| 谓词三处不同步 | taxonomy(16) / VL 默认(10) / 规则产出(12+) 需后续统一 |
| 关系检测质量 | **后续重点优化**（见 [`RELATIONS.md`](RELATIONS.md) §11），当前不实施 |
| 长视频 LLM 成本 | `default.json` 中 `relations.max_windows: 0` 会对所有窗口调 VL |
| 强模型终审 | `relation_verify.strong_model_review_enabled` 默认 false |
| verify 闭环 | QC 报告为主，`final_actions` 默认不自动改写 relations |

工程分工与里程碑见 [`plan.md`](../plan.md)。

---

## 11. 验证与测试

```bash
python scripts/check_openclaw_env.py
python -m pip install -e .
python -m compileall -q src scripts tests
python -m unittest discover -s tests
```

Mock 端到端 smoke：

```bash
python scripts/make_validation_dummy.py
python -m vidvrd_auto.cli \
  --video data/validation_dummy.mp4 \
  --run_dir runs/smoke \
  --config configs/dry_run.json \
  --resume --dry_run_relations --skip_eval
```

---

## 12. OpenClaw 集成

Skill 路径：[`skills/vidvrd-full-auto/SKILL.md`](../skills/vidvrd-full-auto/SKILL.md)

Agent 职责：环境检查 → 选配置 → 调 CLI → 读 manifest → 失败 `--resume` → 汇报路径与 QC 摘要。不内嵌业务逻辑。

---

## 13. 文档阅读顺序（建议）

1. **本文** — 建立全局认识  
2. [`README.md`](../README.md) — 安装与跑通第一条命令  
3. [`ARCHITECTURE.md`](ARCHITECTURE.md) — 包结构与可恢复性细节  
4. [`RELATIONS.md`](RELATIONS.md) — 关系检测与谓词（改关系必读）  
5. [`SCHEMAS.md`](SCHEMAS.md) — 对接 pred/Gold/下游消费  
6. [`configs/CONFIGS.md`](../configs/CONFIGS.md) — 换配置跑不同场景  
7. [`WORKFLOW_AGENT.md`](WORKFLOW_AGENT.md) — Agent 运维  

维护关系检测优化 backlog 时，以 [`RELATIONS.md`](RELATIONS.md) §11 为任务来源。
