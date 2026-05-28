# VIDVRD 全自动视频关系标注

本项目面向大创「Agent 赋能的视频关系理解」，用 OpenClaw/Agent 调用一条稳定、可复现、可恢复的全自动标注主链，把输入视频转换为可审计的结构化结果：

- **物体轨迹**：逐帧 `track_id`、类别、bbox（`trajectories_pred.json`）
- **视频关系**：主体/客体 track、谓词、起止帧、置信度、来源（`relations_pred.json`）
- **质检与报告**：`relation_qc.json`、`reports/run_report.md`；可选 **Presence P/R/F1**（相对 Gold）

## 当前进度（2026-05）

| 能力 | 状态 |
|------|------|
| 12 节点主链 + `--resume` | 已通 |
| Mock 全流程（无 GPU/API） | `configs/dry_run.json` |
| DashScope VL（关系 + 多节点传图） | `run_with_api.json` / `run_with_vl.json` |
| **Rex-Omni-AWQ 真检测** + legacy 追踪 + VL 关系 | `configs/rexomni_full.json`，已在 `data/test1_video.mp4` 全链验收 |
| `relation_llm` | 主包 `relations/clip_relation.py`（非 subprocess） |
| Gold / Presence 评测 | 链路已通；**仅 1 条 smoke 样例**，待标注组交付约 50 条真 Gold |
| Step1/Step2 迁入 `src/vidvrd_auto` | 未完成，仍经 `my_scripts` 适配层 |

更细分工与待办见根目录 [`plan.md`](plan.md)；大创两阶段目标见 [`plan/plan.md`](plan/plan.md)。

## 架构

```text
VIDVRD/
├── src/vidvrd_auto/          # 主工程包（CLI、pipeline、nodes、relations、evaluation）
├── configs/                  # 运行配置（见 configs/CONFIGS.md）
├── gold/                     # 人工 Gold（评测用，schema 与 pred 对齐）
├── skills/vidvrd-full-auto/  # OpenClaw Skill
├── scripts/                  # 薄入口与环境检查、一键脚本
├── tests/                    # unittest / smoke
├── docs/                     # 架构、schema、Agent 工作流
├── my_scripts/               # Step1/Step2 过渡实现（含 Rex-Omni 检测器）
├── tools/                    # evaluate_presence 等
└── plan/                     # 大创会议与阶段规划
```

新开发默认进入 `src/vidvrd_auto/`，不要新增平行主链。

## 环境

推荐 **`vidvrd` conda 环境**（含 torch、opencv、Rex 依赖）。系统 Python 3.11 可跑 mock/smoke，但 Rex 与完整单测建议在 conda 内执行。

```bash
conda activate vidvrd
python -m pip install -e .
python scripts/check_openclaw_env.py
```

环境变量（**勿写入仓库**）：

| 变量 | 用途 |
|------|------|
| `DASHSCOPE_API_KEY` | DashScope：`relation_llm`、`track_qc`、`global_relation` 等 VL |
| `DINOX_API_TOKEN` | 仅当 `detector.backend=dinox` |

Rex-Omni 权重目录（默认 `D:/Rex-Omni-AWQ/`，在仓库外）需本地存在；`model.safetensors` 体积大，**勿提交 Git**。其它机器可复制 `configs/local_rex.example.json` 为 `configs/local_rex.json` 并改路径，或设置环境变量 `REXOMNI_MODEL_PATH`。

## 配置文件

合并规则：以 `configs/default.json` 为底，用户 JSON 深度覆盖。详见 [`configs/CONFIGS.md`](configs/CONFIGS.md)。

| 文件 | 检测/追踪 | VL/API | 典型用途 |
|------|-----------|--------|----------|
| `dry_run.json` | mock | 关系与各 VL 节点 `*_dry_run` | CI、冒烟、无密钥 |
| `dry_run_eval.json` | mock | dry_run | 验证 Presence 评测链路 |
| `run_with_api.json` | mock | **仅** `relation_llm` 真 API | 快速验关系模型 |
| `run_with_vl.json` | mock | screen / track_qc / global / relation_llm 真 VL | 验传图全链（慢） |
| `default.json` | Rex-Omni + legacy 追踪 | 可配；`relations.max_windows: 0` 会对**所有窗口**调 LLM | 正式默认模板 |
| `rexomni_full.json` | Rex + legacy | 真 VL；`max_windows: 2`；`keyframe_screen` 关闭 | **推荐真模型单视频验收** |
| `rexomni_full_kf2.json` | 同上，`keyframe_interval: 2` | 同上 | 更密关键帧实验（更慢） |
| `production.json` | 继承 default | 开评测 | 在 default 上启用 evaluate |
| `production_full.json` | DINO-X | 需 `DINOX_API_TOKEN` | 云端检测备选 |

## 快速运行

### 1. Mock 冒烟（无需 GPU / API）

```bash
python scripts/make_validation_dummy.py
python -m vidvrd_auto.cli \
  --video data/validation_dummy.mp4 \
  --run_dir runs/smoke \
  --config configs/dry_run.json \
  --resume --dry_run_relations --skip_eval
```

### 2. Mock + 真实 DashScope（PowerShell 示例）

```powershell
$env:DASHSCOPE_API_KEY = "sk-你的key"
.\scripts\run_with_api.ps1 -RunDir runs/live_api
# 或全链路 VL：
.\scripts\run_with_vl.ps1 -RunDir runs/live_vl
```

### 3. Rex-Omni 真检测 + 全链 VL（推荐用真实内容视频）

`validation_dummy.mp4` 上 Rex 常检出 0 个目标，主链会在检测后无轨迹；请用 **`data/test1_video.mp4`** 或自有视频。

```powershell
$env:DASHSCOPE_API_KEY = "sk-你的key"
python scripts/run_vidvrd_auto.py `
  --video data/test1_video.mp4 `
  --run_dir runs/rex_prod `
  --config configs/rexomni_full.json `
  --resume --skip_eval
```

首次 Step1 加载 AWQ 约 1–3 分钟；`relation_llm` 按窗口多次调用 VL，终端可能长时间无新行，属正常——查看 `videos/<id>/relation_llm/run.log` 与 `relations_llm.json.progress.json`。

### 4. 批量

```bash
python -m vidvrd_auto.cli \
  --videos data/videos.txt \
  --run_dir runs/batch001 \
  --config configs/rexomni_full.json \
  --resume --api_key "$DASHSCOPE_API_KEY"
```

薄入口（等价 CLI）：

```bash
python scripts/run_vidvrd_auto.py --video data/test1_video.mp4 --run_dir runs/exp001 --config configs/default.json --resume
```

## 主链节点

```text
video_ingest → audio_prior → step1_detect → keyframe_screen → step2_track
  → track_qc → relation_rule → relation_llm → relation_merge
  → global_relation → relation_verify → export
```

- 每节点写 `status.json`；`input_hash` 一致且 `--resume` 时跳过已成功节点。
- `step1_detect`：`backend=mock` 走主包 mock；否则 `legacy_step1` → `my_scripts/step1_*.py`（Rex / DINO-X）。
- `step2_track`：`backend=mock` 或 `legacy` → `my_scripts` OC-SORT 流程。
- `relation_llm`：`relations/clip_relation.py`（storyboard + 分组 VL）。

## 输出

目录：`runs/<run_id>/`

| 路径 | 说明 |
|------|------|
| `run_manifest.json` | 全 run 状态、配置 hash、每视频每节点 |
| `pred/relations_pred.json` | 全视频关系汇总 |
| `videos/<video_id>/export/relations_pred.json` | 单视频关系 |
| `videos/<video_id>/export/trajectories_pred.json` | 单视频轨迹 |
| `videos/<video_id>/export/relation_qc.json` | 关系质检 |
| `videos/<video_id>/export/relation_box_vis.mp4` | `export.relation_viz_video=true` 时（框 + 关系叠画） |
| `reports/run_report.md` | 跑完自动生成 |
| `reports/presence_report.md` | `evaluate.enabled=true` 且 Gold、pred 均存在时 |

补生成关系可视化（已有 run）：

```bash
python scripts/render_relation_video.py --run_dir runs/test1_kf2
```

## Gold 标注与「50 条 Gold」

**Gold** = 人工标注的**标准答案**，文件在 `gold/`，格式与 `pred/relations_pred.json` 一致（见 `docs/SCHEMAS.md`）：

- `gold/relations_gold.json`：按 `video_id` 索引，每条含 `subject_track_id`、`object_track_id`、`predicate`、`start_frame`、`end_frame`
- `gold/trajectories_gold.json`：轨迹 Gold（可选，后续轨迹评测）

**「50 条 Gold」** 来自 [`plan/plan.md`](plan/plan.md) 第一阶段目标：标注组对 **约 50 个视频** 完成人工轨迹 + 时间段关系标注，工程组用同一批视频的自动结果做 **Presence P/R/F1** 迭代。  
当前仓库仅有 **`validation_dummy` 的 2 条关系样例**，用于验证 `tools/evaluate_presence.py` 与 `reports/presence_report.md` 链路，**不能代表真实指标**。

启用评测（需 Gold 中含对应 `video_id`）：

```bash
python -m vidvrd_auto.cli \
  --video data/validation_dummy.mp4 \
  --run_dir runs/smoke_eval \
  --config configs/dry_run_eval.json \
  --resume --dry_run_relations
# 或正式配置中 evaluate.enabled: true，且不要 --skip_eval
```

**Presence** 指标：按 `(subject_track_id, predicate, object_track_id)` 是否在 Gold/Pred 中出现计 TP/FP/FN（不比较起止帧细粒度）。实现见 `tools/evaluate_presence.py`。

## OpenClaw

Skill：[`skills/vidvrd-full-auto/SKILL.md`](skills/vidvrd-full-auto/SKILL.md)

Agent 只调 CLI、读 `run_manifest.json`、失败节点 `--resume`；业务逻辑在 `src/vidvrd_auto/`。

## 文档

| 文档 | 内容 |
|------|------|
| **`docs/TECHNICAL.md`** | **技术总览（建议先读）** |
| `docs/ARCHITECTURE.md` | 架构专篇：包结构、可恢复性 |
| `docs/RELATIONS.md` | 关系检测：谓词体系、规则/VL/merge |
| `docs/SCHEMAS.md` | JSON 产物字段 |
| `docs/WORKFLOW_AGENT.md` | Agent 工作流 |
| `configs/CONFIGS.md` | 配置文件说明 |
| `plan.md`（根目录） | 四人分工与工程现状 |
| `plan/plan.md` | 大创两阶段、50 条 Gold 交付 |
| `gold/README.md` | Gold 文件说明 |

## 旧代码与迁移

- `my_scripts/`：Step1 检测（Rex-Omni / DINO-X）、Step2 追踪仍由此执行；`my_scripts/modules/rexomni_detector.py` 含 Windows + AWQ 加载修复。
- `auto_label/vidvrd_auto_label.py`：旧入口，委托 `vidvrd_auto.cli`。
- 目标：检测/追踪逐步迁入 `src/vidvrd_auto/detection`、`tracking`，主链不再 subprocess 业务脚本。

## 验证

```bash
python scripts/check_openclaw_env.py
python -m pip install -e .
python scripts/make_validation_dummy.py
python -m vidvrd_auto.cli --help
python -m compileall -q src scripts tests
python -m unittest discover -s tests
python -m unittest tests.test_pipeline_smoke -v
```
