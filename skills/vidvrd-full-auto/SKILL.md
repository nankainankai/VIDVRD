---
name: vidvrd-full-auto
description: 运行 VIDVRD 全自动视频关系标注流程。用户要求基于 OpenClaw 自动标注视频、批量生成轨迹和关系、Rex-Omni 真检测、恢复失败运行、Gold/Presence 评测或查看运行报告时使用。
---

# VIDVRD 全自动标注

## 用途

调用本仓库 **OpenClaw-first** 主链（12 节点）。确定性 runner：`vidvrd_auto.cli`（`src/vidvrd_auto/`）。Skill 只说明如何调用、读输出、恢复失败；业务逻辑不在 Skill 内实现。

**现状摘要**（详见根目录 `README.md`、`plan.md`）：

- Mock 全流程、`run_with_api` / `run_with_vl`、**Rex-Omni-AWQ 真检测** 均已验收。
- `relation_llm` 在 `relations/clip_relation.py`（非 subprocess）。
- **50 条 Gold**：标注组待交付；仓库仅 `validation_dummy` smoke 样例，Presence 评测链路可测但无真实批量指标。

## 前置检查

```bash
conda run -n vidvrd python scripts/check_openclaw_env.py
```

无 conda 时：`python scripts/check_openclaw_env.py`

确认项：

1. Python 3.10+；推荐 `vidvrd` conda 或仓库根 `pip install -e .`。
2. 输入：单视频路径/URL，或 `data/videos.txt` 这类列表文件。
3. 配置存在：见下表；默认模板 `configs/default.json`。
4. Smoke：`python scripts/make_validation_dummy.py`。
5. 真 VL / `relation_llm`：环境变量 `DASHSCOPE_API_KEY` 或 `--api_key`（**勿写入仓库**）。
6. **Rex-Omni**：`D:/Rex-Omni-AWQ/`（仓库外，含 `model.safetensors`）；GPU 约 4–6GB；Step1 首次加载约 1–3 分钟。其它机器见 `configs/local_rex.example.json` 或 `REXOMNI_MODEL_PATH`。
7. **DINO-X**：仅 `detector.backend=dinox` 时需要 `DINOX_API_TOKEN`。
8. **无 GPU/API**：`configs/dry_run.json`（mock 检测 + mock 追踪）。
9. CLI 不可导入时：`pip install -e .` 或 `python scripts/run_vidvrd_auto.py`。

### 配置选型

| 配置 | 检测/追踪 | API | 何时用 |
|------|-----------|-----|--------|
| `dry_run.json` | mock | 全 dry_run | 冒烟、CI |
| `dry_run_eval.json` | mock | dry_run | 验 Presence 评测 |
| `run_with_api.json` | mock | 仅 relation_llm | 快速验关系 API |
| `run_with_vl.json` | mock | 多节点 VL 传图 | 验 VL 全链（慢） |
| `rexomni_full.json` | Rex + legacy | 真 VL，`max_windows: 2` | **真模型验收（推荐）** |
| `rexomni_full_kf2.json` | Rex，`keyframe_interval: 2` | 真 VL | 更密关键帧实验 |
| `default.json` | Rex + legacy | 可配；`max_windows: 0` 长视频极慢 | 正式模板 |
| `production_full.json` | DINO-X | 需 token | 云端检测备选 |

合并规则：`default.json` 为底，用户 JSON 深度覆盖（`configs/CONFIGS.md`）。

## 主命令

### 单视频（正式 Rex + VL）

**注意**：`validation_dummy.mp4` 上 Rex 常检出 0 人，主链会在检测后无轨迹。真验收请用 `data/test1_video.mp4` 或真实人物视频。

```bash
conda run -n vidvrd python -m vidvrd_auto.cli \
  --video data/test1_video.mp4 \
  --run_dir runs/<run_id> \
  --config configs/rexomni_full.json \
  --resume \
  --api_key <key>
```

PowerShell 一键（mock + 仅关系 API / mock + 全 VL）：

```powershell
$env:DASHSCOPE_API_KEY = "sk-..."
.\scripts\run_with_api.ps1 -RunDir runs/live_api
.\scripts\run_with_vl.ps1 -RunDir runs/live_vl
```

### 单视频（mock 冒烟）

```bash
conda run -n vidvrd python scripts/make_validation_dummy.py
conda run -n vidvrd python -m vidvrd_auto.cli \
  --video data/validation_dummy.mp4 \
  --run_dir runs/smoke001 \
  --config configs/dry_run.json \
  --resume --dry_run_relations --skip_eval
```

### 批量

```bash
conda run -n vidvrd python -m vidvrd_auto.cli \
  --videos data/videos.txt \
  --run_dir runs/<run_id> \
  --config configs/rexomni_full.json \
  --resume --api_key <key>
```

### Gold / Presence 评测

**50 条 Gold** = 约 50 个视频的人工标准标注（`gold/relations_gold.json` + 可选 `trajectories_gold.json`），与 `pred/relations_pred.json` 同 schema；用于 Presence P/R/F1。当前仓库仅 smoke，见 `plan/plan.md` 第一阶段。

启用评测（Gold 须含对应 `video_id`，勿 `--skip_eval`）：

```bash
conda run -n vidvrd python -m vidvrd_auto.cli \
  --video data/validation_dummy.mp4 \
  --run_dir runs/smoke_eval \
  --config configs/dry_run_eval.json \
  --resume --dry_run_relations
```

## Agent 工作流

1. 新环境：`check_openclaw_env.py` → `dry_run.json` 冒烟。
2. 验 API：`run_with_api.json` 或 `run_with_vl.json`。
3. 验真检测：`rexomni_full.json` + **真实内容视频**（非 dummy）。
4. 读 `runs/<run_id>/run_manifest.json` 与 `reports/run_report.md`。
5. 失败：查节点 `status.json`、`run.log`；`relation_llm` 还可看 `relations_llm.json.progress.json`（seg 内多次 VL，终端可能久无输出属正常）。
6. 相同命令 `--resume`；仅重跑某段用 `--from_node` / `--to_node`；缓存失效用 `--force --from_node <node>`。

`--to_node` 停在 export 前时，manifest 为 `state=partial`，无 `export/*`、不计入 `pred/` 聚合。

节点顺序：

```text
video_ingest → audio_prior → step1_detect → keyframe_screen → step2_track
  → track_qc → relation_rule → relation_llm → relation_merge
  → global_relation → relation_verify → export
```

- `step1_detect`：`mock` 或 `legacy` → `my_scripts`（Rex / DINO-X）。
- `relation_llm`：`clip_relation.py`。

## 输出

向用户报告（完整跑通后）：

- `runs/<run_id>/pred/relations_pred.json`
- `runs/<run_id>/run_manifest.json`
- `runs/<run_id>/reports/run_report.md`（自动生成）
- `runs/<run_id>/videos/<video_id>/export/relations_pred.json`
- `runs/<run_id>/videos/<video_id>/export/trajectories_pred.json`
- `runs/<run_id>/videos/<video_id>/export/relation_qc.json`
- `runs/<run_id>/videos/<video_id>/export/relation_box_vis.mp4`（`export.relation_viz_video=true` 时）
- `runs/<run_id>/reports/presence_report.md`（`evaluate.enabled` 且 Gold、pred 存在时）

`state=partial` 时不承诺 export/pred 存在；提示 `--resume` 跑至 `export`。

补生成关系可视化（不重跑 pipeline）：

```bash
python scripts/render_relation_video.py --run_dir runs/<run_id> --video_id <video_id>
```

## 失败处理

1. `run_manifest.json` → `state=failed` 的视频与节点。
2. `videos/<id>/<node>/status.json`、`run.log`。
3. 修环境/配置/数据后 `--resume`。
4. 必要时 `--force --from_node <node>`。

## 中文诊断

| 现象 | 处理 |
|------|------|
| `missing DASHSCOPE api key` | 设 `DASHSCOPE_API_KEY` 或 `--api_key` |
| Rex 导入/加载失败 | 检查 `rex_model_path`、conda 依赖、`rexomni_detector.py` 日志；或 `backend=dinox` / `mock` |
| Rex 跑通但 0 框、后续全跳过 | 换真实人物视频，勿用 dummy |
| `DINOX_API_TOKEN` 缺失 | 仅 dinox 后端需要 |
| 下载失败 | URL、网络、`video_ingest.download_timeout_sec` |
| `required output missing` | 该节点 `run.log` → `--force --from_node` |
| `relation_llm` 长时间无终端输出 | 看 `relation_llm/run.log`、progress.json；`max_windows` 控制成本 |
| VL JSON 解析失败 | Prompt、返回文本、`vl_dry_run` 配置 |
| Presence skipped | Gold 缺失、`--skip_eval`、`evaluate.enabled=false` |

## 汇报格式

- 运行目录、配置路径。
- 成功/失败/跳过视频数。
- `pred/relations_pred.json` 路径与关系条数（若有）。
- `presence_report.md`（若跑了评测）。
- 失败节点与原因摘要。
- 可参考 `docs/RUN_REPORT_TEMPLATE.md`。

## 延伸阅读

- `README.md` — 快速开始与 Gold 说明
- `plan.md` — 分工与验收
- `plan/plan.md` — 50 条 Gold 与大创两阶段
- `docs/WORKFLOW_AGENT.md` — Agent 细节
