# VIDVRD 全流程说明：产出与接口

本文档面向 VIDVRD 的全自动标注主链，统一说明“从输入到产出”的完整流程、各节点的输入输出接口、运行目录结构、关键配置项，以及恢复运行时的接口语义。当前稳定主入口是 `src/vidvrd_auto/cli.py`，薄包装入口是 `scripts/run_vidvrd_auto.py`。

## 1. 总体流程

主链按固定顺序执行，节点顺序由 `src/vidvrd_auto/pipeline/constants.py` 维护：

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

整体职责划分如下：

- `video_ingest`：把本地视频或 URL 统一物化到运行目录。
- `audio_prior`：读取音频先验，给关系推理提供辅助标签。
- `step1_detect`：生成逐帧检测结果。
- `keyframe_screen`：用规则或可选 VL 模型判断是否继续处理。
- `step2_track`：生成轨迹与窗口切片。
- `track_qc`：对轨迹做质检，记录风险。
- `relation_rule`：从几何和时序规则生成关系候选。
- `relation_llm`：对窗口片段做多模态关系分类。
- `relation_merge`：合并规则与模型结果。
- `global_relation`：跨窗口聚合动态关系。
- `relation_verify`：做关系复核和质检。
- `export`：导出最终交付文件。

## 2. 运行接口

### 2.1 主命令

推荐直接调用 Python 模块：

```bash
conda run -n vidvrd python -m vidvrd_auto.cli --videos data/videos.txt --run_dir runs/exp001 --config configs/default.json --resume --api_key YOUR_DASHSCOPE_KEY
```

不安装包时可以用薄入口：

```bash
conda run -n vidvrd python scripts/run_vidvrd_auto.py --videos data/videos.txt --run_dir runs/exp001 --config configs/default.json --resume
```

### 2.2 CLI 参数

- `--video`：单个视频路径或 URL。
- `--videos`：视频列表文件，或逗号分隔的视频路径 / URL。
- `--run_dir`：本次运行目录，例如 `runs/exp001`。
- `--config`：JSON 配置文件，默认 `configs/default.json`。
- `--api_key`：DashScope API key，未传时回退到 `DASHSCOPE_API_KEY`。
- `--resume`：复用已成功节点，适合中断恢复。
- `--force`：忽略缓存，强制重跑选中节点。
- `--from_node` / `--to_node`：局部重跑区间。
- `--dry_run_relations`：关系节点只产 storyboard，不真正调用多模态关系推理。
- `--skip_eval`：跳过 Presence 评测。

### 2.3 输入约定

- 本地路径默认相对仓库根目录解析。
- URL 会先下载到 `runs/<run_id>/videos/<video_id>/inputs/`。
- `--videos` 文件支持空行和 `#` 注释行。

## 3. 节点级接口与产出

下表按主链顺序列出每个节点的主要输入和输出。每个节点都会在 `runs/<run_id>/videos/<video_id>/<node>/status.json` 记录状态，供 `--resume` 判断是否可复用。

| 节点 | 主要输入 | 主要输出 | 说明 |
|---|---|---|---|
| `video_ingest` | 原始视频路径或 URL、`video_ingest` 配置 | `inputs/source.json`、统一落盘的视频文件 | 物化输入边界，写入文件 hash、来源、路径等元信息。 |
| `audio_prior` | `source`、`audio_prior` 配置 | `audio_prior/audio_prior.json` | 生成音频先验标签，常作为关系节点的辅助条件。 |
| `step1_detect` | 视频文件、`detector` 配置 | `step1_detect/detections_full.jsonl`、`step1_detect/video_meta.json` | 当前由旧 Step1 检测脚本适配实现。 |
| `keyframe_screen` | `detections_full.jsonl`、`keyframe_screen` 配置 | `keyframe_screen/screen_result.json` | 输出 `keep`、`drop`、`crop` 决策以及裁剪建议。 |
| `step2_track` | 视频文件、检测结果、`tracking` 配置 | `step2_track/tracks_full.jsonl`、`step2_track/windows.json` | 当前由旧 Step2 追踪脚本适配实现。 |
| `track_qc` | `tracks_full.jsonl`、`windows.json`、`track_qc` 配置 | `track_qc/track_qc.json` | 记录短轨迹、类别漂移、跳变风险和可选 VL 复核结果。 |
| `relation_rule` | `tracks_full.jsonl`、`windows.json`、`relation_rule` 配置 | `relation_rule/relations_rule.json` | 基于几何、接触、时序等规则生成候选关系。 |
| `relation_llm` | `tracks_full.jsonl`、`windows.json`、`relations` 配置、音频先验 | `relation_llm/relations_llm.json`、storyboard 目录 | 多模态模型关系分类节点，支持 `dry_run`。 |
| `relation_merge` | 规则关系 + LLM 关系 | `relation_merge/relations_merged.json` | 合并并去重候选关系，可做耦合补全。 |
| `global_relation` | `relations_merged.json` | `global_relation/relations_global.json` | 跨窗口聚合动态关系，减少碎片化。 |
| `relation_verify` | `relations_global.json`、`tracks_full.jsonl`、`relation_verify` 配置 | `relation_verify/relations_verified.json`、`relation_verify/relation_qc.json` | 复核低置信度与冲突关系，并产出质检结果。 |
| `export` | `relations_verified.json`、`tracks_full.jsonl`、`relation_qc.json` | `export/relations_pred.json`、`export/trajectories_pred.json`、`export/relation_qc.json` | 统一输出交付 schema，不再做模型推理。 |

## 4. 最终产物

### 4.1 单视频目录

每个视频的中间结果和最终结果都写在：

```text
runs/<run_id>/videos/<video_id>/
```

典型目录结构如下：

```text
inputs/source.json
audio_prior/audio_prior.json
step1_detect/detections_full.jsonl
step1_detect/video_meta.json
keyframe_screen/screen_result.json
step2_track/tracks_full.jsonl
step2_track/windows.json
track_qc/track_qc.json
relation_rule/relations_rule.json
relation_llm/relations_llm.json
relation_merge/relations_merged.json
global_relation/relations_global.json
relation_verify/relations_verified.json
relation_verify/relation_qc.json
export/relations_pred.json
export/trajectories_pred.json
export/relation_qc.json
```

### 4.2 运行级目录

运行级输出位于：

```text
runs/<run_id>/
```

其中最重要的文件是：

- `run_manifest.json`：本次运行的总清单。
- `pred/relations_pred.json`：全视频级最终关系文件。
- `reports/presence_report.md`：存在 Gold 且启用评测时生成。

### 4.3 运行清单字段

`run_manifest.json` 记录运行级元信息，核心字段包括：

- `run_dir`
- `started_at` / `finished_at`
- `config_path`
- `config_hash`
- `nodes`
- `videos`
- `pred_relations_json`
- `pred_relation_count`
- `evaluate`

每个视频条目会记录：

- `video_id`
- `source`
- `source_type`
- `path`
- `file_hash`
- `state`
- `skip_reason` 或 `error`
- `nodes`：各节点的 `status.json` 聚合视图

## 5. 关键 schema

本仓库的稳定 schema 由 [docs/SCHEMAS.md](SCHEMAS.md) 维护，这里只列主链最关键的几个接口约定。

### 5.1 `status.json`

每个节点一个状态文件，路径固定为 `runs/<run_id>/videos/<video_id>/<node>/status.json`。常见字段：

- `node`
- `state`：`running`、`succeeded`、`failed`
- `input_hash`
- `started_at` / `finished_at`
- `outputs`
- `error`

### 5.2 `source.json`

`video_ingest` 写入 `inputs/source.json`，包含：

- `source`
- `source_type`
- `video_path`
- `video_path_rel`
- `downloaded`
- `exists`
- `file_size`
- `file_hash`

### 5.3 `screen_result.json`

`keyframe_screen` 输出：

- `passed`
- `decision`
- `reason`
- `crop_suggestion`
- `sampled_frames`
- `vl_screen`

### 5.4 `track_qc.json`

`track_qc` 输出：

- `track_count`
- `short_track_count`
- `class_drift_count`
- `large_jump_count`
- `risk_items`
- `vl_review`
- `passed`

### 5.5 `relation_qc.json`

`relation_verify` 和 `export` 共同使用的关系质检文件，常见字段：

- `relation_count`
- `low_confidence_count`
- `conflict_count`
- `strong_model_review_enabled`
- `strong_model_review_count`
- `final_actions`

### 5.6 `relations_pred.json`

最终关系文件按视频 ID 聚合，内部统一优先使用 `subject_track_id` 和 `object_track_id`：

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

### 5.7 `trajectories_pred.json`

轨迹导出文件按视频 ID 聚合，单条轨迹通常包含：

- `track_id`
- `category`
- `trajectory`

其中 `trajectory` 是以帧号为键、bbox 为值的映射。

## 6. 配置接口

默认配置在 [configs/default.json](../configs/default.json) 中，由 [src/vidvrd_auto/config/loader.py](../src/vidvrd_auto/config/loader.py) 做深度合并。用户自定义配置只需要覆盖必要字段。

### 6.1 主要配置分组

- `video_ingest`：下载超时、是否覆盖下载。
- `models.vl` / `models.strong_vl`：VL 客户端、模型名、重试、退避、dry-run。
- `audio_prior`：音频先验 CSV、列名和 fallback 标签。
- `detector`：检测后端、Rex-Omni / DINO-X 参数、关键帧间隔、保存视频框等。
- `tracking`：窗口大小、步长、轨迹关联和可视化参数。
- `keyframe_screen`：粗筛阈值、是否启用 VL 复核。
- `track_qc`：轨迹短轨、漂移、跳变阈值。
- `relation_rule`：规则关系阈值和候选对上限。
- `relations`：窗口分组、关系模型、dry-run、重试参数。
- `relation_merge`：耦合补全开关。
- `global_relation`：动态谓词和跨窗口聚合阈值。
- `relation_verify`：低置信度阈值、强模型复核阈值。
- `evaluate`：是否做 Presence 评测以及 Gold 路径。

### 6.2 运行时覆盖规则

- `--config` 指定的 JSON 会在默认配置之上做深度合并。
- `--api_key` 优先于环境变量 `DASHSCOPE_API_KEY`。
- `--dry_run_relations` 会强制关系节点进入 dry-run 行为。
- `--skip_eval` 会跳过评测，即使 `evaluate.enabled` 为真。

## 7. 恢复与重跑语义

主链设计成可恢复：

- 每个节点执行前会比较 `input_hash`。
- `--resume` 且节点状态为 `succeeded`、输出文件仍存在时，会跳过该节点。
- `--force` 会忽略缓存，强制重跑。
- `--from_node` / `--to_node` 只会运行指定区间的节点。
- 如果某个视频在中途失败，其他视频仍会继续处理，并在 `run_manifest.json` 里记录失败状态。

推荐恢复流程：

1. 先看 `runs/<run_id>/run_manifest.json`。
2. 找到失败视频对应的节点状态文件。
3. 先修环境、配置或输入问题，再用原命令加 `--resume`。
4. 如果只需要补跑局部节点，再加 `--from_node` / `--to_node`。

## 8. 推荐交付清单

一次完整运行，通常向外交付以下内容：

- `runs/<run_id>/pred/relations_pred.json`
- `runs/<run_id>/videos/<video_id>/export/trajectories_pred.json`
- `runs/<run_id>/videos/<video_id>/export/relation_qc.json`
- `runs/<run_id>/run_manifest.json`
- `runs/<run_id>/reports/presence_report.md`（若启用评测且 Gold 存在）

如果需要做运行汇报，可直接参考 [docs/RUN_REPORT_TEMPLATE.md](RUN_REPORT_TEMPLATE.md)。

## 9. 现有入口与兼容层

- 新主入口：`src/vidvrd_auto/cli.py`
- 薄入口脚本：`scripts/run_vidvrd_auto.py`
- 兼容旧入口：`auto_label/vidvrd_auto_label.py`
- 旧脚本适配层：`my_scripts/`

新开发建议只围绕 `src/vidvrd_auto/` 继续扩展，不再把新编排逻辑写回旧兼容层。
