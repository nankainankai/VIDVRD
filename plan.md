# 全自动视频关系标注项目分工计划

## 当前项目状态

项目已经形成 OpenClaw-first 的全自动标注主链 MVP，具备统一 CLI、节点编排、配置管理、缓存恢复、manifest 记录、基础 dry-run 和 unittest 验收能力。

当前已经具备的主链为：

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

但项目还不是最终完整形态：检测、追踪、片段关系分类仍有旧脚本适配层；关键帧粗筛、轨迹质检、全局关系、强模型复核虽然已接入 VL/强模型调用入口，但仍需要真实视频验证、Prompt 调优和规则完善。

**2026-05-26 工程闭环更新**：已支持 `detector.backend=mock` / `tracking.backend=mock` 的端到端 dry-run（无需 GPU/API）；`scripts/check_openclaw_env.py`、`scripts/make_validation_dummy.py`、`tests/test_pipeline_smoke.py` 已加入；OpenClaw Skill 与 README 已补充 smoke 命令。

**2026-05-26 plan 续做**：`gold/` 样例 + Presence 评测链路；`pipeline/report.py` 自动生成 `reports/run_report.md`；`configs/production_full.json` + `scripts/run_production.ps1`；storyboard `imwrite` 修复（中文路径）；规则关系扩展 motion（toward/away/follow/chase 等）；`configs/CONFIGS.md`。

**2026-05-26 选项1**：`relation_llm` 迁入主包（`relations/clip_relation.py` 等）；`semi_auto_label_relations.py` 改为薄 CLI 包装。

**2026-05-26 选项2**：`keyframe_screen` / `track_qc` / `global_relation` / `relation_verify` 在 `vl_enabled` 时从视频抽帧拼图并 `call_bgr` 调用 VL；`utils/vl_frames.py`；`dry_run.json` 默认 `vl_dry_run=true` 可本地验证传图链路。

## 分工原则

- 所有新代码默认进入 `src/vidvrd_auto/`。
- 所有配置默认进入 `configs/`。
- 所有说明、注释、运行文档尽量使用中文。
- 所有真实测试统一在 `vidvrd` conda 环境运行。
- `my_scripts/` 只作为过渡兼容层，后续逐步把能力迁入新包。

## 成员 1：工程主链与实验闭环（吴、李）

负责方向：保证整个系统可跑、可恢复、可比较、可汇报。

主要任务：

- 维护 `vidvrd_auto.cli`、`pipeline/runner.py`、`run_manifest.json`、节点缓存和断点续跑逻辑。
- 强化 OpenClaw Skill，使 Agent 能完成环境检查、运行、失败诊断、resume 和结果汇报。
- 维护 `configs/default.json`、`configs/dry_run.json`、配置模板和参数说明。
- 维护批量运行、dry-run、mock 测试和 smoke test。
- 统计模型调用成本、失败节点分布、成功率和产出率。
- 维护中文 `README.md`、`docs/WORKFLOW_AGENT.md`、`docs/RUN_REPORT_TEMPLATE.md`。

对应模块：

- `src/vidvrd_auto/cli.py`
- `src/vidvrd_auto/pipeline/`
- `src/vidvrd_auto/config/`
- `skills/vidvrd-full-auto/`
- `docs/`
- `tests/`

难度与工作量：中高。虽然主链骨架已有，但后续所有成员的模块都需要接入、验收和报告，成员 1 负责项目工程闭环。

## 成员 2：检测、关键帧筛选与轨迹质量（张）

负责方向：完成会议方案第 2、3、4 步，让视频稳定产出高质量检测框和轨迹。

主要任务：

- ※从my_scripts里下载本地部署Rex-Omni-AWQ，接入自己的DINO-X API，部署并验证检测后端。
- 完善关键帧检测策略，控制关键帧频率、检测阈值、可视化输出。
- 完善 `keyframe_screen`，让 VL 模型真正判断 `keep/drop/crop`。
- 实现可选视频裁剪节点，根据粗筛结果生成高质量片段。
- 完善逐帧检测与插值补框策略。
- 完善 OC-SORT 追踪、窗口切分和 pair 可视化。
- 完善 `track_qc`，使用规则和 VL 模型判断是否同一物体、框是否偏移、类别是否正确。
- 逐步把旧 Step1/Step2 逻辑迁入 `src/vidvrd_auto/detection/` 和 `src/vidvrd_auto/tracking/`。

对应模块：

- `src/vidvrd_auto/detection/`
- `src/vidvrd_auto/tracking/`
- `src/vidvrd_auto/nodes/detect.py`
- `src/vidvrd_auto/nodes/screen.py`
- `src/vidvrd_auto/nodes/track.py`
- `src/vidvrd_auto/nodes/track_qc.py`
- `configs/default.json` 中 detector、tracking、keyframe_screen、track_qc 配置

难度与工作量：高。检测和追踪质量直接决定后续关系结果上限，需要处理模型部署、速度、召回、漂移和成本。

## 成员 3：谓词体系、规则关系与片段关系分类（李、吴）

负责方向：完成会议方案第 5、6 步，让每个视频片段能生成可靠关系候选。

主要任务：

- 完善 `configs/predicate_taxonomy.json`，定义谓词中文解释、层级、互斥组、反向耦合、接触要求、运动要求。
- 明确易混谓词边界，例如 `sing with` / `sing to`、接触与靠近、跟随与朝向。
- 扩展规则关系，包括左右、上下、前后、near、overlap/contact、toward/away、follow、moving_together 等。
- 维护片段 storyboard 生成、轨迹 ID 显示、类别提示、音频先验注入。
- 优化片段关系 Prompt，使大模型输出带起止帧、置信度、证据和来源。
- 完善 LLM 输出解析和异常处理，保证模型输出能稳定转成统一 JSON。
- 逐步把旧半自动关系脚本迁入 `src/vidvrd_auto/relations/clip_classifier.py`。

对应模块：

- `configs/predicate_taxonomy.json`
- `src/vidvrd_auto/relations/taxonomy.py`
- `src/vidvrd_auto/relations/ops.py`
- `src/vidvrd_auto/relations/clip_classifier.py`
- `src/vidvrd_auto/nodes/relation_llm.py`
- `src/vidvrd_auto/prompts/`

难度与工作量：高。关系类别定义和片段关系 Prompt 是最终标注质量的核心，需要反复结合 Gold 样例和失败案例调整。

## 成员 4：全局关系、强模型复核与评测交付（黄）

负责方向：完成会议方案第 7、8 步，并负责最终结果能否对齐 Gold 和用于汇报。

主要任务：

- 完善 `global_relation`，对多个窗口关系做跨窗聚合，减少碎片化和重复关系。
- 重点处理动态关系，如 `toward`、`away`、`follow`、`chase`、`moving_together`。
- 调用 VL/强模型查看全视频抽帧和片段关系，复核全局关系。
- 完善 `relation_verify`，处理互斥冲突、低置信度、规则与模型冲突、缺失反向关系。
- 实现强模型最终动作：保留、删除、修改谓词、调整起止帧、补全耦合关系。
- 维护最终导出 schema，保证 `relations_pred.json` 和 `trajectories_pred.json` 可被评测脚本读取。
- 维护 Gold/Pred 对齐和 Presence P/R/F1 评测。
- 产出失败案例分析和质量报告。

对应模块：

- `src/vidvrd_auto/nodes/global_relation.py`
- `src/vidvrd_auto/relations/verify.py`
- `src/vidvrd_auto/relations/ops.py`
- `src/vidvrd_auto/nodes/export.py`
- `src/vidvrd_auto/evaluation/`
- `tools/evaluate_presence.py`
- `gold/`

难度与工作量：中高。该成员负责最终质量闭环，需要综合规则、模型结果、冲突处理和评测指标。

## 协作节奏

建议按三轮推进：

1. 第一轮：每人负责模块跑通 dry-run 和小样例，确保主链不断。
2. 第二轮：接入真实模型和真实视频，记录失败案例并迭代 Prompt/规则。
3. 第三轮：用 Gold 集合评测，输出指标表、失败类型分布和最终汇报材料。

每次合并前至少完成：

- `conda run -n vidvrd python -m vidvrd_auto.cli --help`
- `conda run -n vidvrd python -m compileall -q src scripts tests`
- `conda run -n vidvrd python -m unittest discover -s tests`

## 最终验收目标

- 给定一组视频，能够一键产出轨迹、关系和质检报告。
- 任意节点失败后，能够定位原因并 `--resume` 恢复。
- 会议 8 步中涉及大模型判断的环节都有真实实现或明确 dry-run/mock 模式。
- 旧 `my_scripts/` 不再被主链直接依赖，最多作为兼容包装存在。
- 新成员只读中文 README 和 Agent 工作流文档即可理解项目如何运行。
