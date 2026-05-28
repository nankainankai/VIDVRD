# VIDVRD Auto 架构说明

> **文档定位**：架构专篇（包结构、主链顺序、可恢复性）。系统总览见 [`TECHNICAL.md`](TECHNICAL.md)；关系检测见 [`RELATIONS.md`](RELATIONS.md)。

本项目当前采用 OpenClaw-first 的工程结构：OpenClaw/Agent 只负责调用、恢复和汇报，实际业务逻辑全部落在 `src/vidvrd_auto/`。
`relation_llm` 已迁入 `vidvrd_auto.relations.clip_relation`（storyboard 生成 + DashScope VL 分组询问）；`my_scripts/semi_auto_label_relations.py` 仅保留兼容 CLI。检测/追踪仍通过 `legacy_step1` / `legacy_step2` 适配旧脚本。

## 主入口

安装 editable 包后的推荐命令：

```bash
conda run -n vidvrd python -m vidvrd_auto.cli --videos data/videos.txt --run_dir runs/exp001 --config configs/default.json --resume
```

不安装包时使用薄入口：

```bash
conda run -n vidvrd python scripts/run_vidvrd_auto.py --videos data/videos.txt --run_dir runs/exp001 --config configs/default.json --resume
```

## 运行主链

```text
video_ingest        视频读入：本地文件或 URL 统一落盘
  -> audio_prior    音频先验：VGGSound CSV 或配置兜底
  -> step1_detect   检测出框：Rex-Omni/DINO-X 过渡适配
  -> keyframe_screen 关键帧粗筛：规则 + 可选 VL 判断
  -> step2_track    轨迹生成：OC-SORT 过渡适配
  -> track_qc       轨迹质检：规则风险 + 可选 VL 复核
  -> relation_rule  规则关系：几何/接触/时序关系
  -> relation_llm   片段关系：storyboard + 多模态模型
  -> relation_merge 关系合并：去重、耦合补全
  -> global_relation 全局关系：跨窗口聚合和动态关系复核
  -> relation_verify 关系复核：冲突、低置信度、强模型复核
  -> export         导出：轨迹、关系和质检报告
```

## 包结构

```text
src/vidvrd_auto/
├── cli.py              # CLI 参数解析
├── config/             # 配置加载与合并
├── pipeline/           # 节点顺序、manifest、runner
├── nodes/              # 每个流程节点的可调用入口
├── models/             # 统一模型客户端
├── prompts/            # 中文 Prompt 模板
├── detection/          # 检测能力迁移层
├── tracking/           # 追踪能力迁移层
├── relations/          # 关系规则、分类、合并、复核
├── evaluation/         # 评测入口
└── utils/              # IO、路径、hash、进程工具
```

`my_scripts/` 只作为迁移期间的旧脚本适配层，不再承载新编排逻辑。

关系检测的谓词体系、各节点算法与配置说明见 [`RELATIONS.md`](RELATIONS.md)。

## 可恢复性

每个节点会写：

- `runs/<run_id>/videos/<video_id>/<node>/status.json`
- 必要时写 `run.log`
- 节点输出文件路径会登记到 `run_manifest.json`

恢复失败任务时，用同一命令加 `--resume`；只想重跑局部节点时，用 `--from_node` 和 `--to_node`。

## 验证

所有真实验证使用 `vidvrd` 环境：

```bash
$env:PYTHONPATH="src"
conda run -n vidvrd python -m compileall -q src scripts tests
conda run -n vidvrd python -m unittest discover -s tests
```

当前测试使用标准库 `unittest`，不强制依赖 `pytest`。
