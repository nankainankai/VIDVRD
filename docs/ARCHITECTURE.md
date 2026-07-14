# VIDVRD Auto 架构说明

## 主入口

```bash
python -m vidvrd_auto.cli --videos data/videos.txt --run_dir runs/exp001 --config configs/default.json --resume
```

薄入口（不安装包时）：

```bash
python scripts/run_vidvrd_auto.py --videos data/videos.txt --run_dir runs/exp001 --config configs/default.json --resume
```

## 包结构

```text
src/vidvrd_auto/
├── cli.py              # CLI 参数解析
├── config/             # 配置加载与深度合并
│   └── loader.py
├── pipeline/           # 节点编排
│   ├── constants.py    # NODE_ORDER（12 节点）
│   ├── runner.py       # 主编排器
│   └── manifest.py     # status.json 管理
├── nodes/              # 流程节点（每个节点一个文件）
│   ├── ingest.py       # video_ingest
│   ├── audio_prior.py  # audio_prior
│   ├── detect.py       # step1_detect → legacy_step1 适配
│   ├── screen.py       # keyframe_screen
│   ├── track.py        # step2_track → legacy_step2 适配
│   ├── track_qc.py     # track_qc
│   ├── relation_llm.py # relation_llm → clip_classifier 适配
│   ├── global_relation.py # global_relation
│   └── export.py       # export
├── models/
│   └── vl_client.py    # 统一 DashScope VL 客户端
├── prompts/
│   └── templates.py    # 中文 Prompt 模板
├── relations/          # 关系核心逻辑
│   ├── ops.py          # 规则生成、合并、复核（核心实现）
│   ├── clip_classifier.py # 调旧脚本的适配层
│   ├── object_candidates.py # 类别对→候选谓词
│   └── taxonomy.py     # 谓词定义读取
├── detection/
│   └── legacy_step1.py # Step1 子进程适配
├── tracking/
│   └── legacy_step2.py # Step2 子进程适配
├── evaluation/
│   └── presence.py     # Presence 评测入口
└── utils/
    ├── io.py           # JSON/JSONL 读写
    ├── paths.py        # 仓库根路径
    ├── hashing.py      # 稳定 hash
    └── process.py      # 子进程执行
```

## 旧脚本适配层

`my_scripts/` 中以下文件仍被新包 subprocess 调用：

| 文件 | 调用方 |
|------|--------|
| `step1_full_video_box_detection_dinox.py` | `detection/legacy_step1.py` |
| `step2_full_video_tracking_ocsort_qc_pairviz.py` | `tracking/legacy_step2.py` |
| `semi_auto_label_relations.py` | `relations/clip_classifier.py` |

其余 `my_scripts/` 文件为上述脚本的依赖（`config.py`、`utils_io.py`、`modules/`）。

## 关系处理流程

```text
relation_rule     几何规则 + Object-Aware 候选（低置信度）
       ↓
relation_llm      VL 模型验证候选（Pair Storyboard / 全景 Storyboard）
       ↓
relation_merge    去重 + 耦合补全（left↔right 等）
       ↓
global_relation   跨窗口动态关系聚合
       ↓
relation_verify   互斥消解 + 类别约束过滤 + 最低置信度过滤 + 强模型复核
       ↓
export            导出最终 JSON
```

## 可恢复性

每个节点写 `status.json`（含 `input_hash`），`--resume` 时只重跑 hash 变化或未成功的节点。`--from_node`/`--to_node` 支持局部重跑。

## 验证

```bash
python -m compileall -q src scripts tests
python -m unittest discover -s tests
```
