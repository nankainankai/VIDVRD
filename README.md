# VIDVRD Auto

端到端视频关系检测流水线：动态开放词汇 → Rex-Omni 稀疏检测 → 官方 OC-SORT 轨迹传播 → 窗口级关系识别 → 轨迹对齐评测。

项目内置完整 ImageNet-VidVRD 本体：35 个对象类别、132 个谓词及 base/novel 划分。`stand_left`、`walk_behind` 等组合谓词保留为官方原子标签，同时记录 action/spatial/comparative 组成，便于分析但不改变官方标签。

## 流程

```text
ingest -> vocabulary -> detect -> track -> track_qc
       -> rule + semantic -> merge -> global -> verify -> export -> evaluate
```

- `vocabulary`：固定模式使用完整 35 类；开放模式由云视觉模型发现视频特有对象，并与 35 类取并集。
- `detect`：Rex-Omni 默认每 5 帧检测一次，场景突变可提前触发。
- `track`：未经修改的官方 OC-SORT 核心负责状态传播；适配器负责类别隔离、全局 ID 和项目字段。短缺口在前后真实观测之间线性插值，纯预测框保留来源标记。
- `semantic`：以 30 帧滑动窗口、15 帧步长处理轨迹对拼图；不使用对象对谓词白名单，候选覆盖官方 132 类。
- `verify`：仅对低置信度、互斥冲突或风险轨迹关系做带图复核，动作通过稳定 `relation_id` 应用。
- `evaluate`：先以类别一致和轨迹 vIoU 做匈牙利对齐，再评估关系时间 IoU、P/R/F1、micro AP、逐谓词 mAP、Recall@50/100、tagging P@K 和 base/novel。

没有视频级前筛，也没有音频先验。5 帧检测是检测调度，和前筛不是同一层逻辑。

## 安装与运行

```powershell
conda activate vidvrd
pip install -e .
```

本地 Rex-Omni 模型默认放在 `models/Rex-Omni-AWQ`。

先执行零云调用验证：

```powershell
vidvrd-auto --video path/to/video.mp4 --run-dir runs/smoke --config configs/dry_run.json --dry-run --skip-eval
```

开放词汇生产运行：

```powershell
$env:DASHSCOPE_API_KEY = "your-key"
vidvrd-auto --video path/to/video.mp4 --run-dir runs/exp001 --config configs/production.json
```

固定官方词表并在 Gold 上评测：

```powershell
$env:DASHSCOPE_API_KEY = "your-key"
vidvrd-auto --videos videos.txt --run-dir runs/benchmark --config configs/benchmark.json
```

中断后可在相同命令中增加 `--resume`。输入、配置或上游产物哈希变化时，对应阶段会自动失效。

## Gold

仓库中的 Gold 由本地 50 份官方格式标注生成：

```powershell
vidvrd-build-gold --annotations sample_folder/sample_vidvrd/anno --out-dir gold
```

评测阶段只在最终预测导出后读取 Gold；词表发现、检测、跟踪和关系推理均不读取 Gold，避免标签泄漏。

## 主要输出

- `runs/<run>/pred/relations.json`：全视频关系预测。
- `runs/<run>/pred/trajectories.json`：全视频预测轨迹。
- `runs/<run>/reports/vidvrd.md`：评测报告。
- `runs/<run>/reports/metrics.json`：机器可读评测指标。
- `runs/<run>/run_manifest.json`：阶段状态、配置哈希和产物清单。

详细设计见 [架构](docs/ARCHITECTURE.md)、[数据格式](docs/SCHEMAS.md) 和 [Gold 说明](gold/README.md)。

## 验证

```powershell
python -m compileall -q src tests
python -m unittest discover -s tests -v
```
