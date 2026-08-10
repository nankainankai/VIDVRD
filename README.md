# VIDVRD Auto

端到端视频关系检测流水线：动态开放词汇 → Rex-Omni 稀疏检测 → MASA 外观与运动联合关联 → 离线 tracklet 拼接 → 窗口级关系识别 → 官方兼容评测 + 轨迹对齐诊断。

项目内置完整 ImageNet-VidVRD 本体：35 个对象类别、132 个谓词及 base/novel 划分。`stand_left`、`walk_behind` 等组合谓词保留为官方原子标签，同时记录 action/spatial/comparative 组成，便于分析但不改变官方标签。

## 流程

```text
ingest -> vocabulary -> detect -> track -> track_qc
       -> rule + semantic -> merge -> global -> verify -> export -> evaluate
```

- `vocabulary`：固定模式使用完整 35 类；开放模式由云视觉模型发现视频特有对象，并与 35 类取并集。
- `detect`：Rex-Omni 默认每 5 帧检测一次，场景突变可提前触发。
- `track`：主路线使用 MASA-R50 外观、真实帧时间运动和软类别联合关联，再离线拼接 tracklet；参考路线仍使用未经修改的官方 OC-SORT。短缺口只在全局 ID 确定后、两个真实观测之间插值。
- `semantic`：先用轨迹证据把 132 个官方谓词路由为每方向最多 14 个对比候选，再围绕接近、重叠等事件帧生成“全景 + 对象对近景”的连续帧证据；Agent 只能返回有限动作，并最多触发一次补帧或邻接谓词族扩展。
- `verify`：仅对低排序分、互斥冲突或风险轨迹关系做带图复核；动作经过关系 ID、谓词、证据和区间校验后才应用。
- `evaluate`：官方兼容层直接按类别三元组与关系 tube vIoU 计算逐视频 mAP、Recall@50/100 和 tagging P@1/5/10；原有全轨迹匈牙利对齐只保留为单独的内部诊断。

没有视频级前筛，也没有音频先验。5 帧检测是检测调度，和前筛不是同一层逻辑。

## 两条路线

- `configs/reference_dense.json`：逐帧检测、每帧推进 OC-SORT、关闭 Agent 语义调用，只用于算法参照。
- `configs/main.json`：稀疏检测、MASA 外观联合关联、离线 tracklet 拼接、有界插值和 Agent 关系判断，是正式路线。

命令行默认直接使用 `configs/main.json`。`configs/base.json` 仅供配置合并，不是第三条可运行路线；`configs/config.json` 是与正式 main 等价的兼容入口。

旧配置位于 `configs/archive/`，只用于追溯，不是第三条路线。

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
vidvrd-auto --video path/to/video.mp4 --run-dir runs/exp001 --config configs/main.json
```

固定官方词表并在 Gold 上评测：

```powershell
$env:DASHSCOPE_API_KEY = "your-key"
vidvrd-auto --videos videos.txt --run-dir runs/benchmark --config configs/benchmark_official.json
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
- `runs/<run>/reports/official_vidvrd.md|json`：官方协议兼容报告及指标。
- `runs/<run>/reports/diagnostic_track_aligned.md|json`：项目内部轨迹对齐诊断。
- `runs/<run>/run_manifest.json`：阶段状态、配置哈希和产物清单。

详细设计见 [架构](docs/ARCHITECTURE.md)、[数据格式](docs/SCHEMAS.md)、[MASA 接入](docs/MASA_SETUP.md) 和 [Gold 说明](gold/README.md)。算法来源、项目修改与已知偏差见 [算法登记](docs/algorithm_registry.md)。

## 验证

```powershell
python -m compileall -q src tests
python -m unittest discover -s tests -v
```
