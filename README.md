# VIDVRD Auto

端到端视频关系检测流水线：VidVRD 35 类对象 → Rex-Omni 稀疏检测 → OC-SORT 轨迹关联 → 132 类窗口关系识别 → 官方兼容评测 + 轨迹对齐诊断。

项目内置完整 ImageNet-VidVRD 本体：35 个对象类别、132 个谓词及 base/novel 划分。`stand_left`、`walk_behind` 等组合谓词保留为官方原子标签，同时记录 action/spatial/comparative 组成，便于分析但不改变官方标签。

## 流程

```text
ingest -> vocabulary -> detect -> track -> track_qc
       -> rule + semantic -> merge -> global -> verify -> export -> evaluate
```

- `vocabulary`：正式路线固定使用完整 35 类；开放对象发现代码保留，但默认不启用。
- `detect`：正式路线固定每 3 帧调用一次 Rex-Omni。DINO-X + Rex 混合检测作为独立实验组件保留，不在主链路自动启用。
- `track`：主路线和参照路线都使用仓库内冻结的 OC-SORT 核心。主路线只在 Rex 检测锚点调用 OC-SORT，并在相邻真实观测之间补齐短缺口；项目代码只负责调度、字段转换、类别记录和产物导出。
- `semantic`：先用轨迹证据把 132 个官方谓词路由为每方向最多 24 个对比候选，给原子、组合和易混谓词预留配额。故事板先保留窗口头、中、尾，再补事件帧。同一对象对的最多 6 个连续窗口合并为一次千问请求，每个窗口独立输出和校验；Agent 最多触发一次补帧或邻接谓词族扩展。
- `verify`：仅对低排序分、互斥冲突或风险轨迹关系做带图复核；动作经过关系 ID、谓词、证据和区间校验后才应用。
- `evaluate`：官方兼容层直接按类别三元组与关系 tube vIoU 计算逐视频 mAP、Recall@50/100 和 tagging P@1/5/10；原有全轨迹匈牙利对齐只保留为单独的内部诊断。

没有视频级前筛，也没有音频先验。稀疏检测调度和前筛不是同一层逻辑。

## 两条路线

- `configs/reference_dense.json`：逐帧检测、每帧推进 OC-SORT、关闭 Agent 语义调用，只用于算法参照。
- `configs/main.json`：每 3 帧使用 Rex-Omni 检测，随后推进同一 OC-SORT 核心并做短缺口插值；这是正式路线。调轨阶段可加 `--tracking-only` 在轨迹质检后停止。
- `configs/experimental_hybrid_dinox.json`：每 15 帧用 DINO-X 替换当帧 Rex，其余锚点仍用 Rex；仅用于对照实验。

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

正式闭集运行：

```powershell
$env:DASHSCOPE_API_KEY = "your-key"
vidvrd-auto --video path/to/video.mp4 --run-dir runs/exp001 --config configs/main.json
```

主链路不需要 DINO-X Token。若运行混合检测对照，使用 `configs/experimental_hybrid_dinox.json`，并在项目根目录 `.env` 中配置 `DDS_API_TOKEN`。密钥不会写入运行产物。

当前关系模型为 `qwen3.8-flash`，通过 DashScope 普通多模态接口调用，设置 `relations.enable_thinking=false` 以降低等待时间与输出 token 开销，视觉复核仍使用 `qwen-vl-max`。API key 从环境变量或项目 `.env` 加载。思考模式会影响预测结果，语义缓存按该参数分别保存。

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
- `runs/<run>/reports/presence_instance.json`：固定片段内按对象实例对计算的 Presence P/R/F1；现有稀疏 Gold 默认仅作诊断。
- `runs/<run>/reports/known_label_coverage.md|json`：每视频最多 10 个类别关系标签的已标注标签覆盖率；不要求实例和时空定位准确，不能当作 mAP 或完整标注准确率。
- `runs/<run>/reports/runtime.json`：完成率、API 调用、重试、服务端 token 使用量及耗时；实际账单费用未提供时为 null。
- `runs/<run>/run_manifest.json`：阶段状态、配置哈希和产物清单。

详细设计见 [架构](docs/ARCHITECTURE.md)、[数据格式](docs/SCHEMAS.md) 和 [Gold 说明](gold/README.md)。MASA/hybrid 仅作为未接入主流程的实验模块保存在 [实验模块说明](docs/MASA_SETUP.md)。算法来源与工程适配见 [算法登记](docs/algorithm_registry.md)。

2026-10-02 的修正与验证见 [性能回归记录](docs/performance_validation_20261002.md)。主配置将跨窗口合并限制为 30 帧，保留窗口级标注边界；设 `global_relation.max_merged_span_frames=0` 可恢复连续事件聚合。此限制只控制合并，不切割已有模型输出。API 配额、权限或复核响应异常会保留失败产物并返回非零退出码。

已有预测可离线评测，无需再次调用模型：

```powershell
python -m vidvrd_auto.evaluation.cli --pred-relations runs/exp001/pred/relations.json --pred-trajectories runs/exp001/pred/trajectories.json --out-dir runs/exp001/reports --scope requested --split train
```

固定开发/测试清单及消融命令见 [实验说明](experiments/README.md)。正式 Presence 评测需要独立完整标注的固定片段清单；使用 `--presence-clips clips.json`，不能只换一个宽松指标后将分数解释成泛化提升。

面向辅助标注的固定标签预算指标、真实对照结果和独立片段抽样见 [结项评测与优化](docs/evaluation_for_completion.md)。当前新增策略保留为实验方案，默认配置未替换。

增加三个完整测试视频后的成绩、类别与关系漏项、五视频汇总及耗时见 [追加测试分析](docs/additional_test3_analysis.md)。运行完成率与轨迹/关系非空输出率分别记录，子集标签报告不会替代完整 Gold 范围报告。

## 验证

```powershell
python -m compileall -q src tests
python -m unittest discover -s tests -v
```
