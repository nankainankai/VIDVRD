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
- `track`：主路线和参照路线都使用仓库内冻结的 OC-SORT 核心。主路线只在 Rex 检测锚点调用 OC-SORT，并在相邻真实观测之间补齐短缺口；项目代码负责调度、字段转换、类别记录和产物导出。导出前要求至少两次真实观测，过滤 OC-SORT 新类别启动宽限期产生的单帧伪轨迹，不修改官方核心。
- `semantic`：先用轨迹证据把 132 个官方谓词按“6 原子 + 4 复合 + 2 易混淆 + 2 补充”路由为每方向最多 14 个候选，避免动作＋空间复合谓词被前三关系族系统性裁掉；再围绕事件帧生成“全景 + 对象对近景”的连续帧证据。当前 Kimi 路线每张故事板单独请求，Agent 最多触发一次补帧或邻接谓词族扩展。成功请求按模型、提示词和图片哈希缓存。
- `verify`：仅对低排序分、互斥冲突或风险轨迹关系做带图复核；动作经过关系 ID、谓词、证据和区间校验后才应用。
- `score`：规则支持率和 Agent 分数不再直接比较。最终排序分显式组合证据分、时序覆盖质量、轨迹对质量和来源系数；跨窗口使用证据加权平均，不取片段最大值。来源系数是待开发集校准的工程初值，不是论文结论。
- `evaluate`：官方兼容层直接按类别三元组与关系 tube vIoU 计算逐视频 mAP、Recall@50/100 和 tagging P@1/5/10；原有全轨迹匈牙利对齐只保留为单独的内部诊断。

没有视频级前筛，也没有音频先验。稀疏检测调度和前筛不是同一层逻辑。

## 两条路线

- `configs/reference_dense.json`：逐帧检测、每帧推进 OC-SORT、关闭 Agent 语义调用，只用于算法参照。
- `configs/main.json`：每 3 帧使用 Rex-Omni 检测，随后推进同一 OC-SORT 核心并做短缺口插值；这是正式路线。调轨阶段可加 `--tracking-only` 在轨迹质检后停止。
- `configs/smoke_kimi.json`：保留与正式路线相同的本地检测/跟踪配置，但云端只处理首窗口的一组对象对，不重试、不补帧、不复核，用于低额度集成烟测。
- `configs/experimental_hybrid_dinox.json`：每 15 帧用 DINO-X 替换当帧 Rex，其余锚点仍用 Rex；仅用于对照实验。
- `configs/acceptance_dachuang.json`：大创验收路线，显式启用 DINO-X + Rex、OC-SORT、关系 Agent、视觉复核和 Gold 评测；需要同时配置 `DDS_API_TOKEN` 与 `DASHSCOPE_API_KEY`。
- `configs/ablation_no_agent_review.json`：关闭最终 Agent 复核的消融配置，用于和主路线比较。

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

小额度验证可先用 `configs/main.json --tracking-only --resume` 完成本地检测与跟踪，再在**同一** `--run-dir` 下运行 `configs/smoke_kimi.json --resume --skip-eval`。缓存会跳过相同的本地阶段，云端最多请求一次。`--dry-run` 可预先确认是否选中了对象对；没有对象对时不会请求 API。

正式闭集运行：

```powershell
$env:DASHSCOPE_API_KEY = "your-key"
vidvrd-auto --video path/to/video.mp4 --run-dir runs/exp001 --config configs/main.json
```

主链路不需要 DINO-X Token。若运行混合检测对照，使用 `configs/experimental_hybrid_dinox.json`，并在项目根目录 `.env` 中配置 `DDS_API_TOKEN`。密钥不会写入运行产物。

固定官方词表并在 Gold 上评测：

```powershell
$env:DASHSCOPE_API_KEY = "your-key"
vidvrd-auto --videos videos.txt --run-dir runs/benchmark --config configs/benchmark_official.json
```

中断后可在相同命令中增加 `--resume`。输入、配置或上游产物哈希变化时，对应阶段会自动失效。
语义请求失败或返回缺失的批次结果时，该阶段及整次命令会标记失败并返回非零退出码；细节记录在对应视频的 `semantic/run.log`，不会把空语义结果伪装成成功。

## Gold

仓库中的 Gold 由本地 50 份官方格式标注生成：

```powershell
vidvrd-build-gold --annotations sample_folder/sample_vidvrd/anno --out-dir gold
```

词表发现、检测和跟踪不读取 Gold。关系候选路由可读取由 **30 个训练视频**离线统计的谓词先验（`experiments/predicate_priors_train30.json`）；固定 10 个开发视频和 10 个测试视频均不参与先验拟合，推理时不读取其标注。`experiments/build_candidate_priors.py` 可重建先验，`experiments/candidate_oracle.py` 输出仅使用 Gold 轨迹的候选上限诊断，**不是**真实管线 mAP。

本地 50 视频子集可先生成两个固定视频路径清单：

```powershell
python experiments/prepare_benchmark.py --dataset-videos 'D:\VIDVRD-share-no-model-weights-20260922\sample_folder\sample_vidvrd\videos'
```

开发集运行 `--videos experiments/dev_video_paths.txt --config configs/benchmark_dev.json`；最终测试集运行 `--videos experiments/test_video_paths.txt --config configs/benchmark_official.json`。两次运行须使用不同 `--run-dir`。只有测试集完整跑完时，才能把该 10 视频子集的 mAP 与 Recall 作为子集结果；它仍不是官方 200 视频完整测试集成绩。

## 主要输出

- `runs/<run>/pred/relations.json`：全视频关系预测。
- `runs/<run>/pred/trajectories.json`：全视频预测轨迹。
- `runs/<run>/reports/official_vidvrd.md|json`：官方协议兼容报告及指标。
- `runs/<run>/reports/diagnostic_track_aligned.md|json`：项目内部轨迹对齐诊断。
- `runs/<run>/run_manifest.json`：阶段状态、配置哈希和产物清单。
- `runs/<run>/visualization/*.mp4`：可选的轨迹框、ID、关系箭头和活动三元组叠加视频。

已有预测可以脱离模型 API 重新评测，便于调整评测机制和复现实验：

```powershell
vidvrd-evaluate --pred-relations runs/exp001/pred/relations.json `
  --pred-trajectories runs/exp001/pred/trajectories.json `
  --out-dir runs/exp001/reports --scope gold_split
```

导出验收演示视频：

```powershell
vidvrd-render --video path/to/video.mp4 `
  --relations runs/exp001/pred/relations.json `
  --trajectories runs/exp001/pred/trajectories.json `
  --output runs/exp001/visualization/overlay.mp4
```

详细设计见 [架构](docs/ARCHITECTURE.md)、[数据格式](docs/SCHEMAS.md) 和 [Gold 说明](gold/README.md)。MASA/hybrid 仅作为未接入主流程的实验模块保存在 [实验模块说明](docs/MASA_SETUP.md)。算法来源与工程适配见 [算法登记](docs/algorithm_registry.md)。
大创任务书与当前实现的逐项对应、验收命令和仍需补充的数据见 [大创验收说明](docs/大创验收说明.md)。

## 验证

```powershell
python -m compileall -q src tests
python -m unittest discover -s tests -v
```
