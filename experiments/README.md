# 固定评测与消融

可随代码下载的精简结果见 [测试结果快照](results/performance_validation_20261002/README.md)，包含五视频预测、追加三个视频的检测/轨迹诊断和开发对照；视频、模型及缓存保留在本地。

`dev_videos.txt` 固定 10 个训练集开发视频；`test_videos.txt` 固定本地 Gold 中的 10 个测试视频。文件名含 train 不代表 Gold 的官方 split，应以 `gold/vidvrd_50_manifest.json` 为准。这 10 个测试视频不构成完整 200 视频官方测试集。

```powershell
python experiments/prepare_benchmark.py --dataset-videos sample_folder/sample_vidvrd/videos
python -m vidvrd_auto.cli --videos runs/benchmark_inputs/dev_video_paths.txt --run-dir runs/dev --config configs/benchmark_dev.json
python -m vidvrd_auto.cli --videos runs/benchmark_inputs/test_video_paths.txt --run-dir runs/test --config configs/benchmark_test.json
```

测试前冻结候选、阈值、提示词和预算；测试标签只进入最终评测。主配置不读取训练先验。保留的 `predicate_priors_train30.json` 仅作可选实验，不能把它在已知类别配对上的提升直接视为未见数据泛化。

## 冻结检测回归

```powershell
python experiments/replay_detections.py --source-run runs/closed35_132_00119033_20260829 --source-run runs/smoke_ocsort_batched_00177003_20260828 --run-dir runs/replay --config configs/main.json
```

只复制检测产物，重新运行跟踪和全部关系阶段。运行清单记录原始检测 SHA256。检测质量、GPU 检测耗时及检测成本不在此实验的验证范围内。增加 `--resume` 可复用相同配置的成功阶段；失败阶段会重试，语义缓存按模型、提示词和图片内容匹配。

## 离线控制变量

```powershell
python experiments/evaluate_ablation.py --run runs/replay --out-dir runs/no_review --max-merged-span 0
python experiments/evaluate_ablation.py --run runs/replay --out-dir runs/bounded_no_review --max-merged-span 30
python experiments/evaluate_ablation.py --run runs/replay --out-dir runs/rule_only --rule-only --max-merged-span 30
```

这些实验使用同一批轨迹和已有窗口输出，额外 API 调用为零。`no_review` 只关闭最后的视觉复核，仍包含 VLM 和可能的补帧；不能命名为“无 Agent”。`rule_only` 不使用语义关系输出。若比较补帧的收益，应另外固定首次 VLM 输出，单独禁用补帧，分别报告调用次数及质量。

## 大创结项指标

主结果采用独立完整标注的固定短片段，报告实例对 Presence P/R/F1、人工修改耗时与可直接使用标注比例。辅助报告官方协议兼容 mAP/R@50、边界误差、Gold 帧框 IoU/IDSW、每视频分钟耗时、API 次数、token、实际费用及完成率。至少重复 3 次真实调用验证稳定性；缓存恢复测试单独报告。

`--presence-clips` 接受 JSON 列表，每项如下，时间为左闭右开帧区间：

```json
[{"video_id":"video_001","start_frame":0,"end_frame":30,"annotation_complete":true}]
```

必须先人工完整标注该片段并提供对应 Gold 轨迹和关系，才能将 `annotation_complete` 设为 true。默认自动选现有 Gold 的关系区间，始终标记为不完整诊断；未标关系可能实际成立，其 FP、precision/F1 不能直接作为标注质量结论。Presence 将同片段内同一实例对、同一谓词去重，任意非空时间重叠视为出现；它不衡量持续时长是否正确。

`runtime.json.recorded_api` 汇总最新存储的各阶段调用记录；`invocation_api` 只汇总当前实际执行阶段，在旧产物无执行记录时为 null。续跑耗时与首次运行耗时分开，缓存命中不算新 API 请求。缺失 token 或账单不是免费。

## 结项辅助指标与新实验

每次评测自动生成 `known_label_coverage.json` 和中文 Markdown 报告：主辅助指标固定为每视频最多 10 个去重类别三元组的已标注标签覆盖率，另列 @1/5/20。标签必须精确匹配类别、完整谓词和方向，但不衡量实例、时间和轨迹精度，也不能据稀疏 Gold 声称完整准确率。

```powershell
python experiments/evaluate_ablation.py --run runs/replay --out-dir runs/multiscale --max-merged-span 30 --temporal-proposal-mode multiscale
python experiments/evaluate_candidate_budget.py --run runs/replay --out-dir runs/router_axis --candidate-limit 24 --axis-motion-reserve
python experiments/prepare_completion_review.py --videos runs/benchmark_inputs/test_video_paths.txt --out-dir runs/completion_review_v1
```

候选预算实验只允许训练视频；候选覆盖率不等于真实模型召回率。`experimental_axis_motion.json` 与 `experimental_compositional.json` 是完整开发对照配置，均关闭最终复核，不读取类别先验；模型为 qwen3.8-flash。`experimental_multiscale.json` 是基于 base 的可选聚合配置。上述策略仍有指标取舍，尚未替换默认配置。

独立抽样清单只读取视频元数据，片段采用半开帧区间且默认标记未完整标注，必须另行准备完整 Gold。详细数字、限制及方案选择见 [结项评测与优化](../docs/evaluation_for_completion.md)。
