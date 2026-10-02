# 精简测试结果快照

本目录保存本次代码分析和真实测试的可核对结果。结果数据约 1.67 MiB；不包含视频、图片、模型、API 密钥、响应缓存或完整执行日志。JSON 仅压缩空白和去掉本机工作区绝对路径前缀，未修改预测或指标。`snapshot_manifest.json` 保存原文件与快照文件的 SHA256，可区分两种内容。

## 主要结果

| 范围 | 官方协议兼容 mAP | R@50 | 已标注类别关系标签覆盖率 @10 |
| --- | ---: | ---: | ---: |
| 新增 3 个固定测试视频 | 0% | 0% | 2/21 = 9.52% |
| 连同此前共 5 个测试视频 | 1.33% | 3.45% | 5/27 = 18.52% |

追加三个视频的流程耗时 1267 秒，8 次 API 请求全部成功，共 86114 token；只有两个视频同时导出非空轨迹和关系。流程完成、产生预测和预测正确是三个不同概念，实际账单费用未知。

这些结果不是完整 200 视频官方测试。五视频来自两次运行，前两个复用历史检测、后三个重新检测；不能将其视为统一耗时实验或算法提升对照。Presence 默认使用未完整标注的 Gold，精确率和 F1 仅供诊断。

## 目录与评测分母

- `additional_test3/reports/requested/`：预先选择的追加三个视频，包含空输出视频，是本轮追加结果的分母。
- `additional_test3/reports/`：本地完整 Gold test split 的十视频报告，其余七个未运行视频按无预测计入；另有请求子集标签报告。不要混用两种分母。
- `additional_test3/videos/`：三个视频的原始检测、跟踪和轨迹质检记录，可核对类别变化、轨迹覆盖和空输出。
- `combined_test5/pred/` 与 `reports/`：五视频合并预测、评测与来源记录。
- `qwen38_flash_smoke/reports/completion_eval/`：最早两个测试样例的历史评测。
- `qwen38_axis_baseline/`、`qwen38_axis_reserve/`、`qwen38_compositional/` 与 `optimization_comparison.json`：两个训练开发视频上的真实控制变量实验，没有稳定的综合提升，实验策略未替换默认配置。

详细解释见 [追加测试分析](../../../docs/additional_test3_analysis.md)、[结项评测与优化](../../../docs/evaluation_for_completion.md) 和 [此前验证记录](../../../docs/performance_validation_20261002.md)。快照内引用的 `runs/` 位置对应原始本地产物；精简快照不包含恢复完整端到端执行所需的全部中间文件。

## 复核已保存的五视频预测

在仓库根目录安装项目依赖后执行，离线评测不会调用 API：

```powershell
python -m vidvrd_auto.evaluation.cli --pred-relations experiments/results/performance_validation_20261002/combined_test5/pred/relations.json --pred-trajectories experiments/results/performance_validation_20261002/combined_test5/pred/trajectories.json --video-ids ILSVRC2015_train_00005012,ILSVRC2015_train_00010002,ILSVRC2015_train_00091004,ILSVRC2015_train_00165004,ILSVRC2015_train_00234014 --scope requested --split test --out-dir runs/result_snapshot_check
```

上述命令使用仓库默认 `gold/` 文件，显式选择这五个视频，避免混用完整 split 分母。完整原始 `runs/` 尚在本地时，可运行 `python experiments/export_result_snapshot.py` 重新导出允许清单中的文件；手写的本 README 保留。快照哈希按 UTF-8、LF 换行计算；Windows 检出时若 Git 转为 CRLF，复核快照哈希前需恢复 LF，原始文件哈希则按原始字节计算。
