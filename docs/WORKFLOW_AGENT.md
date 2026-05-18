# Agent 工作流说明

本文说明 OpenClaw/Agent 如何运行本项目。核心原则：Agent 负责调度和诊断，业务逻辑在 `src/vidvrd_auto/` 中实现。

## 标准流程

1. 确认使用 `vidvrd` conda 环境。
2. 检查输入是单个视频、URL，或视频列表文本。
3. 检查配置文件，默认使用 `configs/default.json`。
4. 先执行 dry-run 或局部节点验证。
5. 正式调用 `vidvrd_auto.cli`。
6. 读取 `runs/<run_id>/run_manifest.json`。
7. 如果有失败视频，查看对应节点的 `status.json` 和 `run.log`。
8. 修复环境、配置或数据问题后，用同一命令加 `--resume` 恢复。

## 推荐命令

```bash
conda run -n vidvrd python -m vidvrd_auto.cli --videos data/videos.txt --run_dir runs/exp001 --config configs/default.json --resume --api_key YOUR_DASHSCOPE_KEY
```

无模型 dry-run：

```bash
conda run -n vidvrd python -m vidvrd_auto.cli --videos data/videos.txt --run_dir runs/debug001 --config configs/dry_run.json --resume --dry_run_relations --skip_eval
```

局部恢复：

```bash
conda run -n vidvrd python -m vidvrd_auto.cli --videos data/videos.txt --run_dir runs/exp001 --config configs/default.json --resume --from_node track_qc --to_node export
```

## 常见失败诊断

- 缺少 `DASHSCOPE_API_KEY`：非 dry-run 的 VL 节点会失败。设置环境变量或通过 `--api_key` 传入。
- Rex-Omni 未部署：切换到 DINO-X 后端，或在配置中设置 Rex-Omni 模型路径和依赖。
- DINO-X token 缺失：设置 `DINOX_API_TOKEN`。
- 视频下载失败：检查 URL、网络和 `video_ingest.download_timeout_sec`。
- 节点输出缺失：查看该节点目录下的 `run.log`，再用 `--force --from_node <node>` 重跑。

## 汇报模板

Agent 完成后应汇报：

- 运行目录：`runs/<run_id>/`
- 成功和失败视频数量。
- 最终关系文件：`pred/relations_pred.json`
- 每个失败节点及原因。
- 如果运行评测，报告 `reports/presence_report.md`。
- 如果触发强模型复核，报告复核数量和主要动作。
