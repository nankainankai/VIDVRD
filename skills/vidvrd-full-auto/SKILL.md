---
name: vidvrd-full-auto
description: 运行、恢复和诊断 VIDVRD 全自动视频关系检测流水线。
---

# VIDVRD 全自动流水线

## 前置条件

1. 使用 `vidvrd` Conda 环境并执行 `pip install -e .`。
2. Rex-Omni 默认位于 `models/Rex-Omni-AWQ`。
3. `configs/config.json` 是基础配置；`dry_run.json`、`production.json`、`benchmark.json` 是覆盖配置。
4. 生产语义调用需要 `DASHSCOPE_API_KEY`；dry-run 不需要云密钥。

## 强制执行顺序

首次处理新环境或配置时必须先运行：

```powershell
vidvrd-auto --video <video> --run-dir runs/<unique-smoke-run> --config configs/dry_run.json --dry-run --skip-eval
```

完成后检查 `run_manifest.json`、每个阶段的 `status.json`、词表、检测间隔、轨迹 ID、窗口和最终 JSON。确认无误后才允许运行云端生产配置：

```powershell
vidvrd-auto --video <video> --run-dir runs/<run> --config configs/production.json
```

固定官方词表评测使用：

```powershell
vidvrd-auto --videos <list.txt> --run-dir runs/<run> --config configs/benchmark.json
```

中断后在原命令增加 `--resume`；只有确认所有缓存都应失效时才用 `--force`。

## 云调用边界

1. `vocabulary`：仅 `production.json` 开启，每个视频一次开放对象发现。
2. `semantic`：每个有效窗口轨迹对一次视觉关系判断。
3. `verify`：仅存在风险关系和对应拼图时，最多一次批量复核。

Rex-Omni、OC-SORT、几何规则、跨窗口聚合、导出和评测均为本地执行。

## 诊断顺序

1. 检查 `run_manifest.json` 的失败或未完成阶段。
2. 检查对应阶段的 `status.json` 和 `run.log`。
3. 核对模型路径、显存、密钥、视频和配置。
4. 修复后使用 `--resume`，避免重复消耗 GPU 和云额度。

不得删除或覆盖既有 `runs/`、`models/`、`sample_folder/`。验证目录必须使用唯一名称，并且只清理本次明确创建的目录。
