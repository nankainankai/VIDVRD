# 配置文件说明

| 文件 | 用途 |
|------|------|
| `default.json` | 正式默认：Rex-Omni 检测 + 旧 Step2 追踪 + 可开 VL |
| `dry_run.json` | 无 API/GPU：mock 检测追踪 + 关系 dry_run |
| `run_with_api.json` | mock 检测追踪 + 真实 DashScope 关系 LLM |
| `run_with_vl.json` | mock 检测追踪 + 全链路真实 VL（screen / track_qc / global / relation_llm 传拼图） |
| `production.json` | 在 default 上覆盖：正式关系 + 开启评测 |
| `production_full.json` | 在 default 上覆盖：DINO-X 检测 + 评测（需 `DINOX_API_TOKEN`） |
| `rexomni_full.json` | Rex-Omni-AWQ + legacy 追踪 + 真 VL（`max_windows: 2`）；真模型验收推荐 |
| `rexomni_full_kf2.json` | 同上，`keyframe_interval: 2`（更密、更慢） |
| `dry_run_eval.json` | mock + 开启 Presence 评测（Gold 样例链路） |
| `local_rex.example.json` | Rex 模型路径示例（复制为 `local_rex.json` 覆盖本机路径，该文件勿提交） |

合并规则：`load_config` 以 `default.json` 为底，用户 JSON 深度覆盖。

**Rex-Omni 模型路径**（默认 `D:/Rex-Omni-AWQ`，在仓库外）：

- 配置项：`detector.rex_model_path`（`default.json` 与各 `rexomni_*.json` 已指向 D 盘）
- 环境变量：`REXOMNI_MODEL_PATH`（若设置，覆盖 JSON 中的路径）
- 本机专用：复制 `local_rex.example.json` → `configs/local_rex.json`，运行 `--config configs/local_rex.json`
