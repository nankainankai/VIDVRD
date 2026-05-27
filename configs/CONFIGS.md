# 配置文件说明

| 文件 | 用途 |
|------|------|
| `default.json` | 正式默认：Rex-Omni 检测 + 旧 Step2 追踪 + 可开 VL |
| `dry_run.json` | 无 API/GPU：mock 检测追踪 + 关系 dry_run |
| `run_with_api.json` | mock 检测追踪 + 真实 DashScope 关系 LLM |
| `run_with_vl.json` | mock 检测追踪 + 全链路真实 VL（screen / track_qc / global / relation_llm 传拼图） |
| `production.json` | 在 default 上覆盖：正式关系 + 开启评测 |
| `production_full.json` | 在 default 上覆盖：DINO-X 检测 + 评测（需 `DINOX_API_TOKEN`） |

合并规则：`load_config` 以 `default.json` 为底，用户 JSON 深度覆盖。
