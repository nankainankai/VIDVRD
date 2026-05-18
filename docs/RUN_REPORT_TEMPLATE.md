# 运行报告模板

## 基本信息

- 运行目录：
- 配置文件：
- 输入视频数量：
- 成功视频数量：
- 失败视频数量：
- 最终关系文件：
- 评测报告：

## 节点状态

| 节点 | 成功 | 失败 | 跳过 | 主要原因 |
|---|---:|---:|---:|---|
| video_ingest |  |  |  |  |
| audio_prior |  |  |  |  |
| step1_detect |  |  |  |  |
| keyframe_screen |  |  |  |  |
| step2_track |  |  |  |  |
| track_qc |  |  |  |  |
| relation_rule |  |  |  |  |
| relation_llm |  |  |  |  |
| relation_merge |  |  |  |  |
| global_relation |  |  |  |  |
| relation_verify |  |  |  |  |
| export |  |  |  |  |

## 常见问题

- API key 缺失：
- Rex-Omni/DINO-X 环境问题：
- 视频下载失败：
- 模型输出解析失败：
- 轨迹/关系质检高风险：

## 下一步建议

- 是否需要 `--resume`：
- 是否需要从某个节点 `--force --from_node` 重跑：
- 是否需要人工抽检：
