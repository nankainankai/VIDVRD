# 架构

## 分层

`pipeline` 只编排阶段、状态、缓存和文件契约；`nodes` 提供薄入口；算法分别位于 `detection`、`tracking`、`relations`、`evaluation`。所有 DashScope 调用集中通过 `providers`，算法模块不直接读取环境变量。

```text
configs/vidvrd_ontology.json     官方 35/132 本体与 base/novel 划分
src/vidvrd_auto/core/            配置、路径、数据契约、本体访问
src/vidvrd_auto/pipeline/        单视频和多视频编排
src/vidvrd_auto/nodes/           阶段接口
src/vidvrd_auto/detection/       Rex-Omni 主检测器、DINO-X 实验组件与时序调度
src/vidvrd_auto/tracking/        OC-SORT 工程适配、实验跟踪模块与窗口生成
src/vidvrd_auto/relations/       几何、语义、聚合与复核
src/vidvrd_auto/evaluation/      Gold 转换、轨迹对齐和关系指标
```

算法来源、偏差和目标模式统一登记在 [algorithm_registry.md](algorithm_registry.md)。

## 两条运行路线

- `reference_dense`：逐帧检测，每个视频帧推进一次 OC-SORT，关闭 Agent 关系调用；只用于算法参照。
- `main`：每 3 帧固定调用 Rex-Omni；只在这些检测锚点推进同一 OC-SORT 核心，并在相邻真实观测间补齐短缺口；随后调用 Agent 判断语义关系。

稀疏检测和强模型复核都是 `main` 的配置，不再单独定义运行模式。正式路线固定使用 VidVRD 35 类对象与 132 个谓词；旧版配置只保存在 `configs/archive/`，不属于可运行路线。

`configs/base.json` 是 loader 的内部合并基座，不直接代表运行路线；CLI 默认使用 `configs/main.json`。`configs/config.json` 保留为与 main 合并结果相同的兼容入口，避免同一 `run_mode` 出现两套实际默认。

## 开放词汇检测

正式模式把官方 35 类交给 Rex-Omni。DINO-X 只在独立实验配置中处理同一份 35 类词表。实验性的开放发现代码仍可从均匀采样帧生成视频级拼图并追加对象名，但当前 `main` 不启用它，避免无关背景类别污染轨迹和官方评测。

开放发现是检测之前的词表构建，不是视频前筛：它不拒绝视频，也不决定某帧是否检测。

## 稀疏检测与连续轨迹

视频逐帧解码，正式路线每 3 帧产生一个 Rex-Omni 检测锚点。未检测帧仍写入 JSONL 并明确标记 `skipped`。混合实验配置才会在第 0、15、30…帧用 DINO-X 替换 Rex，并在 DINO-X 单帧失败时回退 Rex。

两条路线使用同一固定提交、未修改的 OC-SORT 核心。`reference_dense` 每个视频帧调用一次；`main` 只在 Rex 检测锚点调用一次，避免稀疏检测之间的空帧重置 OC-SORT 的确认连续性。正式闭集路线按官方对象类别隔离关联，防止人和自行车等高度重叠但类别不同的目标合并；镜头切换时清空活动运动状态，但全局 ID 不复用。项目适配器负责格式转换、类别记录和帧号映射，随后只在两个真实观测之间做不超过 6 帧的线性补全。主路线不使用外观关联或离线 tracklet 拼接。

MASA 外观、hybrid 关联和离线 stitching 源码保留在 `tracking/appearance/`、`tracking/hybrid/` 与 `tracking/stitching.py`，但没有配置入口，也不会被生产流水线导入。

## 窗口级关系

关系以 30 帧窗口、15 帧步长为基本单元：

```text
轨迹框 -> 加权几何投票 --------┐
轨迹证据 -> 分层候选 -> 事件帧双视图 -> 视觉语义判断 ├-> 合并 -> 证据连续性聚合 -> 风险复核
                              ┘
```

几何投票对 observed/interpolated/predicted 分别赋权，并要求至少一帧成对真实观测；输出的是连续证据片段及 `rule_support`，不是整窗口概率。语义节点默认覆盖全部共同可见目标对。每个方向计算归一化距离、边缘间隔、IoU、接近速率、双目标速度、共运动、尺寸比和身份支持等轨迹证据，再按 action/spatial/comparative 根规则映射到可重叠的谓词族。初始候选不超过 14 个，保留易混淆谓词用于对比，不使用类别硬删除。二维可确定的五个几何谓词仍由规则层处理，`front/behind` 和组合谓词保留在语义层。跨窗口阶段按 `(subject, predicate, object)` 聚合区间与证据均连续的片段并生成稳定关系 ID。若人为配置目标对上限，未处理目标对会明确进入 `deferred_pairs`。

语义调用使用 `bounded_batched_agent_v3`。程序优先选择距离最近、IoU 最大和接近变化最强的事件帧，并取最多 5 帧连续局部片段；每帧同时展示带 ID 框的完整场景和高分辨率对象对近景。每个窗口仍生成完整 `EvidencePacket`，包含候选策略、候选谓词族、轨迹特征和证据模式。同一无序对象对的连续窗口按时间排序，默认每 6 个 packet 连同原故事板合并为一次请求；模型按 `packet_id` 分别返回结果，每个结果仍由原 validator 独立检查，因此批处理不放宽谓词、区间或证据约束。Agent 可补看最多 4 个共同可见帧或扩展一个邻接谓词族；同一批次产生的补充 packet 也合并调用，补充后不得再次请求。完整 packet、批次和调用审计保存在 semantic 目录。

## 复核

复核只处理低排序分、互斥冲突和风险轨迹相关项目，并只附带相应轨迹对的真实拼图。模型只能建议 `accept_relation/reject_relation/change_predicate/refine_interval/defer_for_review`；关系 ID、官方谓词、证据帧和只能收窄的时间范围验证通过后才可能修改输出。`main` 保存每次修改的 before/after 快照；非法响应保持原关系不变。规则化互斥消解只有在显式允许动作时执行，反向耦合默认关闭。

## Gold 与评测隔离

Gold 转换器把官方半开区间 `[begin_fid, end_fid)` 转成项目闭区间 `[start_frame, end_frame]`。所有关系 JSON 写盘时显式标记 `span_convention=inclusive`；规范化 Python schema 内部使用 half-open。官方兼容评测适配器按字段读取两种约定，再转换成类别三元组、半开时间区间和连续 subject/object tubes，直接按两条 tube vIoU 的较小值做贪心匹配，不依赖项目轨迹 ID。原有全轨迹匈牙利映射逻辑位于 `evaluation/diagnostic/`，只输出带 `diagnostic_` 前缀的内部指标。

Gold 路径只传给流水线末尾的 `evaluate`，不传给任何预测节点。`benchmark_official.json` 只选择 Gold manifest 中的测试 split；当前本地 Gold 只有 10 个测试视频，因此报告会明确标为 protocol-compatible partial result。完整论文指标仍要求官方 200 个测试视频。失败视频在提交文件中保留空列表并继续进入分母。

## 状态与恢复

每个阶段保存输入哈希、配置哈希、产物列表和 `status.json`。本地输入视频使用完整文件 SHA-256 参与 ingest 输入哈希，同一路径内容变化会使 `--resume` 失效。`--resume` 只复用输入未变且产物完整的成功阶段；`--force` 显式重跑。流水线不会自动删除既有 `runs/`、`models/` 或输入数据。
