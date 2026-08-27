# 算法与实现登记

本文件区分上游标准实现、项目适配器和项目启发式。运行产物中的
`run_manifest.json.provenance` 必须能回到本登记中的实现名称。

## 运行模式

| `run_mode` | 定位 | 当前状态 |
|---|---|---|
| `reference_dense` | 逐帧检测与 OC-SORT 参照 | 每个视频帧调用 `update_public()`；关闭 Agent 语义调用 |
| `main` | 项目正式路线 | 稀疏 Rex 检测、锚点帧 OC-SORT、短缺口补全、批量 Agent 关系判断 |

旧版配置保存在 `configs/archive/legacy_agent_v1.json`，只用于追溯，不是第三种模式。

配置真相源：`configs/base.json` 只提供完整合并基座；CLI 默认和正式运行都指向 `configs/main.json`；`configs/config.json` 保留为与 main 等价的兼容入口。运行 manifest 记录合并后的配置哈希和实际采样模式。

## Rex-Omni 检测

- 上游仓库：<https://github.com/IDEA-Research/Rex-Omni>
- 项目入口：`src/vidvrd_auto/detection/rex.py`
- 视频调度：`src/vidvrd_auto/detection/video.py`
- 当前模型配置：`models/Rex-Omni-AWQ`，Transformers backend
- 当前项目修改：开放类别输入、批量锚点帧、最小框面积过滤、类别归一、每 5 帧检测及场景变化触发。
- 当前适配：使用 `temperature=0`、`top_p=0.05`、`top_k=1`；Rex 未提供原生分数时输出 `score=null`，不再伪造概率。

## OC-SORT 跟踪

- 上游仓库：<https://github.com/noahcao/OC_SORT>
- 固定 commit：`8462e7e729a93ccd3bd995c0a79a890336cb3a0b`
- 许可证：MIT
- 冻结源码：`src/vidvrd_auto/tracking/third_party/oc_sort/`
- 来源和文件哈希：`src/vidvrd_auto/tracking/third_party/oc_sort/SOURCE.md`
- 项目适配器：`src/vidvrd_auto/tracking/ocsort/adapter.py`
- 工程适配：检测框裁剪与格式转换、输出 ID 映射、类别投票和轨迹元数据；不修改 `third_party` 中的 OC-SORT 算法源码。
- `ocsort_reference`：`reference_dense` 在每个视频帧调用上游 `update_public()`，只用于逐帧检测参照。
- `sparse_ocsort`：当前 `main`。只在 Rex 检测锚点调用同一上游核心，OC-SORT 时间步为检测锚点；确认输出映射回真实帧号，短缺口补全发生在跟踪器之外。
- Rex 没有分数时，OC-SORT 内部关联统一使用常量权重；该值不写成检测或轨迹置信度。

## 未接入的 MASA/hybrid 实验模块

- 上游仓库：<https://github.com/siyuanliii/masa>
- 核对 commit：`c5472b9c7615f35abdf1188cb1a0c5408fe50d66`
- 许可证：Apache-2.0
- 官方模型：MASA-R50 plug-and-play
- 项目外观适配器：`src/vidvrd_auto/tracking/appearance/masa.py`
- 项目在线关联器：`src/vidvrd_auto/tracking/hybrid/tracker.py`
- 项目离线拼接器：`src/vidvrd_auto/tracking/stitching.py`
- 这组源码不在 `track_video()` 的算法选择集合中，正式配置也没有 `hybrid_sparse_reid` 入口。
- 实验实现原计划只在真实检测锚点提取 MASA embedding，再用运动、外观、IoU 和软类别联合代价做关联。
- 类别不是硬分区。每条 tracklet 保存加权类别分布，类别不一致只产生低权重惩罚。
- 在线编号写为 `local_tracklet_id`；离线同场景 DAG 路径覆盖生成最终 `track_id`，边及各项代价完整落盘。全局 ID 写回之后才允许短缺口插值。
- 该模块借用 MASA 官方外观特征，但其联合代价、稀疏时钟与离线拼接均为项目自有实验代码，不属于当前工程路线，也不作为算法效果结论。
- 无原生检测分数时，传给 MASA 的 `1.0` 仅表示统一关联权重；不得把它写为检测或轨迹置信度。轨迹 `confidence` 在没有原生分数时保持 `null`。

## 关系候选与时间聚合

- 实现位置：`src/vidvrd_auto/relations/`、`src/vidvrd_auto/nodes/global_relation.py`
- 来源：项目自有规则与编排实现，不是某篇论文的原样复现。
- 几何规则：仅处理二维框可判定的左右、上下和邻近关系；按帧记录证据并输出连续片段。`front/behind` 由视觉语义节点处理。
- 候选路由：`predicate_hierarchy.py` 以官方 action/spatial/comparative 分解建立可重叠谓词族和混淆组；`evidence_features.py` 提取有向轨迹证据；`candidate_router.py` 生成最多 14 个初始候选，必要时扩展到最多 24 个。它们是项目自有启发式，不是训练式分类器，分数不是概率。
- 语义关系：事件中心连续帧的“完整场景 + 对象对近景”故事板 + DashScope 多模态模型；输出必须给出区间、展示帧中的证据帧和 `agent_score`。
- 目标对覆盖：默认处理窗口内所有满足共同可见条件的目标对。若显式设置上限，未处理目标对写入 `run.log.pair_coverage.deferred_pairs`。
- 分数：规则输出 `rule_support`，模型输出 `agent_score`；`ranking_score` 只用于排序，不表示校准概率，重复命中不额外加分。
- 时间聚合：仅合并区间连续且证据帧间隔不超过阈值的同三元组；不再把单次窗口阳性默认扩成整个窗口。
- 已知边界：轨迹路由只负责召回与缩小比较范围，不能从框轨迹区分 `touch/bite`；最终语义仍由 Qwen 的局部视觉证据判断。当前没有姿态、手部、嘴部、分割或训练式时间定位器，二维几何规则也不是论文关系模型。
- 研究参考：RePro/OpenVoc-VidVRD、VrdONE、OpenVidVRD、METOR；项目未复现这些方法时，不得使用其方法名描述项目模块。

## Agent 节点

- Provider：`src/vidvrd_auto/providers/dashscope.py`
- Prompt：`src/vidvrd_auto/prompts/templates.py`
- 词表发现：`src/vidvrd_auto/nodes/vocabulary.py`
- 关系判断：`src/vidvrd_auto/relations/semantic.py`
- 关系复核：`src/vidvrd_auto/relations/ops.py`
- Agent 契约：`src/vidvrd_auto/agents/`，登记名 `bounded_batched_agent_v3`。
- 当前性质：EvidencePacket + 有限 AgentAction + 确定性 validator。同一对象对最多 6 个连续窗口共享一次请求，响应仍按 `packet_id` 独立验证。语义节点允许 `request_more_frames` 和一次邻接谓词族扩展，两者共享一次补充调用；补帧只能来自目标对共同可见帧。
- 正式写入：语义接受动作验证后才生成候选关系；最终复核只有在 `apply_actions=true` 且动作通过关系 ID、谓词和区间校验后才修改结果，并保存前后快照。
- 明确不支持：自动重检测、重跟踪、模型升级、多轮循环和 Agent 直接修改轨迹。

## 评测

- 官方兼容实现：`src/vidvrd_auto/evaluation/official/vidvrd.py`，登记名 `imagenet_vidvrd_official_2017_compatible_v1`。
- 对照来源：MIT 许可的 `xdshang/VidVRD-helper`；逐视频 VOC AP、每视频 top-K 后累计 Recall、类别三元组 tagging P@K 与上游协议一致。
- 项目适配：ID 关系与轨迹字典先转换为类别三元组和关系区间内连续 tubes；轨迹缺口会切成多个连续预测并记录数量。
- 诊断实现：`src/vidvrd_auto/evaluation/diagnostic/track_aligned.py`，登记名 `diagnostic_track_aligned_v2`；在 VidVRD 实际有标注的 Gold 帧上计算轨迹 IoU 并做匈牙利映射，时间 IoU、base/novel 与逐谓词分析仅用于内部定位问题。
- 数据边界：`benchmark_official.json` 只选择 test split。当前本地 50 视频 Gold 中只有 10 个 test 视频，所以只能产生协议兼容的部分结果，不能冒充完整 200 视频测试集结果。

## 代码治理规则

1. `third_party` 目录禁止语义修改；更新上游版本必须同时更新来源、许可和哈希。
2. 项目增强只能放在 adapter、scheduler、localizer 或 pipeline 层，并使用独立算法名。
3. 所有模型名、prompt 版本、配置哈希、代码 revision 和时间区间约定写入 manifest。
4. 未经正式实验验证的分数只能称为 support 或 heuristic score，不能称为校准概率。
5. 新产物的规范化时间区间使用半开区间；v1 旧产物仍可按显式闭区间读取。
