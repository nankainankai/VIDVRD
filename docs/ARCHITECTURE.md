# 架构

## 分层

`pipeline` 只编排阶段、状态、缓存和文件契约；`nodes` 提供薄入口；算法分别位于 `detection`、`tracking`、`relations`、`evaluation`。所有 DashScope 调用集中通过 `providers`，算法模块不直接读取环境变量。

```text
configs/vidvrd_ontology.json     官方 35/132 本体与 base/novel 划分
src/vidvrd_auto/core/            配置、路径、数据契约、本体访问
src/vidvrd_auto/pipeline/        单视频和多视频编排
src/vidvrd_auto/nodes/           阶段接口
src/vidvrd_auto/detection/       Rex-Omni 与时序检测调度
src/vidvrd_auto/tracking/        OC-SORT 适配器、插值与窗口生成
src/vidvrd_auto/relations/       几何、语义、聚合与复核
src/vidvrd_auto/evaluation/      Gold 转换、轨迹对齐和关系指标
```

## 开放词汇检测

固定模式直接把官方 35 类交给 Rex-Omni。开放模式先从均匀采样帧生成视频级拼图，云模型只返回简短对象名；本体归一化后与官方 35 类及显式扩展类取并集。检测同时保留 `raw_class_name` 和标准化 `class_name`，因此本体内类别可直接评测，本体外类别仍可继续跟踪和关系推理。

开放发现是检测之前的词表构建，不是视频前筛：它不拒绝视频，也不决定某帧是否检测。

## 稀疏检测与连续轨迹

视频逐帧解码，Rex-Omni 默认处理第 0、5、10…帧；若与上一锚点的缩略图差异超过阈值且满足最小间隔，则提前检测。未检测帧仍写入 JSONL 并明确标记 `skipped`。

跟踪器使用固定提交、未经修改的官方 OC-SORT 核心。项目适配器按类别运行核心实例，并统一分配全局 ID，避免跨类别切换和分组 ID 碰撞。中间帧先由 OC-SORT 传播；仅当前后都有真实观测且缺口不超过阈值时，才用线性插值替换短缺口。这样轨迹连续，但不会把无依据的预测伪装成检测。

## 窗口级关系

关系以 30 帧窗口、15 帧步长为基本单元：

```text
轨迹框 -> 加权几何投票 --------┐
轨迹对局部拼图 -> 视觉语义判断 ├-> 合并 -> 跨窗口聚合 -> 风险复核 -> 反向耦合
                              ┘
```

几何投票对 observed/interpolated/predicted 分别赋权，并要求至少一帧成对真实观测。语义节点对每个方向提供完整官方谓词候选，类别不负责硬删候选。跨窗口阶段按 `(subject, predicate, object)` 合并重叠或相邻片段并生成稳定关系 ID。

## 复核

复核只处理低置信度、互斥冲突和风险轨迹相关项目，并只附带相应轨迹对的真实拼图。模型可建议 `keep/delete/change_predicate/adjust_span`；谓词和时间范围验证通过后才真正修改输出。规则化互斥消解、置信度过滤和反向耦合在复核后执行。

## Gold 与评测隔离

Gold 转换器把官方半开区间 `[begin_fid, end_fid)` 转成项目闭区间 `[start_frame, end_frame]`。评测先在每个视频内按标准化类别和轨迹体积 IoU 做一对一匈牙利匹配，再以映射后的主客体 ID、谓词和时间 IoU 匹配关系。

Gold 路径只传给流水线末尾的 `evaluate`，不传给任何预测节点。`benchmark.json` 关闭动态词表发现，保证 Gold 视频上的检测词表固定且可重复。

## 状态与恢复

每个阶段保存输入哈希、配置哈希、产物列表和 `status.json`。`--resume` 只复用输入未变且产物完整的成功阶段；`--force` 显式重跑。流水线不会自动删除既有 `runs/`、`models/` 或输入数据。
