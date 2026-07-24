# Gold 数据

本目录的三个 JSON 文件由 `sample_folder/sample_vidvrd/anno` 中 50 份官方格式标注机械转换生成：50 个视频、130 条轨迹、1140 个关系实例，实际出现 85 种谓词。

生成命令：

```powershell
vidvrd-build-gold --annotations sample_folder/sample_vidvrd/anno --out-dir gold
```

转换不改类别、谓词、轨迹框或 track ID，只把官方关系的半开时间区间 `[begin_fid, end_fid)` 转为项目使用的闭区间 `[start_frame, end_frame]`，并附加官方开放词汇 base/novel 划分。

为防止标签泄漏，Gold 只允许由末尾评测节点读取。词表发现、Rex-Omni、OC-SORT、关系规则、语义模型和复核节点均不得读取本目录。
