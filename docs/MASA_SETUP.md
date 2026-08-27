# MASA-R50 实验模块（未接入主流程）

该目录记录一个未完成效果验证的实验模块：使用 MASA 官方 plug-and-play R50 模型提取实例外观向量，再交给项目自有 hybrid 关联与 stitching 代码。当前主路线固定使用 OC-SORT，`track_video()` 不提供 `hybrid_sparse_reid` 入口，正式配置也不会加载 MASA。

## 版本与目录

- 官方仓库：<https://github.com/siyuanliii/masa>
- 当前核对提交：`c5472b9c7615f35abdf1188cb1a0c5408fe50d66`
- 许可证：Apache-2.0
- 官方权重：<https://huggingface.co/dereksiyuanli/masa/resolve/main/masa_r50.pth>

建议目录：

```text
external/masa/                                      # 上述固定提交
models/MASA/masa_r50.pth                            # 官方权重
external/masa/configs/masa-one/masa_r50_plug_and_play.py
```

检出版本：

```powershell
git clone https://github.com/siyuanliii/masa.git external/masa
git -C external/masa checkout c5472b9c7615f35abdf1188cb1a0c5408fe50d66
```

按 MASA 官方 `INSTALL.md` 在视频模型环境中安装 MMEngine、MMCV、MMDetection 与 MASA 本身。基础项目依赖不强行加入整套 OpenMMLab，避免参考路线和纯契约测试也被重依赖绑死。

## 历史实验配置

```json
{
  "tracking": {
    "algorithm": "hybrid_sparse_reid",
    "masa_config": "external/masa/configs/masa-one/masa_r50_plug_and_play.py",
    "masa_checkpoint": "models/MASA/masa_r50.pth",
    "masa_revision": "c5472b9c7615f35abdf1188cb1a0c5408fe50d66",
    "appearance_device": "cuda:0",
    "appearance_fp16": true
  }
}
```

这段配置只记录实验模块原有参数，不能直接传给当前流水线。正常运行项目无须安装 MASA、MMCV 或 MMDetection。

## 输出语义

实验模块中，MASA 只提供每个检测框的外观 embedding，不直接接管项目 ID；联合关联与离线拼接均为项目实验代码。它们保留在仓库中供以后单独研究，不属于当前项目算法，也不参与主链路结果。
