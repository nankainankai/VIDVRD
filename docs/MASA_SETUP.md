# MASA-R50 接入

主路线 `hybrid_sparse_reid` 使用 MASA 官方 plug-and-play R50 模型提取实例外观向量。项目不复制或修改 MASA 源码；当前适配层调用官方 `init_masa`、`inference_masa` 和官方 `track_head.predict`。

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

## 配置

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

主路线缺少 MASA 环境或权重时直接报错，不静默换成颜色直方图或其他 ReID。`reference_dense` 不加载 MASA。

## 输出语义

MASA 只提供每个检测框的外观 embedding，不直接接管项目 ID。项目联合关联器统一计算运动、外观、IoU 和软类别代价，离线拼接器再生成全局 ID。这样两级 ID、代价和拼接边都能审计，且不会把 MASA 自带 tracker 的时间假设混入 3–5 帧稀疏锚点时钟。
