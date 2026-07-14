# my_scripts：旧脚本适配层

`my_scripts/` 现在不是主工程入口，只保留被 `src/vidvrd_auto/` 调用的 legacy 算子和少量环境辅助脚本。新的编排、恢复、导出和评测入口统一在 `python -m vidvrd_auto.cli`。

## 当前保留文件

| 文件 | 作用 | 调用方 |
|---|---|---|
| `step1_full_video_box_detection_dinox.py` | Rex-Omni / DINO-X 检测，输出 `detections_full.jsonl`、`video_meta.json` | `src/vidvrd_auto/detection/legacy_step1.py` |
| `step2_full_video_tracking_ocsort_qc_pairviz.py` | OC-SORT 追踪，输出 `tracks_full.jsonl`、`windows.json` | `src/vidvrd_auto/tracking/legacy_step2.py` |
| `semi_auto_label_relations.py` | Storyboard + Qwen-VL 片段关系验证 | `src/vidvrd_auto/relations/clip_classifier.py` |
| `config.py`、`utils_io.py`、`modules/` | 上述 legacy 脚本的依赖 | legacy 脚本内部 |
| `install_rexomni_deps.py`、`download_rexomni_model.py` | Rex-Omni 环境和模型准备 | 手动运行 |

## 推荐运行方式

从仓库根目录调用新主链：

```bash
python -m vidvrd_auto.cli --videos data/videos.txt --run_dir runs/exp001 --config configs/default.json --resume
```

语义关系实验使用：

```bash
python -m vidvrd_auto.cli --videos data/videos_semantic.txt --run_dir runs/semantic_v1 --config configs/semantic_relations.json --resume --api_key YOUR_DASHSCOPE_KEY
```

## 单独调试 legacy 脚本

只有在定位检测、追踪或关系分类节点问题时，才建议直接运行本目录脚本。

Step1 检测：

```bash
python my_scripts/step1_full_video_box_detection_dinox.py --backend rexomni --rex_model_path my_scripts/models/Rex-Omni-AWQ --rex_categories "person,skateboard,bicycle,horse,dog,cat,ball,guitar,microphone,chair,table,car,bag,cup,phone,surfboard" --video path/to/video.mp4 --output_dir runs/debug/step1_detect
```

Step2 追踪：

```bash
python my_scripts/step2_full_video_tracking_ocsort_qc_pairviz.py --video path/to/video.mp4 --detections_jsonl runs/debug/step1_detect/detections_full.jsonl --output_dir runs/debug/step2_track
```

片段关系验证：

```bash
python my_scripts/semi_auto_label_relations.py --windows_json runs/debug/step2_track/windows.json --tracks_jsonl runs/debug/step2_track/tracks_full.jsonl --output_json runs/debug/relation_llm/relations_llm.json --video_id debug_video --save_storyboards_dir runs/debug/relation_llm/storyboards --pair_storyboard --resume --api_key YOUR_DASHSCOPE_KEY
```

## 维护约定

- 不再新增 `run_all.py`、Step3/4/5 这类平行主链。
- 新能力优先进入 `src/vidvrd_auto/`，本目录只保留必要 legacy adapter。
- 如果某个 legacy 脚本不再被 `src/vidvrd_auto/` 调用，应迁移或删除，并同步更新本文件。
