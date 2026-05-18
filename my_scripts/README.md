# my_scripts：五步主链

当前运行主链：

1. Step1：整段视频框级标注（默认 Rex-Omni，可切 DINO-X）
2. Step2：OC-SORT 基于视频帧逐帧追踪
3. Step3：关系分类（方法2 + 方法3 + 融合）
4. Step4：视频级关系事件聚合
5. Step5：Qwen 生成视频关系自然语言描述

## 环境

使用 conda 环境 `vidvrd` 运行。

### API Key / Token

本仓库不再在 `config.py` 明文保存密钥。

- DashScope/Qwen：设置环境变量 `DASHSCOPE_API_KEY`，或在 Step2/Step3/Step5 通过 `--api_key` 传入。
- DINO-X（可选）：如切换到 DINO-X 后端，需要设置环境变量 `DINOX_API_TOKEN`（Step1 使用）。

## 一键运行（Step1→Step2→可选 Step3~Step5）

脚本：`my_scripts/run_all.py`

- 仅本地（无 API key）：会跑 Step1+Step2，然后自动跳过 Step3~Step5

`python run_all.py --backend rexomni --rex_model_path "models/Rex-Omni" --rex_categories "person" --keyframe_interval 25 --no_save_box_video`

- 带 DashScope key（可跑全链 Step1~Step5）：

`python run_all.py --backend rexomni --rex_model_path "models/Rex-Omni" --rex_categories "person" --keyframe_interval 25 --no_save_box_video --api_key YOUR_KEY`

## 一键半自动 Phase-1（Step2→候选生成→Presence 评测）

脚本：`my_scripts/run_phase1.py`

用途：在已有 Step1 输出（`detections_full.jsonl`）的前提下，一条命令完成：

- Step2 追踪生成 `windows.json` / `tracks_full.jsonl`
- 半自动关系候选生成 `pred/relations_pred.json`（可选保存 storyboard 证据图、可断点续跑）
- 若存在 `gold/relations_gold.json`，自动做 presence 评测并输出报告

最常用（推荐，带证据图 + 断点续跑）：

`python run_phase1.py --output_dir C:/video_output2 --save_storyboards_dir C:/video_output2/storyboards --resume --api_key YOUR_KEY`

注意：`run_phase1.py` 目前不会自动调用 Step1。第一次处理新视频时，请先运行 `run_all.py` 的 Step1~Step2，或单独运行 Step1 生成同一 `output_dir` 下的 `detections_full.jsonl`；否则 Step2 会因为缺少检测结果退出。

只想先看抽帧/画框是否合理（不调用模型）：

`python run_phase1.py --output_dir C:/video_output2 --save_storyboards_dir C:/video_output2/storyboards --dry_run --resume`

如果你之前用 `--dry_run --resume` 跑过，后续正式跑发现全是 `SKIP seg=... (resume)`，可以加 `--reset_progress` 重置断点：

`python run_phase1.py --output_dir C:/video_output2 --save_storyboards_dir C:/video_output2/storyboards --resume --reset_progress --api_key YOUR_KEY`

输出：

- Pred：`pred/relations_pred.json`
- 评测报告（若 gold 存在）：`OUTPUT_DIR/presence_report.md`

## 人工标注界面（使用学长 Tkinter 平台）

关系标注 GUI 已迁到：`tools/manual_annotation/anno_platform.py`。

它需要两类输入：

- 抽帧后的图片文件夹：`<data_folder>/<video_id>/*.jpg`（文件名用帧号，例如 `000123.jpg`）
- 轨迹 JSON：顶层 key 必须等于 `<video_id>`，并且包含 `anno:[{tid, category, trajectory:{frame:[x1,y1,x2,y2]}}]`

本仓库 Step2 输出是 `tracks_full.jsonl + windows.json`，格式不同。可用下面脚本一键转换 + 导出帧：

`python prepare_anno_platform_inputs.py --windows_json C:/video_output2/windows.json --tracks_jsonl C:/video_output2/tracks_full.jsonl --out_frames_root C:/video_output2/anno_frames --prefer_observed_bbox`

脚本会打印一条可直接运行的 GUI 启动命令（含 `--data_folder/--track_json/--annotation_folder`）。

## Step1：整段视频框级标注（后端可选）

脚本：`my_scripts/step1_full_video_box_detection_dinox.py`

Rex-Omni 依赖一键安装（推荐先执行）：

`python install_rexomni_deps.py --upgrade_pip`

如果你需要顺带安装 torch（按机器选择）：

`python install_rexomni_deps.py --torch cpu`

可选值：`--torch cpu|cu121|cu124|skip`（Windows 下不建议折腾 flash-attn/vllm）

运行（默认使用 Rex-Omni，本地）：

`C:/software/anaconda3/envs/vidvrd/python.exe my_scripts/step1_full_video_box_detection_dinox.py --backend rexomni --rex_categories "person" --video "你的视频路径.mp4"`

如果你希望“运行 Step1 时自动安装 Rex-Omni 依赖”（首次跑更省事）：

`python step1_full_video_box_detection_dinox.py --backend rexomni --rex_categories "person" --auto_install_rexomni --auto_install_torch cpu`

如果你这边网络访问 HuggingFace 经常超时（推荐先离线下载模型到本地，再跑 Step1）：

- 设置镜像（可选，PowerShell）：`setx HF_ENDPOINT "https://hf-mirror.com"`
- 下载模型快照到本地：

`python download_rexomni_model.py --repo_id IDEA-Research/Rex-Omni --local_dir models/Rex-Omni`

- 用本地模型路径运行：

`python step1_full_video_box_detection_dinox.py --backend rexomni --rex_model_path models/Rex-Omni --rex_categories "person"`

可选：切换回 DINO-X（云端）作为检测后端：

`C:/software/anaconda3/envs/vidvrd/python.exe my_scripts/step1_full_video_box_detection_dinox.py --backend dinox --video "你的视频路径.mp4"`

说明：

- 默认后端为 `rexomni`（见 `config.py`）。也可以通过环境变量 `DETECTOR_BACKEND=dinox` 切换到 DINO-X。
- Rex-Omni 依赖较重（torch/transformers 等），未安装时启用会报导入错误。
- Rex-Omni 相关环境变量：`REXOMNI_MODEL_PATH`、`REXOMNI_BACKEND`、`REXOMNI_CATEGORIES`。

输出到 `OUTPUT_DIR`：

- `detections_full.jsonl`
- `video_meta.json`

## Step2：OC-SORT 逐帧轨迹追踪 + qwen-vl-max 质检

脚本：`my_scripts/step2_full_video_tracking_ocsort_qc_pairviz.py`

运行：

`C:/software/anaconda3/envs/vidvrd/python.exe my_scripts/step2_full_video_tracking_ocsort_qc_pairviz.py --video "你的视频路径.mp4" --detections_jsonl C:/video_output/detections_full.jsonl`

输出到 `OUTPUT_DIR`：

- `tracks_full.jsonl`
- `track_index.json`
- `windows.json`（30 帧窗口，15 帧重叠）
- `qc_report.json`（`qwen-vl-max` 数量一致性质检）
- `review_bundle/pair_videos/*.mp4`（轨迹对可视化视频）
- `review_bundle/pair_videos_index.json`

## Step3：关系分类

脚本：`my_scripts/step3_window_relation_classification.py`

运行：

`C:/software/anaconda3/envs/vidvrd/python.exe my_scripts/step3_window_relation_classification.py --windows_json C:/video_output/windows.json --tracks_jsonl C:/video_output/tracks_full.jsonl`

Step3 内部执行：

- 方法2：窗口级 storyboard 描述 -> 三元组抽取
- 方法3：轨迹对聚焦 + 四类关系提示（静态位置/动态位置/静态动作/动态动作）
- 融合：对方法2和方法3候选关系做融合，生成最终关系

输出到 `OUTPUT_DIR`：

- `segment_descriptions/seg_XXXX.txt`
- `segment_relations/seg_XXXX.method2.json`
- `segment_relations/seg_XXXX.method3.json`
- `segment_relations/seg_XXXX.final.json`
- `relations_candidates.json`
- `relations_final.json`

## Step4：视频级关系事件聚合

脚本：`my_scripts/step4_video_relation_event_aggregation.py`

运行：

`C:/software/anaconda3/envs/vidvrd/python.exe my_scripts/step4_video_relation_event_aggregation.py --relations_final C:/video_output/relations_final.json`

输出到 `OUTPUT_DIR`：

- `video_relations.json`

## Step5：Qwen 生成视频关系自然语言描述

脚本：`my_scripts/step5_video_relation_natural_language_qwen.py`

运行：

`C:/software/anaconda3/envs/vidvrd/python.exe my_scripts/step5_video_relation_natural_language_qwen.py --video_relations_json C:/video_output/video_relations.json --api_key "你的QwenKey"`

说明：

- Step5 会调用 Qwen 模型生成一段自然语言描述。
- `config.py` 中 `STEP5_QWEN_API_KEY` 默认留空，也可以通过 `--api_key` 传入。

---

## Phase-1：半自动关系标注（候选生成）

脚本：`my_scripts/semi_auto_label_relations.py`

用途：读取 Step2 的 `windows.json` + `tracks_full.jsonl` + 原视频，生成窗口级关系候选（Pred），供人工复核后沉淀为 Gold。

最小运行（会调用 Qwen-VL，多模态）：

`C:/software/anaconda3/envs/vidvrd/python.exe my_scripts/semi_auto_label_relations.py --windows_json C:/video_output/windows.json --tracks_jsonl C:/video_output/tracks_full.jsonl --output_json pred/relations_pred.json`

常用选项：

- `--relations "left,right,above,below"`：谓词列表（逗号分隔）。脚本会把常见中文同义词归一化到英文 canonical（例如“左边”→`left`），最终写入 Pred 也是 canonical。
- `--group_size 3`：每次问模型的谓词数（减少幻觉，便于控制 token）。
- `--max_frames_per_window 8`：每个窗口抽取的关键帧数上限。
- `--save_storyboards_dir C:/video_output/storyboards`：保存每个窗口的拼图证据图，方便人工复核。
- `--resume`：断点续跑（会写入 `pred/relations_pred.json.progress.json` 记录已处理的 segment）。
- `--reset_progress`：重置断点（忽略并覆盖 progress sidecar），用于你想从头重跑的场景。
- `--dry_run`：只生成/保存 storyboard，不调用模型（用于先检查抽帧和画框是否合理）。
- `--retries 2 --backoff_sec 1.5 --sleep_sec 0.2`：失败重试 + 退避 + 限速。

输出：

- `pred/relations_pred.json`：关系候选（Presence 评测用 Pred）
- `pred/relations_pred.json.progress.json`：进度 sidecar（断点续跑用）

## Phase-1：Presence 评测（Gold vs Pred）

脚本：`tools/evaluate_presence.py`

用途：忽略时间段，只按 `(subject_id, predicate, object_id)` 做 presence 匹配，输出 Precision/Recall/F1 及 FP/FN 案例。

示例：

`C:/software/anaconda3/envs/vidvrd/python.exe tools/evaluate_presence.py --gold gold/relations_gold.json --pred pred/relations_pred.json`
