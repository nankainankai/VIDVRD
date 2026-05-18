"""最小化配置：仅保留 Step1/Step2/Step3 运行所需字段。"""

import os

# -------------------- DashScope --------------------
API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
API_MODEL = "qwen-vl-max"

# -------------------- DINO-X --------------------
DINOX_API_TOKEN = os.getenv("DINOX_API_TOKEN", "").strip()
DINOX_TEXT_PROMPT = "person"
DINOX_MODEL = "DINO-X-1.0"
DINOX_BBOX_THRESHOLD = 0.25
DINOX_IOU_THRESHOLD = 0.8
DINOX_NMS_IOU_THRESHOLD = 0.55
DINOX_MIN_BOX_AREA = 500.0
DINOX_MAX_DETECTIONS_PER_FRAME = 60
DINOX_IMAGE_MAX_LONG_EDGE = 640
DINOX_DETECTION_INTERVAL = 5
DINOX_MAX_CALLS = 50
DINOX_REQUEST_RETRIES = 1
DINOX_REQUEST_BACKOFF_SEC = 1.2

# -------------------- Detector Backend --------------------
# 可选：'dinox' 或 'rexomni'
# 当前默认：rexomni（按项目现阶段需求优先本地推理；如需云端 DINO-X，可设置 DETECTOR_BACKEND=dinox）
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "rexomni").strip().lower()

# -------------------- Rex-Omni (optional) --------------------
# 说明：Rex-Omni 依赖较重（torch/transformers 等），仅在启用时才会被导入。
REXOMNI_MODEL_PATH = os.getenv("REXOMNI_MODEL_PATH", "IDEA-Research/Rex-Omni").strip()
REXOMNI_BACKEND = os.getenv("REXOMNI_BACKEND", "transformers").strip()

# 用于 detection task 的类别列表（逗号分隔）。为空时会回退到 DINOX_TEXT_PROMPT。
_rex_cats = os.getenv("REXOMNI_CATEGORIES", "").strip()
REXOMNI_CATEGORIES = [c.strip() for c in _rex_cats.split(",") if c.strip()] if _rex_cats else []

REXOMNI_DETECTION_INTERVAL = int(os.getenv("REXOMNI_DETECTION_INTERVAL", "1"))
REXOMNI_MIN_BOX_AREA = float(os.getenv("REXOMNI_MIN_BOX_AREA", str(DINOX_MIN_BOX_AREA)))
REXOMNI_MAX_DETECTIONS_PER_FRAME = int(
	os.getenv("REXOMNI_MAX_DETECTIONS_PER_FRAME", str(DINOX_MAX_DETECTIONS_PER_FRAME))
)

# Rex-Omni generation length: smaller is faster (detection JSON is short).
# On CPU, consider 256/512 for acceptable speed.
REXOMNI_MAX_TOKENS = int(os.getenv("REXOMNI_MAX_TOKENS", "512"))

# -------------------- OC-SORT --------------------
TRACKING_IOU_THRESHOLD = 0.5
TRACKING_MAX_AGE = 30
TRACKING_MIN_HITS = 3
TRACKING_CLASS_AWARE = True
TRACKING_MAX_CENTER_DIST_RATIO = 0.8
TRACKING_MIN_NEW_TRACK_CONF = 0.35
TRACKING_VELOCITY_ALPHA = 0.8

# -------------------- IO --------------------
OUTPUT_DIR = r"C:\video_output2"
FULL_DETECTIONS_JSONL_NAME = "detections_full.jsonl"
FULL_TRACKS_JSONL_NAME = "tracks_full.jsonl"
TRACK_INDEX_JSON_NAME = "track_index.json"
WINDOWS_JSON_NAME = "windows.json"
VIDEO_META_JSON_NAME = "video_meta.json"
QC_REPORT_JSON_NAME = "qc_report.json"
REVIEW_BUNDLE_DIR_NAME = "review_bundle"
STEP1_EXPORT_BOX_VIDEO = True
STEP1_BOX_VIS_VIDEO_NAME = "step1_detection_box_vis.mp4"

# -------------------- Sliding Window --------------------
WINDOW_SIZE_FRAMES = 30
WINDOW_STRIDE_FRAMES = 15

# -------------------- Step2 QC (qwen-vl-max) --------------------
ENABLE_STEP2_LLM_QC = True
STEP2_LLM_QC_MODEL = "qwen-vl-max"
STEP2_QC_SAMPLE_FRAMES = 8
STEP2_QC_COUNT_DIFF_THRESHOLD = 1

# -------------------- Step2 Pair Visualization --------------------
EXPORT_PAIR_VIZ_VIDEOS = True
PAIR_VIZ_MAX_WINDOWS = 0
PAIR_VIZ_MAX_PAIRS_PER_WINDOW = 6

# -------------------- Step5 Qwen Summary --------------------
STEP5_QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
STEP5_QWEN_MODEL = "qwen-omni-turbo-realtime-latest"
STEP5_QWEN_FALLBACK_MODEL = "qwen-max"
STEP5_MAX_EVENTS = 60
