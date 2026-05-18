from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2

from modules.object_detector import ObjectDetector
import config


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _select_video_gui() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        video_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")],
        )
    finally:
        root.destroy()
    return str(video_path or "")


def _compact_objects(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for o in objects or []:
        try:
            bbox = o.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            out.append(
                {
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "class": int(o.get("class", -1)),
                    "class_name": str(o.get("class_name", "unknown")),
                    "confidence": float(o.get("confidence", 0.0)),
                }
            )
        except Exception:
            continue
    return out


def _draw_detection_boxes(frame_bgr, objects: List[Dict[str, Any]]):
    out = frame_bgr.copy()
    for o in objects or []:
        bbox = o.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        try:
            x1, y1, x2, y2 = [int(float(v)) for v in bbox]
            cls_name = str(o.get("class_name", "unknown"))
            conf = float(o.get("confidence", 0.0))
        except Exception:
            continue

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(out, label, (x1 + 4, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _bbox_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    try:
        ax1, ay1, ax2, ay2 = [float(x) for x in a]
        bx1, by1, bx2, by2 = [float(x) for x in b]
    except Exception:
        return 0.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def _greedy_match_by_iou(
    prev_objs: List[Dict[str, Any]],
    next_objs: List[Dict[str, Any]],
    iou_thresh: float,
) -> List[Tuple[int, int]]:
    """Greedy IoU matching within the same class_name."""

    pairs: List[Tuple[float, int, int]] = []
    for i, po in enumerate(prev_objs):
        pb = po.get("bbox")
        if not (isinstance(pb, (list, tuple)) and len(pb) == 4):
            continue
        pcls = str(po.get("class_name", "unknown"))
        for j, no in enumerate(next_objs):
            nb = no.get("bbox")
            if not (isinstance(nb, (list, tuple)) and len(nb) == 4):
                continue
            ncls = str(no.get("class_name", "unknown"))
            if pcls != ncls:
                continue
            iou = _bbox_iou_xyxy(pb, nb)
            if iou >= float(iou_thresh):
                pairs.append((float(iou), i, j))

    pairs.sort(key=lambda x: x[0], reverse=True)
    used_i: set[int] = set()
    used_j: set[int] = set()
    out: List[Tuple[int, int]] = []
    for iou, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((i, j))
    return out


def _interpolate_objects(
    prev_objs: List[Dict[str, Any]],
    next_objs: List[Dict[str, Any]],
    alpha: float,
    matches: List[Tuple[int, int]],
    hold_prev_unmatched: bool = True,
) -> List[Dict[str, Any]]:
    """Linear interpolate bbox/score for matched boxes; optionally hold prev unmatched."""

    alpha = float(max(0.0, min(1.0, alpha)))
    out: List[Dict[str, Any]] = []
    matched_prev = set(i for i, _ in matches)

    for i, j in matches:
        po = prev_objs[i]
        no = next_objs[j]
        pb = po.get("bbox")
        nb = no.get("bbox")
        if not (isinstance(pb, (list, tuple)) and len(pb) == 4):
            continue
        if not (isinstance(nb, (list, tuple)) and len(nb) == 4):
            continue

        ibox = [
            (1.0 - alpha) * float(pb[0]) + alpha * float(nb[0]),
            (1.0 - alpha) * float(pb[1]) + alpha * float(nb[1]),
            (1.0 - alpha) * float(pb[2]) + alpha * float(nb[2]),
            (1.0 - alpha) * float(pb[3]) + alpha * float(nb[3]),
        ]
        pconf = float(po.get("confidence", 0.0))
        nconf = float(no.get("confidence", 0.0))
        iconf = (1.0 - alpha) * pconf + alpha * nconf

        out.append(
            {
                "bbox": ibox,
                "class": int(po.get("class", no.get("class", -1))),
                "class_name": str(po.get("class_name", no.get("class_name", "unknown"))),
                "confidence": float(iconf),
            }
        )

    if hold_prev_unmatched:
        for idx, po in enumerate(prev_objs):
            if idx in matched_prev:
                continue
            out.append(po)

    return _compact_objects(out)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step1: full-video box detection (backend selectable)")
    parser.add_argument("--video", type=str, default="", help="视频路径，不传则弹窗选择")
    parser.add_argument("--output_dir", type=str, default=str(getattr(config, "OUTPUT_DIR", "C:/video_output")))

    parser.set_defaults(save_box_video=bool(getattr(config, "STEP1_EXPORT_BOX_VIDEO", True)))
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--save_box_video",
        dest="save_box_video",
        action="store_true",
        help="导出带检测框可视化视频（默认按 config.py）",
    )
    g.add_argument(
        "--no_save_box_video",
        dest="save_box_video",
        action="store_false",
        help="不导出可视化视频（可明显加速/省磁盘 IO）",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="",
        choices=["", "dinox", "rexomni"],
        help="Detector backend: dinox (default) or rexomni",
    )

    parser.add_argument(
        "--rex_model_path",
        type=str,
        default="",
        help="Rex-Omni model path or HuggingFace repo id (only used when --backend rexomni)",
    )
    parser.add_argument(
        "--rex_backend",
        type=str,
        default="",
        help="Rex-Omni engine backend (e.g. transformers). Only used when --backend rexomni",
    )
    parser.add_argument(
        "--rex_categories",
        type=str,
        default="",
        help="Comma-separated categories for Rex-Omni detection. Only used when --backend rexomni",
    )

    parser.add_argument(
        "--keyframe_interval",
        type=int,
        default=1,
        help="每 N 帧做一次真实检测；当 N>1 时，中间帧对框做插值补全（线性插值 + IoU 匹配）",
    )
    parser.add_argument(
        "--interp_iou_thresh",
        type=float,
        default=0.1,
        help="关键帧前后框匹配的 IoU 阈值（越大越严格）",
    )

    parser.add_argument(
        "--auto_install_rexomni",
        action="store_true",
        help="If rexomni backend deps are missing, auto-run install_rexomni_deps.py then retry once",
    )
    parser.add_argument(
        "--auto_install_torch",
        type=str,
        default="skip",
        choices=["skip", "cpu", "cu121", "cu124"],
        help="Torch channel for auto install (only used with --auto_install_rexomni)",
    )
    return parser


def _maybe_auto_install_rexomni(args: argparse.Namespace) -> bool:
    if not bool(getattr(args, "auto_install_rexomni", False)):
        return False

    script_path = Path(__file__).resolve().parent / "install_rexomni_deps.py"
    if not script_path.exists():
        print(f"ERROR: auto-install script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path), "--upgrade_pip"]
    torch_opt = str(getattr(args, "auto_install_torch", "skip") or "skip").strip().lower()
    if torch_opt and torch_opt != "skip":
        cmd.extend(["--torch", torch_opt])

    print("=" * 70)
    print("Auto installing Rex-Omni dependencies...")
    print("Command:", " ".join(cmd))
    print("=" * 70)
    try:
        subprocess.check_call(cmd)
        return True
    except Exception as e:
        print("ERROR: auto-install failed:", e)
        return False


def main() -> None:
    args = _build_parser().parse_args()

    video_path = (args.video or "").strip()
    if not video_path:
        video_path = _select_video_gui()
    if not video_path or not os.path.exists(video_path):
        print(f"ERROR: 视频不存在: {video_path}")
        return

    output_dir = Path(args.output_dir).expanduser().resolve()
    _ensure_dir(output_dir)

    detections_name = str(getattr(config, "FULL_DETECTIONS_JSONL_NAME", "detections_full.jsonl"))
    meta_name = str(getattr(config, "VIDEO_META_JSON_NAME", "video_meta.json"))
    box_video_name = str(getattr(config, "STEP1_BOX_VIS_VIDEO_NAME", "step1_detection_box_vis.mp4"))
    detections_path = output_dir / detections_name
    meta_path = output_dir / meta_name
    box_video_path = output_dir / box_video_name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: 无法打开视频")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = float(total_frames / float(max(1, fps)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    box_video_writer = None
    if bool(args.save_box_video):
        box_video_writer = cv2.VideoWriter(
            str(box_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            int(max(1, fps)),
            (max(1, width), max(1, height)),
        )

    rex_categories = None
    if args.rex_categories.strip():
        rex_categories = [c.strip() for c in args.rex_categories.split(",") if c.strip()]

    keyframe_interval = max(1, int(getattr(args, "keyframe_interval", 1)))
    detector_kwargs: Dict[str, Any] = dict(
        backend=args.backend,
        rex_model_path=(args.rex_model_path.strip() or None),
        rex_backend=(args.rex_backend.strip() or None),
        rex_categories=rex_categories,
    )
    # When doing keyframe mode, we control the skipping outside the detector.
    if keyframe_interval > 1:
        detector_kwargs["detection_interval"] = 1

    detector = ObjectDetector(**detector_kwargs)
    try:
        detector.load_model()
    except ImportError as e:
        print("ERROR: detector backend 不可用（缺少依赖或导入失败）")
        print(str(e))

        # optional auto install for rexomni
        if str(getattr(detector, "backend", "")).strip().lower() == "rexomni":
            if _maybe_auto_install_rexomni(args):
                try:
                    detector.load_model()
                except ImportError as e2:
                    print("ERROR: auto-install finished but backend still unavailable")
                    print(str(e2))
                    print("你也可以手动运行：python install_rexomni_deps.py --upgrade_pip")
                    print("如需安装 torch：python install_rexomni_deps.py --torch cpu")
                    return
            else:
                print("你也可以手动运行：python install_rexomni_deps.py --upgrade_pip")
                print("如需安装 torch：python install_rexomni_deps.py --torch cpu")
                return
        else:
            print("建议先运行：python install_rexomni_deps.py --upgrade_pip")
            return

    print("=" * 70)
    print("Step1: 全视频框级标注")
    print(f"detector_backend={getattr(detector, 'backend', 'unknown')}")
    print(f"视频: {video_path}")
    print(f"fps={fps}, total_frames={total_frames}, duration={duration:.2f}s")
    print(f"输出: {detections_path}")
    if keyframe_interval > 1:
        print(f"keyframe_interval={keyframe_interval} (interpolate=on, iou_thresh={float(args.interp_iou_thresh):.3f})")
    print("=" * 70)

    frame_num = 0
    err_count = 0
    # keyframe interpolation states
    prev_key_idx: Optional[int] = None
    prev_key_objs: List[Dict[str, Any]] = []
    buffer_frames: List[Tuple[int, Any]] = []  # (frame_idx, frame_bgr)

    def _write_row(
        fp,
        idx: int,
        frame_bgr,
        objects_compact: List[Dict[str, Any]],
    ) -> None:
        ts = float(idx / max(1, fps))
        row = {"frame": int(idx), "timestamp": float(ts), "objects": objects_compact}
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        if box_video_writer is not None and frame_bgr is not None:
            vis = _draw_detection_boxes(frame_bgr, objects_compact)
            if vis.shape[1] != width or vis.shape[0] != height:
                vis = cv2.resize(vis, (max(1, width), max(1, height)))
            box_video_writer.write(vis)
    with detections_path.open("w", encoding="utf-8") as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if keyframe_interval <= 1:
                try:
                    t0 = time.time()
                    objects = detector.detect(frame)
                    dt = time.time() - t0
                except Exception as e:
                    objects = []
                    dt = -1.0
                    err_count += 1
                    if err_count <= 3:
                        print(f"WARN: detect failed at frame={frame_num}: {e}")
                _write_row(f, frame_num, frame, _compact_objects(objects))

                if frame_num % 10 == 0:
                    if dt >= 0:
                        print(f"Processed frame {frame_num}/{max(1, total_frames - 1)} (detect {dt:.2f}s)")
                    else:
                        print(f"Processed frame {frame_num}/{max(1, total_frames - 1)}")

            else:
                # Keyframe mode: write frames in order, but delay intermediate frames
                if prev_key_idx is None:
                    # first frame becomes the first keyframe
                    try:
                        t0 = time.time()
                        objects = detector.detect(frame)
                        dt0 = time.time() - t0
                    except Exception as e:
                        objects = []
                        dt0 = -1.0
                        err_count += 1
                        if err_count <= 3:
                            print(f"WARN: detect failed at frame={frame_num}: {e}")

                    prev_key_idx = int(frame_num)
                    prev_key_objs = _compact_objects(objects)
                    _write_row(f, frame_num, frame, prev_key_objs)
                    if dt0 >= 0:
                        print(f"Keyframe {frame_num}: detected in {dt0:.2f}s")
                else:
                    buffer_frames.append((int(frame_num), frame))

                    # when buffer reaches keyframe_interval, the last frame is the next keyframe
                    if len(buffer_frames) >= keyframe_interval:
                        next_key_idx, next_key_frame = buffer_frames[-1]
                        print(f"Keyframe {next_key_idx}: running detection...")
                        try:
                            t1 = time.time()
                            next_objs_raw = detector.detect(next_key_frame)
                            dt1 = time.time() - t1
                        except Exception as e:
                            next_objs_raw = []
                            dt1 = -1.0
                            err_count += 1
                            if err_count <= 3:
                                print(f"WARN: detect failed at frame={next_key_idx}: {e}")
                        next_key_objs = _compact_objects(next_objs_raw)

                        if dt1 >= 0:
                            print(f"Keyframe {next_key_idx}: detected in {dt1:.2f}s")

                        matches = _greedy_match_by_iou(prev_key_objs, next_key_objs, float(args.interp_iou_thresh))

                        # write intermediate frames with interpolation
                        denom = max(1, next_key_idx - int(prev_key_idx))
                        for mid_idx, mid_frame in buffer_frames[:-1]:
                            alpha = float((mid_idx - int(prev_key_idx)) / float(denom))
                            mid_objs = _interpolate_objects(prev_key_objs, next_key_objs, alpha, matches, hold_prev_unmatched=True)
                            _write_row(f, mid_idx, mid_frame, mid_objs)

                        # write keyframe itself
                        _write_row(f, next_key_idx, next_key_frame, next_key_objs)

                        prev_key_idx = int(next_key_idx)
                        prev_key_objs = next_key_objs
                        buffer_frames = []

            frame_num += 1

        # flush tail frames if any (no next keyframe available)
        if keyframe_interval > 1 and buffer_frames and prev_key_idx is not None:
            for tail_idx, tail_frame in buffer_frames:
                _write_row(f, int(tail_idx), tail_frame, _compact_objects(prev_key_objs))

    cap.release()
    if box_video_writer is not None:
        box_video_writer.release()

    detector_stats: Dict[str, Any] = {}
    try:
        detector_stats = detector.get_stats()
    except Exception:
        detector_stats = {}

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "video": {
                    "path": str(video_path),
                    "fps": int(fps),
                    "total_frames": int(total_frames),
                    "duration": float(duration),
                },
                "detector_stats": detector_stats,
                "detections_jsonl": str(detections_path.name),
                "box_vis_video": str(box_video_path.name) if bool(args.save_box_video) else "",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 70)
    print("DONE: Step1 完成")
    print(f"frames={frame_num}")
    print(f"detections={detections_path}")
    print(f"meta={meta_path}")
    if bool(args.save_box_video):
        print(f"box_video={box_video_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
