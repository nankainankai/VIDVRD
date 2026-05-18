from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import cv2  # type: ignore

    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False


def _safe_read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def _extract_video_id_from_windows(windows_json: Path) -> Tuple[str, Path]:
    obj = _safe_read_json(windows_json)
    if not isinstance(obj, dict):
        raise SystemExit(f"ERROR: windows.json is not a dict: {windows_json}")
    video = obj.get("video", {})
    if not isinstance(video, dict):
        raise SystemExit(f"ERROR: windows.json missing video field: {windows_json}")
    video_path = Path(str(video.get("path", "") or "")).expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"ERROR: video.path not found on disk: {video_path}")
    return video_path.stem, video_path


def convert_tracks_jsonl_to_anno_platform_json(
    *,
    tracks_jsonl: Path,
    video_id: str,
    out_track_json: Path,
    prefer_observed_bbox: bool,
) -> None:
    # Build: { video_id: {"anno": [ {"tid":..., "category":..., "traj_name":..., "trajectory":{frame:[x1,y1,x2,y2]}} ] } }
    tracks: Dict[int, Dict[str, Any]] = {}

    for row in _iter_jsonl(tracks_jsonl):
        try:
            frame = int(row.get("frame"))
        except Exception:
            continue
        items = row.get("tracks", [])
        if not isinstance(items, list) or not items:
            continue

        for t in items:
            if not isinstance(t, dict):
                continue
            try:
                tid = int(t.get("track_id"))
            except Exception:
                continue

            bbox_key = "bbox_observed" if prefer_observed_bbox and ("bbox_observed" in t) else "bbox"
            bbox = t.get(bbox_key)
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            except Exception:
                continue

            cls = str(t.get("class_name", "") or "").strip() or "unknown"

            if tid not in tracks:
                tracks[tid] = {
                    "tid": tid,
                    "category": cls,
                    "traj_name": f"{cls}_{tid}",
                    "trajectory": {},
                }

            # Keep first non-unknown category if available
            if tracks[tid].get("category") in ("", "unknown") and cls not in ("", "unknown"):
                tracks[tid]["category"] = cls
                tracks[tid]["traj_name"] = f"{cls}_{tid}"

            tracks[tid]["trajectory"][str(frame)] = [x1, y1, x2, y2]

    out_obj: Dict[str, Any] = {
        str(video_id): {
            "anno": [tracks[k] for k in sorted(tracks.keys())],
        }
    }

    out_track_json.parent.mkdir(parents=True, exist_ok=True)
    with out_track_json.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)


def export_video_frames(
    *,
    video_path: Path,
    out_frames_root: Path,
    video_id: str,
    jpg_quality: int,
    overwrite: bool,
    resume: bool,
    max_frames: int,
) -> Path:
    if not HAS_CV2 or cv2 is None:
        raise SystemExit("ERROR: opencv-python not installed (cv2 import failed)")

    out_dir = out_frames_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if overwrite and out_dir.exists():
        # Delete only image files to be safe
        for p in out_dir.glob("*.jpg"):
            try:
                p.unlink()
            except Exception:
                pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: failed to open video: {video_path}")

    q = int(jpg_quality)
    q = max(10, min(100, q))

    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if max_frames > 0 and fid >= int(max_frames):
            break

        out_path = out_dir / f"{fid:06d}.jpg"
        if resume and out_path.exists():
            fid += 1
            continue

        ok2 = cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok2:
            cap.release()
            raise SystemExit(f"ERROR: cv2.imwrite failed: {out_path}")

        fid += 1

    cap.release()
    return out_dir


def build_gui_command(*, data_folder: Path, track_json: Path, annotation_folder: Path) -> str:
    # We intentionally keep it simple and Windows-friendly.
    gui = Path(__file__).resolve().parents[1] / "tools" / "manual_annotation" / "anno_platform.py"
    return (
        f'python "{gui}" '
        f'--data_folder "{data_folder}" '
        f'--track_json "{track_json}" '
        f'--annotation_folder "{annotation_folder}"'
    )


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Prepare inputs for senior's anno_platform.py (export frames + convert tracks_full.jsonl)."
    )
    ap.add_argument("--windows_json", type=str, required=True, help="Step2 windows.json")
    ap.add_argument("--tracks_jsonl", type=str, required=True, help="Step2 tracks_full.jsonl")

    ap.add_argument(
        "--out_frames_root",
        type=str,
        default="",
        help="Output root directory; will create <out_frames_root>/<video_id>/*.jpg",
    )
    ap.add_argument(
        "--out_track_json",
        type=str,
        default="",
        help="Output track json path for anno_platform (default: <out_frames_root>/tracks_anno_platform.json)",
    )
    ap.add_argument(
        "--annotation_folder",
        type=str,
        default="",
        help="Where anno_platform saves annos (default: <out_frames_root>/annos)",
    )

    ap.add_argument("--jpg_quality", type=int, default=90)
    ap.add_argument("--overwrite_frames", action="store_true", help="Delete existing exported JPGs first")
    ap.add_argument("--resume_frames", action="store_true", help="Skip exporting frames that already exist")
    ap.add_argument("--max_frames", type=int, default=0, help="For debugging; 0 = all")

    ap.add_argument(
        "--prefer_observed_bbox",
        action="store_true",
        help="If tracks contain bbox_observed, prefer it over bbox",
    )
    return ap


def main() -> None:
    args = _build_parser().parse_args()

    windows_json = Path(args.windows_json).expanduser().resolve()
    tracks_jsonl = Path(args.tracks_jsonl).expanduser().resolve()

    if not windows_json.exists():
        raise SystemExit(f"ERROR: windows_json not found: {windows_json}")
    if not tracks_jsonl.exists():
        raise SystemExit(f"ERROR: tracks_jsonl not found: {tracks_jsonl}")

    video_id, video_path = _extract_video_id_from_windows(windows_json)

    out_frames_root = str(args.out_frames_root or "").strip()
    if not out_frames_root:
        out_frames_root = str((windows_json.parent / "anno_frames").resolve())
    out_frames_root_p = Path(out_frames_root).expanduser().resolve()

    out_track_json = str(args.out_track_json or "").strip()
    if not out_track_json:
        out_track_json = str((out_frames_root_p / "tracks_anno_platform.json").resolve())
    out_track_json_p = Path(out_track_json).expanduser().resolve()

    ann_folder = str(args.annotation_folder or "").strip()
    if not ann_folder:
        ann_folder = str((out_frames_root_p / "annos").resolve())
    ann_folder_p = Path(ann_folder).expanduser().resolve()
    ann_folder_p.mkdir(parents=True, exist_ok=True)

    frames_dir = export_video_frames(
        video_path=video_path,
        out_frames_root=out_frames_root_p,
        video_id=video_id,
        jpg_quality=int(args.jpg_quality),
        overwrite=bool(args.overwrite_frames),
        resume=bool(args.resume_frames),
        max_frames=int(args.max_frames),
    )

    convert_tracks_jsonl_to_anno_platform_json(
        tracks_jsonl=tracks_jsonl,
        video_id=video_id,
        out_track_json=out_track_json_p,
        prefer_observed_bbox=bool(args.prefer_observed_bbox),
    )

    print("=" * 70)
    print("DONE prepare_anno_platform_inputs")
    print(f"video_id={video_id}")
    print(f"video={video_path}")
    print(f"frames_dir={frames_dir}")
    print(f"track_json={out_track_json_p}")
    print(f"annotation_folder={ann_folder_p}")
    print("Run GUI:")
    print(build_gui_command(data_folder=out_frames_root_p, track_json=out_track_json_p, annotation_folder=ann_folder_p))
    print("=" * 70)


if __name__ == "__main__":
    main()
