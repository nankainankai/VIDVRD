from __future__ import annotations

"""Convert official ImageNet-VidVRD annotations into project Gold files."""

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from vidvrd_auto.core.ontology import normalize_object, predicate_splits
from vidvrd_auto.core.schema import serialize_relation_artifact
from vidvrd_auto.utils.io import read_json, write_json


def _annotation_files(root: Path) -> Iterable[Tuple[str, Path]]:
    for split_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(split_dir.glob("*.json")):
            yield split_dir.name, path


def _bbox(item: Dict[str, Any]) -> List[float] | None:
    value = item.get("bbox", {})
    if not isinstance(value, dict):
        return None
    try:
        return [float(value["xmin"]), float(value["ymin"]), float(value["xmax"]), float(value["ymax"])]
    except (KeyError, TypeError, ValueError):
        return None


def build_gold(*, annotations_dir: Path, relations_path: Path, trajectories_path: Path, manifest_path: Path) -> Dict[str, Any]:
    predicate_split = predicate_splits()
    relations: Dict[str, List[Dict[str, Any]]] = {}
    trajectories: Dict[str, List[Dict[str, Any]]] = {}
    video_splits: Dict[str, str] = {}
    relation_counts: Counter[str] = Counter()

    for split, path in _annotation_files(annotations_dir):
        obj = read_json(path)
        if not isinstance(obj, dict):
            continue
        video_id = str(obj.get("video_id", path.stem))
        video_splits[video_id] = split
        categories = {
            int(item["tid"]): normalize_object(str(item.get("category", "unknown")))
            for item in obj.get("subject/objects", [])
            if isinstance(item, dict) and item.get("tid") is not None
        }
        by_track: Dict[int, Dict[str, Any]] = {}
        for frame, frame_items in enumerate(obj.get("trajectories", [])):
            for item in frame_items if isinstance(frame_items, list) else []:
                if not isinstance(item, dict) or item.get("tid") is None:
                    continue
                track_id = int(item["tid"])
                box = _bbox(item)
                if box is None:
                    continue
                track = by_track.setdefault(
                    track_id,
                    {"track_id": track_id, "category": categories.get(track_id, "unknown"), "trajectory": {}},
                )
                track["trajectory"][str(frame)] = box
        trajectories[video_id] = [by_track[key] for key in sorted(by_track)]

        video_relations: List[Dict[str, Any]] = []
        for relation in obj.get("relation_instances", []):
            if not isinstance(relation, dict):
                continue
            try:
                predicate = str(relation["predicate"]).strip().lower()
                start = int(relation["begin_fid"])
                end = int(relation["end_fid"]) - 1
                subject_id = int(relation["subject_tid"])
                object_id = int(relation["object_tid"])
            except (KeyError, TypeError, ValueError):
                continue
            if predicate not in predicate_split or end < start:
                continue
            video_relations.append(
                serialize_relation_artifact({
                    "subject_track_id": subject_id,
                    "predicate": predicate,
                    "object_track_id": object_id,
                    "start_frame": start,
                    "end_frame": end,
                    "predicate_split": predicate_split[predicate],
                    "source": "vidvrd_gold",
                })
            )
            relation_counts[predicate] += 1
        relations[video_id] = video_relations

    write_json(relations_path, relations)
    write_json(trajectories_path, trajectories)
    manifest = {
        "annotations_dir": str(annotations_dir),
        "video_count": len(video_splits),
        "trajectory_count": sum(len(items) for items in trajectories.values()),
        "relation_count": sum(len(items) for items in relations.values()),
        "predicate_count": len(relation_counts),
        "video_splits": video_splits,
        "predicate_counts": dict(sorted(relation_counts.items())),
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build project Gold files from official VidVRD annotations")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("gold"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_gold(
        annotations_dir=args.annotations,
        relations_path=args.out_dir / "vidvrd_50_relations.json",
        trajectories_path=args.out_dir / "vidvrd_50_trajectories.json",
        manifest_path=args.out_dir / "vidvrd_50_manifest.json",
    )
    print(f"videos={manifest['video_count']} trajectories={manifest['trajectory_count']} relations={manifest['relation_count']}")


if __name__ == "__main__":
    main()
