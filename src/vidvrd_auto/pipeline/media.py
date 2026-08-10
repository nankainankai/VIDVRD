from __future__ import annotations

from pathlib import Path
from typing import Tuple

from vidvrd_auto.core import AppConfig, VideoPaths
from vidvrd_auto.nodes.detect import detect_video
from vidvrd_auto.nodes.ingest import materialize_video, planned_video_path, source_fingerprint
from vidvrd_auto.nodes.track import track_video
from vidvrd_auto.nodes.track_qc import run_track_qc
from vidvrd_auto.nodes.vocabulary import build_vocabulary
from vidvrd_auto.pipeline.files import Artifacts
from vidvrd_auto.pipeline.stage import StageRunner
from vidvrd_auto.utils.hashing import sha256_file
from vidvrd_auto.utils.io import read_json


def run_media(
    *,
    source: str,
    video_id: str,
    config: AppConfig,
    api_key: str,
    paths: VideoPaths,
    files: Artifacts,
    stages: StageRunner,
) -> Tuple[Path, str]:
    """Run ingest through track QC and return video path plus hash."""

    video_path = planned_video_path(source, paths.inputs_dir)
    ingest_cfg = config.section("video_ingest").to_dict()
    input_identity = source_fingerprint(source, paths.inputs_dir)
    stages.run(
        "ingest",
        {"source": source, "source_fingerprint": input_identity, "config": ingest_cfg},
        [files.source, video_path],
        lambda: materialize_video(source, paths.inputs_dir, ingest_cfg),
    )
    if not files.source.exists() or not video_path.exists():
        materialize_video(source, paths.inputs_dir, ingest_cfg)
    actual_video_hash = sha256_file(video_path)
    source_meta = read_json(files.source)
    if str(source_meta.get("file_hash", "")) != actual_video_hash:
        source_meta = materialize_video(source, paths.inputs_dir, ingest_cfg)
    video_hash = str(source_meta.get("file_hash", "")) or actual_video_hash

    vocabulary_cfg = config.section("vocabulary").to_dict()
    stages.run(
        "vocabulary",
        {"video": video_hash, "config": vocabulary_cfg},
        [files.vocabulary],
        lambda: build_vocabulary(
            video_path=video_path,
            out_json=files.vocabulary,
            evidence_path=paths.vocabulary_dir / "evidence.jpg",
            config=vocabulary_cfg,
            api_key=api_key,
        ),
    )

    detect_cfg = config.section("detector").to_dict()
    vocabulary = read_json(files.vocabulary)
    detect_cfg["rex_categories"] = vocabulary.get("categories", [])
    detect_cfg["category_aliases"] = vocabulary.get("label_map", {})
    stages.run(
        "detect",
        {"video": video_hash, "config": detect_cfg},
        [files.detections, files.detect_meta],
        lambda: detect_video(
            video_path=video_path,
            out_dir=paths.detect_dir,
            config=detect_cfg,
            log_path=paths.detect_dir / "run.log",
        ),
    )

    track_cfg = config.section("tracking").to_dict()
    stages.run(
        "track",
        {"video": video_hash, "detections": sha256_file(files.detections), "config": track_cfg},
        [files.tracks, files.tracklets, files.stitch_links, files.windows],
        lambda: track_video(
            video_path=video_path,
            detections_path=files.detections,
            out_dir=paths.track_dir,
            config=track_cfg,
        ),
    )

    qc_cfg = config.section("track_qc").to_dict()
    stages.run(
        "track_qc",
        {"tracks": sha256_file(files.tracks), "config": qc_cfg},
        [files.track_report],
        lambda: run_track_qc(
            tracks_jsonl=files.tracks,
            windows_json=files.windows,
            out_json=files.track_report,
            config=qc_cfg,
        ),
    )
    return video_path, video_hash
