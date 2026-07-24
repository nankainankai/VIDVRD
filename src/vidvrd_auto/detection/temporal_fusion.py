from __future__ import annotations

"""Utilities for auditable multi-frame detector batches.

The functions in this module do not load a model.  They define the contract
between a batched detector and the frame-wise JSONL consumed by tracking.
"""

from typing import Any, Dict, Iterable, List, Sequence


def annotate_batch_detections(
    detections_by_frame: Sequence[Iterable[Dict[str, Any]]],
    *,
    frame_indices: Sequence[int],
    batch_id: int,
    source: str,
) -> List[List[Dict[str, Any]]]:
    """Attach provenance without changing box/class values."""

    if len(detections_by_frame) != len(frame_indices):
        raise ValueError("detections_by_frame and frame_indices must have equal length")
    if len(set(int(x) for x in frame_indices)) != len(frame_indices):
        raise ValueError("a temporal batch must contain distinct frame indices")

    indices = [int(x) for x in frame_indices]
    output: List[List[Dict[str, Any]]] = []
    for frame_detections in detections_by_frame:
        annotated: List[Dict[str, Any]] = []
        for detection in frame_detections:
            item = dict(detection)
            item.setdefault("source", str(source))
            item["batch_id"] = int(batch_id)
            item["batch_frame_indices"] = list(indices)
            annotated.append(item)
        output.append(annotated)
    return output


def make_frame_batches(frame_indices: Sequence[int], batch_size: int = 5) -> List[List[int]]:
    """Split ordered frame indices into distinct, non-overlapping batches."""

    size = max(1, int(batch_size))
    indices = [int(x) for x in frame_indices]
    return [indices[pos : pos + size] for pos in range(0, len(indices), size)]
