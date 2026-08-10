from __future__ import annotations

"""Thin adapter for the official MASA-R50 plug-and-play model."""

from pathlib import Path
from typing import Any

import numpy as np

from vidvrd_auto.utils.paths import repo_root


class MasaAppearanceEncoder:
    """Extract one normalized MASA embedding for each supplied detection box.

    MASA remains an external research dependency.  This adapter calls its
    public ``init_masa``/``inference_masa`` API and captures the official
    track-head features before MASA's own tracker consumes them.
    """

    def __init__(
        self,
        *,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda:0",
        fp16: bool = True,
    ) -> None:
        import torch
        from masa.apis import build_test_pipeline, inference_masa, init_masa

        config_path = Path(config_path).expanduser()
        checkpoint_path = Path(checkpoint_path).expanduser()
        config_path = config_path if config_path.is_absolute() else repo_root() / config_path
        checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else repo_root() / checkpoint_path
        self._torch = torch
        self._inference_masa = inference_masa
        self.model = init_masa(str(config_path), str(checkpoint_path), device=device)
        self.pipeline = build_test_pipeline(self.model.cfg)
        self.fp16 = bool(fp16)
        self._labels: dict[str, int] = {}

    @staticmethod
    def _association_score(detection: dict[str, Any]) -> float:
        value = detection.get("score")
        if value is None:
            value = detection.get("confidence")
        # MASA requires a numeric detector weight. When the detector exposes no
        # native score, use a uniform association weight without exporting it
        # as confidence.
        return 1.0 if value is None else float(value)

    def encode(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        *,
        frame_num: int,
        video_len: int,
    ) -> np.ndarray:
        if not detections:
            return np.empty((0, 0), dtype=np.float32)

        boxes = [
            list(map(float, detection["bbox"])) + [self._association_score(detection)]
            for detection in detections
        ]
        labels = []
        for detection in detections:
            name = str(detection.get("class_name", "unknown")).strip().lower() or "unknown"
            labels.append(self._labels.setdefault(name, len(self._labels)))

        torch = self._torch
        device = next(self.model.parameters()).device
        det_bboxes = torch.tensor(boxes, dtype=torch.float32, device=device)
        det_labels = torch.tensor(labels, dtype=torch.long, device=device)
        captured: list[Any] = []
        original_predict = self.model.track_head.predict

        def capture_predict(*args: Any, **kwargs: Any) -> Any:
            features = original_predict(*args, **kwargs)
            captured.append(features.detach())
            return features

        self.model.track_head.predict = capture_predict
        try:
            with torch.no_grad():
                self._inference_masa(
                    self.model,
                    frame,
                    frame_id=int(frame_num),
                    video_len=int(video_len),
                    test_pipeline=self.pipeline,
                    det_bboxes=det_bboxes,
                    det_labels=det_labels,
                    fp16=self.fp16,
                )
        finally:
            self.model.track_head.predict = original_predict

        features = captured[0].float()
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy().astype(np.float32, copy=False)
