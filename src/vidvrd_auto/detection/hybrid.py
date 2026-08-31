from __future__ import annotations

"""Fixed-interval DINO-X keyframes with Rex-Omni between them."""

from typing import Any, Dict, List, Sequence

import numpy as np


class HybridDetector:
    def __init__(self, *, rex: Any, dinox: Any, dinox_interval: int = 15) -> None:
        self.rex = rex
        self.dinox = dinox
        self.dinox_interval = max(1, int(dinox_interval))
        self._sources: Dict[int, str] = {}
        self._dinox_fallbacks = 0

    def load_model(self) -> None:
        self.dinox.load_model()
        self.rex.load_model()

    def detect_batch(self, frames_bgr: Sequence[np.ndarray]) -> List[List[Dict[str, Any]]]:
        return self.rex.detect_batch(frames_bgr)

    def detect_batch_indexed(
        self,
        frames_bgr: Sequence[np.ndarray],
        *,
        frame_indices: Sequence[int],
    ) -> List[List[Dict[str, Any]]]:
        frames = list(frames_bgr)
        indices = [int(value) for value in frame_indices]
        if len(frames) != len(indices):
            raise ValueError("frames_bgr and frame_indices must have equal length")

        output: List[List[Dict[str, Any]]] = [[] for _ in frames]
        rex_positions: List[int] = []
        for position, (frame, frame_index) in enumerate(zip(frames, indices)):
            if frame_index % self.dinox_interval != 0:
                rex_positions.append(position)
                continue
            try:
                output[position] = self.dinox.detect_batch([frame])[0]
                self._sources[frame_index] = "dinox"
            except Exception:
                rex_positions.append(position)
                self._sources[frame_index] = "rexomni_fallback"
                self._dinox_fallbacks += 1

        if rex_positions:
            rex_results = self.rex.detect_batch([frames[position] for position in rex_positions])
            for position, detected in zip(rex_positions, rex_results):
                output[position] = detected
                frame_index = indices[position]
                self._sources.setdefault(frame_index, "rexomni")
        return output

    def source_for_frame(self, frame_index: int) -> str:
        return self._sources.get(int(frame_index), "rexomni")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend": "hybrid_rex_dinox",
            "dinox_interval": self.dinox_interval,
            "dinox_fallbacks": self._dinox_fallbacks,
            "rex": self.rex.get_stats(),
            "dinox": self.dinox.get_stats(),
        }
