"""Object detection models, video execution, and temporal fusion."""

from vidvrd_auto.detection.rex import RexDetector
from vidvrd_auto.detection.video import detect_video

__all__ = ["RexDetector", "detect_video"]
