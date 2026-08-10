# Source and compatibility target

This module is a Python 3 compatibility implementation of the MIT-licensed
`xdshang/VidVRD-helper` relation evaluator:

- upstream: https://github.com/xdshang/VidVRD-helper
- compatibility reference commit: `1b4de175ce6e7a103d5feaae66b68d32a306a877`
- reference files: `evaluation/visual_relation_detection.py` and
  `evaluation/common.py`
- protocol: per-video VOC AP averaged across videos, per-video top-K detection
  recall accumulated across all ground-truth instances, and mean per-video
  relation-tagging precision.

The project-specific artifact adapter is local code. It converts inclusive
project spans and track dictionaries into the upstream half-open contiguous
tube format before evaluation.
