# Cleanup Audit

This document records which historical files are part of the OpenClaw-first runtime, which files are retained as manual tooling or data assets, and which files are archived.

## Runtime Owner

The production runtime is:

- `src/vidvrd_auto/`
- `configs/`
- `skills/vidvrd-full-auto/`
- `scripts/run_vidvrd_auto.py`

New automatic-labeling features should be implemented under `src/vidvrd_auto/`.

## Historical Directories

| Path | Current Role | Integrated Into Runtime | Action | Delete Risk |
|---|---|---:|---|---|
| `legacy/old_vidvrd_snapshot/` | Old VIDVRD code snapshot | No | Archived from `大创会议/VIDVRD/` | Low to medium |
| `legacy/semi_auto_prototypes/` | Early semi-auto prototypes | Partially | Archived from `大创会议/半自动标注代码/` | Low to medium |
| `legacy/third_party_labelme/` | Vendored LabelMe source | No | Archived from `大创会议/人工标注-轨迹标注/task/labelme-5.8.0/` | Medium |
| `tools/manual_annotation/process_frame.py` | One-off frame rename helper | No | Copied from the historical relation-labeling folder | Low |
| `tools/manual_annotation/anno_platform.py` | Manual relation annotation GUI | No, but still useful | Copied from the historical relation-labeling folder | High if deleted |
| `tools/manual_annotation/trajectory_completion/` | Manual trajectory interpolation for Gold | No, still useful | Copied from `大创会议/人工标注-插帧/code/` | Medium to high |
| `data/manual_samples/add_VidVRD/` | Example/manual relation data | No | Copied from the historical relation-labeling folder | High if unbacked |
| `data/manual_samples/traj_vis_7_7/` | Example/manual trajectory labels | No | Copied from the historical track-labeling folder | High if unbacked |

## Integrated Historical Ideas

- First-frame or keyframe filtering is implemented by `src/vidvrd_auto/nodes/screen.py`.
- Rex-Omni/DINO-X detection is invoked through `src/vidvrd_auto/nodes/detect.py`.
- Tracking and window creation is invoked through `src/vidvrd_auto/nodes/track.py`.
- Semi-auto relation prompting is invoked through `src/vidvrd_auto/nodes/relation_llm.py`.
- Geometry rules, coupling, conflict detection, and export are implemented in `src/vidvrd_auto/relations/` and `src/vidvrd_auto/nodes/export.py`.

## Remaining Feature Gaps

- Batch VGGSound audio prior integration.
- Vision-language keyframe screening with keep/drop/crop suggestions.
- Explicit track quality-control node.
- Global relation aggregation and dynamic relation review.
- Strong-model second-pass verification for high-risk relations.

## Cleanup Policy

Do not delete data-like directories until they are copied to `data/manual_samples/` or `gold/raw/` and referenced from documentation. Archive code-like historical directories under `legacy/` first; remove only after the team confirms no active workflow imports or executes them.
