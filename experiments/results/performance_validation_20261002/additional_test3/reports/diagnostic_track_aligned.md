# Track-aligned diagnostic (not official VidVRD metrics)

- Track matching on annotated Gold frames: IoU >= 0.30
- Relation matching: tIoU >= 0.50
- Evaluator: `diagnostic_track_aligned_v2`
- Tracks: 4/31 matched, mean Gold-supported IoU=0.8051
- Gold-frame MOTA=0.1087, MOTP=0.9017, ID switches=0

## Overall

TP=0, FP=81, FN=156, Precision=0.0000, Recall=0.0000, F1=0.0000
Diagnostic micro AP=0.0000, diagnostic predicate macro AP=0.0000
Diagnostic recall@50=0.0000, recall@100=0.0000
Diagnostic tagging P@1=0.0000, P@5=0.2500, P@10=0.2500

## Base / novel

| Split | TP | FP | FN | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0 | 45 | 133 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| novel | 0 | 36 | 23 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Per predicate

| Predicate | Split | TP | FP | FN | P | R | F1 | AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| above | novel | 0 | 17 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| away | novel | 0 | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| behind | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| beneath | novel | 0 | 17 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bite | novel | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| chase | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fly_past | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fly_with | base | 0 | 0 | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| follow | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| front | base | 0 | 1 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| larger | base | 0 | 2 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| left | base | 0 | 17 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lie_behind | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lie_front | base | 0 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lie_left | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lie_right | base | 0 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| move_behind | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| move_beneath | base | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| move_left | base | 0 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| move_right | base | 0 | 1 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| move_with | base | 0 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ride | base | 0 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| right | base | 0 | 17 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| run_above | novel | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| run_behind | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| run_beneath | novel | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| run_front | base | 0 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| run_left | base | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| run_right | base | 0 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sit_above | base | 0 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sit_left | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sit_right | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stand_above | novel | 0 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stand_behind | base | 0 | 0 | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stand_front | base | 0 | 0 | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stand_left | base | 0 | 0 | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stand_next_to | base | 0 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stand_right | base | 0 | 0 | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stand_with | novel | 0 | 0 | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| taller | base | 0 | 1 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| toward | novel | 0 | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| walk_above | novel | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| walk_away | novel | 0 | 0 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| walk_behind | base | 0 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| walk_front | base | 0 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| walk_left | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| walk_right | base | 0 | 6 | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| watch | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
