# Track-aligned diagnostic (not official VidVRD metrics)

- Track matching on annotated Gold frames: IoU >= 0.30
- Relation matching: tIoU >= 0.50
- Evaluator: `diagnostic_track_aligned_v2`
- Tracks: 8/10 matched, mean Gold-supported IoU=0.7490
- Gold-frame MOTA=0.6286, MOTP=0.8429, ID switches=0

## Overall

TP=1, FP=208, FN=28, Precision=0.0048, Recall=0.0345, F1=0.0084
Diagnostic micro AP=0.0034, diagnostic predicate macro AP=0.0417
Diagnostic recall@50=0.0345, recall@100=0.0345
Diagnostic tagging P@1=0.2500, P@5=0.2750, P@10=0.2062

## Base / novel

| Split | TP | FP | FN | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0 | 128 | 26 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| novel | 1 | 80 | 2 | 0.0123 | 0.3333 | 0.0238 | 0.0833 |

## Per predicate

| Predicate | Split | TP | FP | FN | P | R | F1 | AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| above | novel | 0 | 36 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| away | novel | 0 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| behind | base | 0 | 4 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| beneath | novel | 0 | 36 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bite | novel | 1 | 2 | 0 | 0.3333 | 1.0000 | 0.5000 | 1.0000 |
| chase | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| follow | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| front | base | 0 | 2 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hold | novel | 0 | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| larger | base | 0 | 6 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| left | base | 0 | 37 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lie_behind | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lie_front | base | 0 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lie_left | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lie_right | base | 0 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| move_beneath | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| move_left | base | 0 | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| move_right | base | 0 | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| next_to | base | 0 | 18 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ride | base | 0 | 8 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| right | base | 0 | 37 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| run_above | novel | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| run_behind | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| run_beneath | novel | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| run_left | base | 0 | 1 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| run_right | base | 0 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sit_above | base | 0 | 3 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stand_behind | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stand_left | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| taller | base | 0 | 3 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| touch | base | 0 | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| toward | novel | 0 | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| walk_right | base | 0 | 6 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| watch | base | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
