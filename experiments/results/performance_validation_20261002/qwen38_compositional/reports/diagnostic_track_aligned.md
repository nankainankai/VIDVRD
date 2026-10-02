# Track-aligned diagnostic (not official VidVRD metrics)

- Track matching on annotated Gold frames: IoU >= 0.30
- Relation matching: tIoU >= 0.50
- Evaluator: `diagnostic_track_aligned_v2`
- Tracks: 4/4 matched, mean Gold-supported IoU=0.7620
- Gold-frame MOTA=0.8933, MOTP=0.7854, ID switches=0

## Overall

TP=6, FP=179, FN=9, Precision=0.0324, Recall=0.4000, F1=0.0600
Diagnostic micro AP=0.1189, diagnostic predicate macro AP=0.0997
Diagnostic recall@50=0.4000, recall@100=0.4000
Diagnostic tagging P@1=1.0000, P@5=0.2000, P@10=0.1556

## Base / novel

| Split | TP | FP | FN | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 6 | 122 | 6 | 0.0469 | 0.5000 | 0.0857 | 0.1505 |
| novel | 0 | 57 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Per predicate

| Predicate | Split | TP | FP | FN | P | R | F1 | AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| above | novel | 0 | 27 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| beneath | novel | 0 | 27 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| faster | base | 0 | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| jump_beneath | base | 0 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| larger | base | 0 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| left | base | 0 | 16 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| move_above | novel | 0 | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| next_to | base | 0 | 54 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ride | base | 5 | 22 | 0 | 0.1852 | 1.0000 | 0.3125 | 0.3489 |
| right | base | 0 | 16 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sit_above | base | 1 | 6 | 4 | 0.1429 | 0.2000 | 0.1667 | 0.0500 |
| stand_above | novel | 0 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| taller | base | 0 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| touch | base | 0 | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| walk_beneath | novel | 0 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
