# Track-aligned diagnostic (not official VidVRD metrics)

- Track matching on annotated Gold frames: IoU >= 0.30
- Relation matching: tIoU >= 0.50
- Evaluator: `diagnostic_track_aligned_v2`
- Tracks: 4/4 matched, mean Gold-supported IoU=0.7620
- Gold-frame MOTA=0.8933, MOTP=0.7854, ID switches=0

## Overall

TP=7, FP=175, FN=8, Precision=0.0385, Recall=0.4667, F1=0.0711
Diagnostic micro AP=0.0964, diagnostic predicate macro AP=0.0863
Diagnostic recall@50=0.4667, recall@100=0.4667
Diagnostic tagging P@1=1.0000, P@5=0.3000, P@10=0.1806

## Base / novel

| Split | TP | FP | FN | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 7 | 120 | 5 | 0.0551 | 0.5833 | 0.1007 | 0.1205 |
| novel | 0 | 55 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Per predicate

| Predicate | Split | TP | FP | FN | P | R | F1 | AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| above | novel | 0 | 27 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| beneath | novel | 0 | 27 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| jump_beneath | base | 0 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| left | base | 0 | 16 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| next_to | base | 0 | 54 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ride | base | 5 | 22 | 0 | 0.1852 | 1.0000 | 0.3125 | 0.2216 |
| right | base | 0 | 16 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sit_above | base | 2 | 11 | 3 | 0.1538 | 0.4000 | 0.2222 | 0.1238 |
| stand_above | novel | 0 | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| touch | base | 0 | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| walk_beneath | novel | 0 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
