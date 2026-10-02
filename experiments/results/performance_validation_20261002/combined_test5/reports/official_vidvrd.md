# Official-compatible ImageNet-VidVRD evaluation

- Evaluator: `imagenet_vidvrd_official_2017_compatible_v1`
- Dataset split: `test`
- Evaluation scope: `requested` (5 videos)
- Official test annotations complete: `false`
- Full-test submission complete: `false`
- Tube vIoU threshold: 0.50

> This is a protocol-compatible partial result, not a publishable full-test benchmark.

## Relation detection

- mAP: 0.013333
- Recall@50: 0.034483
- Recall@100: 0.034483

## Relation tagging

- Precision@1: 0.200000
- Precision@5: 0.200000
- Precision@10: 0.100000

## Submission coverage

- Video set exactly matches evaluation scope: `true`
- Missing prediction videos: 0
- Extra prediction videos: 0
- Explicitly empty prediction videos: 1
- Ground-truth relations: 29
- Adapted predicted relations: 221

## Partial-run diagnostic (not an official benchmark score)

- Requested Gold videos: 5/5
- Coverage fraction: 1.000000
- mAP on requested Gold videos only: 0.013333

## Optimistic track ceiling (diagnostic, not mAP)

- Relations even potentially matchable at tube vIoU 0.50: 15/29
- Category absent: 5
- Same-category track coverage insufficient: 9
- Actual true positives at any rank: 1
- Potentially matchable but lacking the correct triplet: 13
- Of those, predicate was routed to the semantic model: 0
- Potentially matchable but failing tube localization: 1

> This is an optimistic upper bound from detected tracks, not achieved relation recall. It ignores prediction-only tube volume, track-pair consistency and predicate errors.

## Error diagnosis

- Atomic predicate candidate recall: 0.333333
- Composite predicate candidate recall: 0.058824
- Candidate missing: 24
- Localization failures: 4
- Ranking failures: 1
- Relations involving risk tracks: 0
