# Official-compatible ImageNet-VidVRD evaluation

- Evaluator: `imagenet_vidvrd_official_2017_compatible_v1`
- Dataset split: `test`
- Evaluation scope: `gold_split` (10 videos)
- Official test annotations complete: `false`
- Full-test submission complete: `false`
- Tube vIoU threshold: 0.50

> This is a protocol-compatible partial result, not a publishable full-test benchmark.

> Only a subset of the Gold videos was requested. The official-scope mAP includes every Gold video, including those with no predictions.

## Relation detection

- mAP: 0.000000
- Recall@50: 0.000000
- Recall@100: 0.000000

## Relation tagging

- Precision@1: 0.000000
- Precision@5: 0.040000
- Precision@10: 0.020000

## Submission coverage

- Video set exactly matches evaluation scope: `false`
- Missing prediction videos: 7
- Extra prediction videos: 0
- Explicitly empty prediction videos: 8
- Ground-truth relations: 156
- Adapted predicted relations: 83

## Partial-run diagnostic (not an official benchmark score)

- Requested Gold videos: 3/10
- Coverage fraction: 0.300000
- mAP on requested Gold videos only: 0.000000

## Semantic API coverage

- Attempted evidence packets: 22
- Omitted packets after provider image moderation: 0
- Videos with omitted packets: 0
- Missing semantic logs: 0

## Optimistic track ceiling (diagnostic, not mAP)

- Relations even potentially matchable at tube vIoU 0.50: 12/156
- Category absent: 138
- Same-category track coverage insufficient: 6
- Actual true positives at any rank: 0
- Potentially matchable but lacking the correct triplet: 11
- Of those, predicate was routed to the semantic model: 8
- Potentially matchable but failing tube localization: 1

> This is an optimistic upper bound from detected tracks, not achieved relation recall. It ignores prediction-only tube volume, track-pair consistency and predicate errors.

## Error diagnosis

- Atomic predicate candidate recall: 0.090909
- Composite predicate candidate recall: 0.000000
- Candidate missing: 154
- Localization failures: 2
- Ranking failures: 0
- Relations involving risk tracks: 0

## Candidate routing diagnosis (predicted-track aligned; not official mAP)

- Routed candidate recall: 0.060811
- Semantic-eligible ground-truth relations: 148
- Track pair unaligned: 138
- Rule-only spatial predicates: 2
- No temporal pair packet: 5
- Predicate absent from routed packet: 2
- Predicate present in routed packet: 9
- Top categories in relations with unmatched tracks: zebra (126), airplane (42), person (35), bicycle (21), dog (17), skateboard (12), antelope (10), fox (10)
