# Official-compatible ImageNet-VidVRD evaluation

- Evaluator: `imagenet_vidvrd_official_2017_compatible_v1`
- Dataset split: `train`
- Evaluation scope: `requested` (2 videos)
- Official test annotations complete: `false`
- Full-test submission complete: `false`
- Tube vIoU threshold: 0.50

> This is a protocol-compatible partial result, not a publishable full-test benchmark.

## Relation detection

- mAP: 0.125499
- Recall@50: 0.466667
- Recall@100: 0.466667

## Relation tagging

- Precision@1: 1.000000
- Precision@5: 0.300000
- Precision@10: 0.175000

## Submission coverage

- Video set exactly matches evaluation scope: `true`
- Missing prediction videos: 0
- Extra prediction videos: 0
- Explicitly empty prediction videos: 0
- Ground-truth relations: 15
- Adapted predicted relations: 183

## Partial-run diagnostic (not an official benchmark score)

- Requested Gold videos: 2/2
- Coverage fraction: 1.000000
- mAP on requested Gold videos only: 0.125499

## Semantic API coverage

- Attempted evidence packets: 32
- Omitted packets after provider image moderation: 0
- Videos with omitted packets: 0
- Missing semantic logs: 0

## Optimistic track ceiling (diagnostic, not mAP)

- Relations even potentially matchable at tube vIoU 0.50: 15/15
- Category absent: 0
- Same-category track coverage insufficient: 0
- Actual true positives at any rank: 7
- Potentially matchable but lacking the correct triplet: 7
- Of those, predicate was routed to the semantic model: 2
- Potentially matchable but failing tube localization: 1

> This is an optimistic upper bound from detected tracks, not achieved relation recall. It ignores prediction-only tube volume, track-pair consistency and predicate errors.

## Error diagnosis

- Atomic predicate candidate recall: 1.000000
- Composite predicate candidate recall: 0.300000
- Candidate missing: 7
- Localization failures: 1
- Ranking failures: 7
- Relations involving risk tracks: 0

## Candidate routing diagnosis (predicted-track aligned; not official mAP)

- Routed candidate recall: 0.666667
- Semantic-eligible ground-truth relations: 15
- Track pair unaligned: 0
- Rule-only spatial predicates: 0
- No temporal pair packet: 0
- Predicate absent from routed packet: 5
- Predicate present in routed packet: 10
