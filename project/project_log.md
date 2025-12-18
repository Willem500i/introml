# Project Log (Willem)

## Dec 2-4
- Looked for parking violation labels. BDD100K only has vehicles/curbs, no “illegal” tags. Roboflow had an illegal-parking set (positives only).
- Tried other sources (Kartaview, random street sets) but no usable violation labels. Decided to scope to binary “illegal parking vs background.”

## Dec 7-9
- Built negatives from BDD100K: cropped car/truck/bus with padding, resized 640×640, empty labels → background.
- Merged Roboflow positives + BDD negatives into balanced splits (train 1226, val 150, test 80). YOLOv8 default aug (flip/jitter/mosaic) kept framing consistent.

## Dec 12-14
- Early YOLOv8n runs overfired without proper negatives; fixed merge and added custom background-aware confusion script (to see TN/FP/FN).
- 1-epoch balanced sanity checks: recall was low; needed a few epochs to warm up.

## Dec 15-17
- 5-epoch run (final for now): val (150 imgs, 75/75) → P 0.996, R 0.987, mAP50 0.994, mAP50-95 0.782; confusion w/ background (conf=0.25): TP 74, FP 0, FN 1, TN 75.
- Exported charts/examples to `presentation_assets/` (results, confusion, PR/precision curves, val preds/labels) and made a ticket/evidence view (`ticket_example_ep5.png`).

## Notes / What’s left
- Strong metrics likely benefit from the car-centered framing; wider-scene negatives and condition-specific aug (night/rain) would help deployment.
- Single class only; real violation types need labeled data (hydrant, crosswalk, signage).
- Could try a slightly larger model (yolov8s) later if latency allows; data quality is the real limiter now.

