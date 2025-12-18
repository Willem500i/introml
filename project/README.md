# Illegal Parking Detector (YOLOv8: Roboflow positives + BDD negatives)

Binary detector for “illegal parking” using Roboflow positives and BDD100K car crops as negatives. Includes trained weights, the 150-image val split used for evaluation, and presentation/ticket assets.

## What’s included
- Trained weights: `results_illegal_parking/yolov8_illegal_ep5_balanced/weights/best.pt`
- Val split (150 imgs + labels): `testing_data/full_val/{images,labels}/`
- Presentation/ticket assets: `presentation_assets/`, `used_images/`
- Ticket generator: `generate_ticket_example.py`
- Eval utilities: `confusion_with_background.py`, `export_detections.py`
- Dataset prep: `make_negative_car_crops.py`, `prepare_illegal_parking_dataset.py`

## Quick setup
```bash
cd project
pip install -r requirements.txt
```

## Quick validate (no retraining)
Run YOLOv8 val on the included 150-image split.
```bash
cd project
yolo detect val \
  model=results_illegal_parking/yolov8_illegal_ep5_balanced/weights/best.pt \
  data=test_data.yaml \
  imgsz=640 batch=16
```
Outputs go to `runs/detect/val*`.

## Rebuild dataset (optional)
```bash
cd project
python make_negative_car_crops.py \
  --bdd-root data/100k \
  --output data/negative_crops_cars

python prepare_illegal_parking_dataset.py \
  --roboflow-dir "data/illegal parking.v1i.yolov8" \
  --bdd-negative-dir data/negative_crops_cars \
  --output-dir data/illegal_parking_merged \
  --train-neg 800 --val-neg 150 --test-neg 150 \
  --val-pos 150
```
- Positives: Roboflow illegal parking set.
- Negatives: car-centered crops from BDD100K (empty labels = background).

## Train (optional)
```bash
cd project
yolo detect train \
  data=data/illegal_parking_merged/data.yaml \
  model=yolov8n.pt \
  epochs=5 imgsz=640 batch=16 \
  project=results_illegal_parking name=yolov8_illegal_ep5_balanced
```
Adjust `epochs` (e.g., 20) if you want a longer run.

## Background-aware confusion (optional)
```bash
cd project
python confusion_with_background.py \
  --pred runs/detect/val*/predictions.json \
  --labels testing_data/full_val/labels \
  --conf 0.25 \
  --output presentation_assets/confusion_with_bg_custom.png
```

## Ticketing example
```bash
cd project
python generate_ticket_example.py \
  --run-dir results_illegal_parking/yolov8_illegal_ep5_balanced_val \
  --data test_data.yaml \
  --split val \
  --conf 0.25 \
  --output presentation_assets/ticket_example_ep5.png
```

## Directory guide (committed artifacts)
- `results_illegal_parking/` – trained weights/logs (ep5 run).
- `testing_data/full_val/` – 150-image val set with labels.
- `testing_images/` – qualitative positives/negatives.
- `used_images/` – figures used in the deck.
- `presentation_assets/` – charts/visuals.
- `test_data.yaml` – points YOLO to the committed val set.

## Notes
- Only `project/data/` stays gitignored; evaluation assets above are tracked.
- Empty labels are intentional for negatives to teach the detector background.

