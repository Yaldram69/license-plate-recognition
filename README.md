# License Plate Recognition

Detects license plates in video with a YOLOv8 model fine-tuned on a single class (`license_plate`) and reads the text of each detected plate with PaddleOCR. The repository includes the training and test scripts, base YOLO weights, and a sample input and output video.

## Features

- Dataset preparation: copies images and labels into a YOLO-format folder and writes `data.yaml` (`train.py`).
- Fine-tuning YOLOv8n for one class, 20 epochs, image size 640, batch size 16, mixed precision when on GPU (`train.py`).
- Per-frame plate detection and OCR, drawing each box and recognized text onto the video (`Test.py`).

## Project structure

- `Train.py` - builds `yolo_dataset/` from `images/` and `labels/` folders and trains YOLOv8n. Saves output under `runs/train/license_plate_detection`.
- `Test.py` - runs the trained detector and PaddleOCR on a video and writes an annotated MP4.
- `README.txt` - pip commands and run notes.
- `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolo11n.pt` - base pretrained YOLO weights. `yolov8n.pt` is the starting point used by `Train.py`.
- `e.mp4` - sample input video.
- `output_videoD.mp4` - sample annotated output video.
- `yolo_dataset/` - not included (see Notes).

## Requirements

Imports seen in the code: `ultralytics`, `torch`, `cv2` (opencv-python), `paddleocr`, `yaml` (PyYAML), `shutil`, `os`.

```bash
pip install ultralytics torch opencv-python paddlepaddle paddleocr pyyaml
```

## Installation

```bash
git clone <this-repo-url>
cd license-plate-recognition
pip install ultralytics torch opencv-python paddlepaddle paddleocr pyyaml
```

## Usage

```bash
# 1. Prepare the dataset and train (Train.py reads ../LicensePlate/images and ../LicensePlate/labels)
python Train.py

# 2. Run detection and OCR on a video (see Notes for the paths to fix first)
python Test.py
```

`Train.py` uses `DATASET_PATH = "../LicensePlate"`, so it expects to run from a sibling folder of a `LicensePlate` folder that contains `images/{train,val,test}` and `labels/{train,val,test}`.

## Notes

- **Excluded: `yolo_dataset/`** (about 24 GB). It is the output of `Train.py`: `train/`, `val/` and `test/`, each with `images/` and `labels/`, plus `data.yaml`. The `train` split alone has about 51,000 images. The `data.yaml` in the original copy has absolute Windows paths, which `Train.py` regenerates. To reproduce it you need the raw `images/` and `labels/` folders (YOLO-format labels, one class `0`); those are not in this repository.
- **Excluded: `.idea/`**, the PyCharm project folder.
- The fine-tuned model is written to `runs/train/license_plate_detection/weights/best.pt` by `Train.py`. No fine-tuned checkpoint is included. `Test.py` loads `./runs/train/license_plate_Final/weights/best.pt`, which does not match the output name in `Train.py`; rename the path or the training `name` so they agree.
- `Test.py` as written has bugs: it uses `frames_dir` and `frame_count` without defining them (a `NameError` on the first frame), reads `./a.mp4` while the included file is `e.mp4`, and writes `output_videoA.mp4`. Define the frame output folder and counter, or remove the frame-saving lines, before running.
- `Test.py` calls `PaddleOCR(use_gpu=...)`, which is an older PaddleOCR argument; newer versions may need a different device option.
- OCR is applied to every detected box in every frame, which is slow on CPU.

## Pretrained YOLO weights (not included)

The following off-the-shelf Ultralytics checkpoints were removed from this repository because they are publicly downloadable. The code loads them by name, and Ultralytics downloads them automatically on first use:

- `yolo11n.pt`
- `yolov8m.pt`
- `yolov8n.pt`
- `yolov8s.pt`

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # downloaded automatically if not present
```

Or download them manually from https://github.com/ultralytics/assets/releases
