# Object detection demo (Pixi multi-environment PoC)

A proof of concept for running tools with incompatible runtimes (different Python, PyTorch and NumPy versions) from one project. Each tool runs in its own [Pixi](https://pixi.sh) environment and the tools pass data to each other through files.

The pipeline in `main.py` has three steps:

1. **preprocess**: resizes the input image (Pillow, Python 3.12).
2. **detect**: one of two detectors, each in its own environment:
   - `detect-yolo`: Ultralytics YOLO (Python 3.14, PyTorch 2.12+)
   - `detect-rtdetr`: Hugging Face RT-DETR (Python 3.11, PyTorch 2.4/2.5, NumPy 1.x)
3. **annotate**: draws the bounding boxes (OpenCV).

Each tool is a Pixi task defined only in its own feature in `pixi.toml`, so `pixi run <task>` picks the right environment automatically.

## Requirements

- Linux x86-64
- [Pixi](https://pixi.sh)

## Run

Put an image at `input.jpg`, then:

```bash
PYTHONPATH=. pixi run -e default python main.py
```

The annotated image is written to `annotated.jpg`. The first run installs the environments and downloads model weights, so it takes longer.

To switch detectors, set `DETECTOR` in `main.py` to `"yolo"` or `"rtdetr"`.

You can also run a single tool directly:

```bash
pixi task list
pixi run detect-yolo input.jpg detections.json
```
