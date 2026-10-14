import argparse
import json
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
args = parser.parse_args()

model = YOLO("yolo11n.pt")
results = model.predict(args.input, verbose=False)

detections = []
for box in results[0].boxes:
    detections.append({
        "bbox": box.xyxy[0].tolist(),
        "class_id": int(box.cls.item()),
        "confidence": float(box.conf.item()),
    })

with open(args.output, "w") as f:
    json.dump(detections, f)

