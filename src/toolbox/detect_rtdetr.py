import argparse
import json

import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

MODEL_ID = "PekingU/rtdetr_r18vd"

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("--threshold", type=float, default=0.5)
args = parser.parse_args()

image = Image.open(args.input).convert("RGB")

processor = RTDetrImageProcessor.from_pretrained(MODEL_ID)
model = RTDetrForObjectDetection.from_pretrained(MODEL_ID).eval()

inputs = processor(images=image, return_tensors="pt")
with torch.inference_mode():
    outputs = model(**inputs)

result = processor.post_process_object_detection(
    outputs,
    target_sizes=torch.tensor([(image.height, image.width)]),
    threshold=args.threshold,
)[0]

detections = []
for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
    class_id = int(label.item())
    detections.append({
        "bbox": box.tolist(),
        "class_id": class_id,
        "label": model.config.id2label[class_id],
        "confidence": float(score.item()),
    })

with open(args.output, "w") as f:
    json.dump(detections, f)
