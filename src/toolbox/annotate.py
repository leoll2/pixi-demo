import argparse
import json
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("detections")
parser.add_argument("output")
args = parser.parse_args()

image = cv2.imread(args.image)

with open(args.detections) as f:
    detections = json.load(f)

for detection in detections:
    x1, y1, x2, y2 = map(int, detection["bbox"])
    cv2.rectangle(
        image, (x1, y1), (x2, y2),
        (0, 255, 0), 2
    )

if not cv2.imwrite(args.output, image):
    raise RuntimeError("Failed to save annotated image")

