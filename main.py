# PYTHONPATH=. pixi run -e default python main.py

import json
import tempfile
from pathlib import Path

from src.runner import run_tool


def detect_and_annotate(
    image_path: str | Path,
    output_path: str | Path,
) -> list[dict]:
    image_path = Path(image_path).resolve()
    output_path = Path(output_path).resolve()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        preprocessed = tmp / "preprocessed.jpg"
        detections_file = tmp / "detections.json"

        # Step 1: Pillow environment
        run_tool(
            "preprocess",
            image_path,
            preprocessed,
        )

        # Step 2: PyTorch + YOLO environment
        run_tool(
            "detect",
            preprocessed,
            detections_file,
        )

        # Step 3: OpenCV environment
        run_tool(
            "annotate",
            preprocessed,
            detections_file,
            output_path,
        )

        return json.loads(
            detections_file.read_text()
        )


if __name__ == "__main__":
    detections = detect_and_annotate(
        image_path="input.jpg",
        output_path="annotated.jpg",
    )
    print(f"Found {len(detections)} objects")

