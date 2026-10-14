import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
args = parser.parse_args()

image = Image.open(args.input).convert("RGB")
image.thumbnail((640, 640))
image.save(args.output)

