import json
import os
from datasets import load_dataset

ds = load_dataset("mapo80/DocCornerDataset", split="train")
os.makedirs("images", exist_ok=True)
labels = {}

for i, row in enumerate(ds):
  if row["is_negative"]:  # Skip images without a rectangle
    continue

  img = row["image"]
  w, h = img.size
  filename = f"img_{i}.jpg"
  img.save(f"images/{filename}")

  # Convert normalized (0-1) corners back to pixel coordinates [x, y]
  labels[filename] = [
      [row["corner_tl_x"] * w, row["corner_tl_y"] * h],
      [row["corner_tr_x"] * w, row["corner_tr_y"] * h],
      [row["corner_br_x"] * w, row["corner_br_y"] * h],
      [row["corner_bl_x"] * w, row["corner_bl_y"] * h],
  ]

with open("labels.json", "w") as f:
  json.dump(labels, f, indent=4)
