from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "vintext"

TRAIN_JSON = DATASET_DIR / "train.json"

print("=" * 60)
print("VinText Dataset")
print("=" * 60)

print("Dataset:", DATASET_DIR)
print("Train JSON:", TRAIN_JSON)

with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

print()

print("Keys:")
print(data.keys())

print()

print("Images:", len(data["images"]))
print("Annotations:", len(data["annotations"]))
print("Categories:", len(data["categories"]))

print()
print("=" * 60)
print("First image")
print("=" * 60)
print(data["images"][0])

print()
print("=" * 60)
print("First annotation")
print("=" * 60)
print(data["annotations"][0])