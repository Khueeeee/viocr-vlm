from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VINTEXT_DIR = PROJECT_ROOT / "data" / "raw" / "vintext"
FLORENCE_DIR = PROJECT_ROOT / "data" / "florence"

PADDING = 3
MIN_CROP_WIDTH = 2
MIN_CROP_HEIGHT = 2

SPLITS = {
    "train": VINTEXT_DIR / "train_vintext_label.txt",
    "val": VINTEXT_DIR / "val_vintext_label.txt",
}


def parse_label_line(line: str) -> tuple[str, list[dict]]:
    """
    Đọc một dòng VinText.

    Ví dụ:

    train_images/im0498.jpg
    [{'transcription': 'SỞ', 'points': [[...], ...]}]
    """

    line = line.strip()

    if not line:
        raise ValueError("Dòng label rỗng.")

    annotation_start = line.find("[{")

    if annotation_start == -1:
        raise ValueError(
            "Không tìm thấy phần bắt đầu annotation '[{'."
        )

    image_relative_path = line[:annotation_start].strip()
    annotation_text = line[annotation_start:].strip()

    if not image_relative_path:
        raise ValueError("Không đọc được đường dẫn ảnh.")

    try:
        annotations = ast.literal_eval(annotation_text)
    except (SyntaxError, ValueError) as error:
        raise ValueError(
            "Không thể parse danh sách annotation."
        ) from error

    if not isinstance(annotations, list):
        raise TypeError(
            "Annotations phải là list, "
            f"nhưng nhận được {type(annotations).__name__}."
        )

    return image_relative_path, annotations


def crop_polygon_bbox(
    image: np.ndarray,
    points: list[list[int | float]],
    padding: int,
) -> np.ndarray | None:
    """
    Crop bằng bounding box bao quanh polygon bốn điểm.
    """

    try:
        polygon = np.asarray(points, dtype=np.float32)
    except (TypeError, ValueError):
        return None

    if polygon.shape != (4, 2):
        return None

    if not np.isfinite(polygon).all():
        return None

    image_height, image_width = image.shape[:2]

    x_min = int(np.floor(polygon[:, 0].min())) - padding
    y_min = int(np.floor(polygon[:, 1].min())) - padding
    x_max = int(np.ceil(polygon[:, 0].max())) + padding
    y_max = int(np.ceil(polygon[:, 1].max())) + padding

    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, image_width)
    y_max = min(y_max, image_height)

    if x_max <= x_min or y_max <= y_min:
        return None

    crop = image[y_min:y_max, x_min:x_max]

    if crop.size == 0:
        return None

    crop_height, crop_width = crop.shape[:2]

    if (
        crop_width < MIN_CROP_WIDTH
        or crop_height < MIN_CROP_HEIGHT
    ):
        return None

    return crop


def reset_split_output(split_name: str) -> tuple[Path, Path]:
    """
    Xóa output cũ của một split rồi tạo lại.
    """

    split_dir = FLORENCE_DIR / split_name
    images_dir = split_dir / "images"
    jsonl_path = FLORENCE_DIR / f"{split_name}.jsonl"

    if split_dir.exists():
        shutil.rmtree(split_dir)

    if jsonl_path.exists():
        jsonl_path.unlink()

    images_dir.mkdir(parents=True, exist_ok=True)

    return images_dir, jsonl_path


def process_split(
    split_name: str,
    label_file: Path,
) -> dict[str, int]:
    """
    Chuyển một split VinText sang dữ liệu OCR của Florence-2.
    """

    print()
    print("=" * 70)
    print(f"PROCESS SPLIT: {split_name.upper()}")
    print("=" * 70)
    print(f"Label file: {label_file}")

    if not label_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file label: {label_file}"
        )

    images_dir, jsonl_path = reset_split_output(split_name)

    counters = {
        "label_lines": 0,
        "source_images": 0,
        "saved_crops": 0,
        "ignored_text": 0,
        "invalid_annotations": 0,
        "missing_images": 0,
        "unreadable_images": 0,
        "parse_errors": 0,
    }

    with (
        label_file.open("r", encoding="utf-8") as label_reader,
        jsonl_path.open("w", encoding="utf-8") as jsonl_writer,
    ):
        for line_number, line in enumerate(
            label_reader,
            start=1,
        ):
            counters["label_lines"] += 1

            if not line.strip():
                continue

            try:
                image_relative_path, annotations = (
                    parse_label_line(line)
                )
            except (ValueError, TypeError) as error:
                counters["parse_errors"] += 1

                print(
                    f"[PARSE ERROR] line={line_number}: {error}"
                )
                continue

            source_image_path = (
                VINTEXT_DIR / image_relative_path
            )

            if not source_image_path.exists():
                counters["missing_images"] += 1

                print(
                    "[MISSING IMAGE] "
                    f"line={line_number} "
                    f"path={source_image_path}"
                )
                continue

            image = cv2.imread(str(source_image_path))

            if image is None:
                counters["unreadable_images"] += 1

                print(
                    "[UNREADABLE IMAGE] "
                    f"line={line_number} "
                    f"path={source_image_path}"
                )
                continue

            counters["source_images"] += 1

            for annotation in annotations:
                if not isinstance(annotation, dict):
                    counters["invalid_annotations"] += 1
                    continue

                transcription = str(
                    annotation.get("transcription", "")
                ).strip()

                points = annotation.get("points")

                # ### là vùng chữ không đọc được trong VinText.
                if (
                    not transcription
                    or transcription == "###"
                ):
                    counters["ignored_text"] += 1
                    continue

                if not isinstance(points, list):
                    counters["invalid_annotations"] += 1
                    continue

                crop = crop_polygon_bbox(
                    image=image,
                    points=points,
                    padding=PADDING,
                )

                if crop is None:
                    counters["invalid_annotations"] += 1
                    continue

                counters["saved_crops"] += 1
                sample_id = counters["saved_crops"]

                output_name = f"{sample_id:08d}.jpg"
                output_path = images_dir / output_name

                saved = cv2.imwrite(
                    str(output_path),
                    crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )

                if not saved:
                    counters["invalid_annotations"] += 1
                    counters["saved_crops"] -= 1

                    print(
                        "[SAVE ERROR] "
                        f"path={output_path}"
                    )
                    continue

                record = {
                    "image": (
                        f"{split_name}/images/{output_name}"
                    ),
                    "prompt": "<OCR>",
                    "target": transcription,
                }

                jsonl_writer.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if counters["source_images"] % 100 == 0:
                print(
                    f"Processed images: "
                    f"{counters['source_images']} | "
                    f"Saved crops: "
                    f"{counters['saved_crops']}"
                )

    print()
    print("-" * 70)
    print(f"SUMMARY: {split_name.upper()}")
    print("-" * 70)

    for key, value in counters.items():
        print(f"{key}: {value}")

    print(f"Images directory: {images_dir}")
    print(f"JSONL file: {jsonl_path}")

    return counters


def show_jsonl_examples(
    jsonl_path: Path,
    number_of_examples: int = 3,
) -> None:
    """
    In một vài dòng JSONL để kiểm tra Unicode tiếng Việt.
    """

    print()
    print("=" * 70)
    print(f"JSONL EXAMPLES: {jsonl_path.name}")
    print("=" * 70)

    with jsonl_path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file, start=1):
            print(line.rstrip())

            if index >= number_of_examples:
                break


def main() -> None:
    print("=" * 70)
    print("PREPARE VINTEXT FOR FLORENCE-2")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"VinText directory: {VINTEXT_DIR}")
    print(f"Florence directory: {FLORENCE_DIR}")
    print(f"Padding: {PADDING}")

    FLORENCE_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, int]] = {}

    for split_name, label_file in SPLITS.items():
        result = process_split(
            split_name=split_name,
            label_file=label_file,
        )

        all_results[split_name] = result

    for split_name in SPLITS:
        show_jsonl_examples(
            FLORENCE_DIR / f"{split_name}.jsonl"
        )

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    for split_name, result in all_results.items():
        print(
            f"{split_name}: "
            f"{result['saved_crops']} samples"
        )

    print()
    print("SUCCESS")
    print("Đã tạo dataset Florence-2 cho train và validation.")


if __name__ == "__main__":
    main()