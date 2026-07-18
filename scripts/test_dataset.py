from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FLORENCE_DATA_DIR = PROJECT_ROOT / "data" / "florence"

TRAIN_JSONL_PATH = FLORENCE_DATA_DIR / "train.jsonl"
VAL_JSONL_PATH = FLORENCE_DATA_DIR / "val.jsonl"

MODEL_NAME = "microsoft/Florence-2-large-ft"

BATCH_SIZE = 2
NUM_WORKERS = 0

MAX_TARGET_LENGTH = 128


# ============================================================
# DATASET
# ============================================================

class VinTextFlorenceDataset(Dataset):
    """
    PyTorch Dataset dùng cho dữ liệu VinText đã được crop.

    Mỗi dòng JSONL có dạng:

    {
        "image": "train/images/00000001.jpg",
        "prompt": "<OCR>",
        "target": "SỞ"
    }

    Dataset chỉ thực hiện:

    1. Đọc metadata từ JSONL.
    2. Đọc ảnh bằng PIL.
    3. Chuyển ảnh sang RGB.
    4. Trả về image, prompt và target.

    Việc tokenizer và xử lý ảnh được thực hiện trong collate_fn.
    """

    def __init__(
        self,
        jsonl_path: Path,
        data_root: Path,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.data_root = data_root

        if not self.jsonl_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy JSONL: {self.jsonl_path}"
            )

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục dữ liệu: {self.data_root}"
            )

        self.samples = self._load_jsonl()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Không có sample nào trong: {self.jsonl_path}"
            )

    def _load_jsonl(self) -> list[dict[str, str]]:
        """
        Đọc toàn bộ JSONL và kiểm tra cấu trúc từng sample.
        """

        samples: list[dict[str, str]] = []

        with self.jsonl_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "JSON không hợp lệ tại "
                        f"{self.jsonl_path}, dòng {line_number}."
                    ) from error

                required_fields = {
                    "image",
                    "prompt",
                    "target",
                }

                missing_fields = (
                    required_fields - set(record.keys())
                )

                if missing_fields:
                    raise ValueError(
                        f"Thiếu field tại dòng {line_number}: "
                        f"{sorted(missing_fields)}"
                    )

                image_relative_path = str(
                    record["image"]
                ).strip()

                prompt = str(record["prompt"]).strip()
                target = str(record["target"]).strip()

                if not image_relative_path:
                    raise ValueError(
                        f"Đường dẫn ảnh rỗng tại dòng {line_number}."
                    )

                if not prompt:
                    raise ValueError(
                        f"Prompt rỗng tại dòng {line_number}."
                    )

                if not target:
                    raise ValueError(
                        f"Target rỗng tại dòng {line_number}."
                    )

                samples.append(
                    {
                        "image": image_relative_path,
                        "prompt": prompt,
                        "target": target,
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        sample = self.samples[index]

        image_path = self.data_root / sample["image"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy ảnh: {image_path}"
            )

        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
        except Exception as error:
            raise RuntimeError(
                f"Không thể đọc ảnh: {image_path}"
            ) from error

        return {
            "image": image,
            "prompt": sample["prompt"],
            "target": sample["target"],
            "image_path": str(image_path),
        }


# ============================================================
# COLLATOR
# ============================================================

class FlorenceOCRCollator:
    """
    Chuyển danh sách sample thành một batch tensor.

    Processor xử lý:

    - images -> pixel_values
    - prompts -> input_ids và attention_mask
    - targets -> labels
    """

    def __init__(
        self,
        processor: AutoProcessor,
        max_target_length: int = 128,
    ) -> None:
        self.processor = processor
        self.max_target_length = max_target_length

    def __call__(
        self,
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:
        images = [
            sample["image"]
            for sample in batch
        ]

        prompts = [
            sample["prompt"]
            for sample in batch
        ]

        targets = [
            sample["target"]
            for sample in batch
        ]

        image_paths = [
            sample["image_path"]
            for sample in batch
        ]

        # Xử lý đồng thời ảnh và prompt.
        model_inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Tokenize target OCR.
        target_tokens = self.processor.tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
        )

        labels = target_tokens["input_ids"]

        # Padding token không được tham gia vào loss.
        labels = labels.clone()

        labels[
            labels == self.processor.tokenizer.pad_token_id
        ] = -100

        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "pixel_values": model_inputs["pixel_values"],
            "labels": labels,
            "targets": targets,
            "image_paths": image_paths,
        }


# ============================================================
# CHECK FUNCTIONS
# ============================================================

def print_dataset_sample(
    dataset: VinTextFlorenceDataset,
    index: int = 0,
) -> None:
    """
    Kiểm tra một sample trước khi tạo DataLoader.
    """

    sample = dataset[index]
    image = sample["image"]

    print()
    print("=" * 70)
    print("RAW DATASET SAMPLE")
    print("=" * 70)

    print(f"Index: {index}")
    print(f"Image path: {sample['image_path']}")
    print(f"Image mode: {image.mode}")
    print(f"Image size: {image.size}")
    print(f"Prompt: {sample['prompt']!r}")
    print(f"Target: {sample['target']!r}")


def decode_labels(
    processor: AutoProcessor,
    labels: torch.Tensor,
) -> list[str]:
    """
    Chuyển labels tensor ngược lại thành text để kiểm tra.

    Các vị trí -100 được đổi lại thành pad_token_id
    trước khi decode.
    """

    labels_for_decode = labels.clone()

    labels_for_decode[labels_for_decode == -100] = (
        processor.tokenizer.pad_token_id
    )

    decoded_labels = processor.tokenizer.batch_decode(
        labels_for_decode,
        skip_special_tokens=True,
    )

    return decoded_labels


def print_batch_information(
    batch: dict[str, Any],
    processor: AutoProcessor,
) -> None:
    """
    In thông tin tensor của một batch.
    """

    print()
    print("=" * 70)
    print("DATALOADER BATCH")
    print("=" * 70)

    print(f"Batch keys: {list(batch.keys())}")

    print()
    print("Tensor shapes:")

    print(
        "input_ids:",
        tuple(batch["input_ids"].shape),
        batch["input_ids"].dtype,
    )

    print(
        "attention_mask:",
        tuple(batch["attention_mask"].shape),
        batch["attention_mask"].dtype,
    )

    print(
        "pixel_values:",
        tuple(batch["pixel_values"].shape),
        batch["pixel_values"].dtype,
    )

    print(
        "labels:",
        tuple(batch["labels"].shape),
        batch["labels"].dtype,
    )

    decoded_labels = decode_labels(
        processor=processor,
        labels=batch["labels"],
    )

    print()
    print("Sample details:")

    for index, (
        path,
        original_target,
        decoded_target,
    ) in enumerate(
        zip(
            batch["image_paths"],
            batch["targets"],
            decoded_labels,
        ),
        start=1,
    ):
        print()
        print(f"Sample {index}:")
        print(f"  Image: {path}")
        print(f"  Original target: {original_target!r}")
        print(f"  Decoded target:  {decoded_target!r}")


def validate_batch(
    batch: dict[str, Any],
) -> None:
    """
    Kiểm tra các điều kiện tối thiểu của batch.
    """

    required_keys = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "labels",
    }

    missing_keys = required_keys - set(batch.keys())

    if missing_keys:
        raise RuntimeError(
            f"Batch thiếu các key: {sorted(missing_keys)}"
        )

    batch_size = batch["input_ids"].shape[0]

    if batch_size != BATCH_SIZE:
        raise RuntimeError(
            "Batch size không đúng. "
            f"Expected={BATCH_SIZE}, actual={batch_size}"
        )

    if batch["pixel_values"].ndim != 4:
        raise RuntimeError(
            "pixel_values phải có 4 chiều "
            "[batch, channels, height, width]."
        )

    if batch["input_ids"].ndim != 2:
        raise RuntimeError(
            "input_ids phải có 2 chiều "
            "[batch, sequence_length]."
        )

    if batch["labels"].ndim != 2:
        raise RuntimeError(
            "labels phải có 2 chiều "
            "[batch, sequence_length]."
        )

    if not torch.isfinite(batch["pixel_values"]).all():
        raise RuntimeError(
            "pixel_values chứa NaN hoặc Infinity."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 70)
    print("TEST VINTEXT FLORENCE DATASET")
    print("=" * 70)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset root: {FLORENCE_DATA_DIR}")
    print(f"Train JSONL: {TRAIN_JSONL_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")

    print()
    print("Loading Florence-2 processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    print("Processor loaded successfully.")

    print()
    print("Loading train dataset...")

    train_dataset = VinTextFlorenceDataset(
        jsonl_path=TRAIN_JSONL_PATH,
        data_root=FLORENCE_DATA_DIR,
    )

    print(
        f"Train dataset loaded: "
        f"{len(train_dataset)} samples"
    )

    print_dataset_sample(
        dataset=train_dataset,
        index=0,
    )

    collator = FlorenceOCRCollator(
        processor=processor,
        max_target_length=MAX_TARGET_LENGTH,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    print()
    print("Reading first DataLoader batch...")

    first_batch = next(iter(train_loader))

    validate_batch(first_batch)

    print_batch_information(
        batch=first_batch,
        processor=processor,
    )

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)
    print("Dataset hoạt động.")
    print("DataLoader hoạt động.")
    print("Processor hoạt động.")
    print("Ảnh và nhãn đã được chuyển thành tensor thành công.")


if __name__ == "__main__":
    main()