from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "florence"
TRAIN_JSONL_PATH = DATA_ROOT / "train.jsonl"

MODEL_NAME = "microsoft/Florence-2-large-ft"

BATCH_SIZE = 1
NUM_WORKERS = 0
MAX_TARGET_LENGTH = 64


# ============================================================
# DATASET
# ============================================================

class VinTextFlorenceDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path,
        data_root: Path,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.data_root = data_root

        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy JSONL: {jsonl_path}"
            )

        self.samples: list[dict[str, str]] = []

        with jsonl_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSON lỗi tại dòng {line_number}"
                    ) from error

                for required_key in (
                    "image",
                    "prompt",
                    "target",
                ):
                    if required_key not in record:
                        raise ValueError(
                            f"Dòng {line_number} thiếu "
                            f"field {required_key!r}"
                        )

                self.samples.append(
                    {
                        "image": str(record["image"]),
                        "prompt": str(record["prompt"]),
                        "target": str(record["target"]),
                    }
                )

        if not self.samples:
            raise RuntimeError(
                f"Dataset rỗng: {jsonl_path}"
            )

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

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

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
    def __init__(
        self,
        processor: Any,
        max_target_length: int,
    ) -> None:
        self.processor = processor
        self.max_target_length = max_target_length

    def __call__(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        images = [
            sample["image"]
            for sample in samples
        ]

        prompts = [
            sample["prompt"]
            for sample in samples
        ]

        targets = [
            sample["target"]
            for sample in samples
        ]

        model_inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        target_tokens = self.processor.tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
        )

        labels = target_tokens["input_ids"].clone()

        pad_token_id = (
            self.processor.tokenizer.pad_token_id
        )

        labels[labels == pad_token_id] = -100

        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs[
                "attention_mask"
            ],
            "pixel_values": model_inputs[
                "pixel_values"
            ],
            "labels": labels,
            "targets": targets,
            "image_paths": [
                sample["image_path"]
                for sample in samples
            ],
        }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def select_dtype(
    device: torch.device,
) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16

    return torch.float32


def print_gpu_information() -> None:
    print()
    print("=" * 70)
    print("DEVICE INFORMATION")
    print("=" * 70)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        total_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3
        )

        print(f"GPU memory: {total_memory:.2f} GB")


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    model_batch = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch[
            "attention_mask"
        ].to(device),
        "pixel_values": batch[
            "pixel_values"
        ].to(
            device=device,
            dtype=dtype,
        ),
        "labels": batch["labels"].to(device),
    }

    return model_batch


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 70)
    print("FLORENCE-2 FORWARD PASS TEST")
    print("=" * 70)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model: {MODEL_NAME}")
    print(f"JSONL: {TRAIN_JSONL_PATH}")
    print(f"Batch size: {BATCH_SIZE}")

    device = select_device()
    dtype = select_dtype(device)

    print_gpu_information()

    print()
    print(f"Selected device: {device}")
    print(f"Selected dtype: {dtype}")

    print()
    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    print("Processor loaded.")

    print()
    print("Loading dataset...")

    dataset = VinTextFlorenceDataset(
        jsonl_path=TRAIN_JSONL_PATH,
        data_root=DATA_ROOT,
    )

    collator = FlorenceOCRCollator(
        processor=processor,
        max_target_length=MAX_TARGET_LENGTH,
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    print(f"Dataset samples: {len(dataset)}")

    print()
    print("Reading first batch...")

    batch = next(iter(data_loader))

    print(f"Image: {batch['image_paths'][0]}")
    print(f"Target: {batch['targets'][0]!r}")

    print()
    print("Loading Florence-2 model...")
    print(
        "Lần đầu có thể mất thời gian vì phải tải trọng số."
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    model_batch = move_batch_to_device(
        batch=batch,
        device=device,
        dtype=dtype,
    )

    print()
    print("Tensor shapes:")

    for key, tensor in model_batch.items():
        print(
            f"{key}: "
            f"shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, "
            f"device={tensor.device}"
        )

    print()
    print("Running forward pass...")

    with torch.no_grad():
        outputs = model(
            input_ids=model_batch["input_ids"],
            attention_mask=model_batch[
                "attention_mask"
            ],
            pixel_values=model_batch[
                "pixel_values"
            ],
            labels=model_batch["labels"],
        )

    loss = outputs.loss

    print()
    print("=" * 70)
    print("FORWARD RESULT")
    print("=" * 70)

    print(f"Loss tensor: {loss}")
    print(f"Loss value: {loss.item():.6f}")
    print(
        f"Logits shape: "
        f"{tuple(outputs.logits.shape)}"
    )

    if not torch.isfinite(loss):
        raise RuntimeError(
            "Loss là NaN hoặc Infinity."
        )

    if loss.item() <= 0:
        raise RuntimeError(
            f"Loss không hợp lệ: {loss.item()}"
        )

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)
    print("Florence-2 đã load thành công.")
    print("Batch đã được đưa vào model thành công.")
    print("Model đã tính được loss hữu hạn.")
    print("Pipeline đã sẵn sàng để xây dựng training script.")


if __name__ == "__main__":
    main()