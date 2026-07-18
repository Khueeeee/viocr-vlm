from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.collator import (  # noqa: E402
    FlorenceOCRCollator,
    decode_labels,
)
from src.config import get_default_config  # noqa: E402
from src.dataset import VinTextFlorenceDataset  # noqa: E402
from src.model import (  # noqa: E402
    load_florence_processor,
)
from src.utils import (  # noqa: E402
    print_device_information,
    print_tensor_information,
    set_seed,
)


def main() -> None:
    print("=" * 70)
    print("TEST PROJECT MODULES")
    print("=" * 70)

    config = get_default_config()

    set_seed(config.seed)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model: {config.model_name}")
    print(f"Train JSONL: {config.train_jsonl}")
    print(f"Validation JSONL: {config.val_jsonl}")
    print(f"Data root: {config.data_root}")
    print(f"Device: {config.device}")
    print(f"Model dtype: {config.model_dtype}")

    print_device_information(config.device)

    print()
    print("Loading processor...")

    processor = load_florence_processor(
        model_name=config.model_name,
        trust_remote_code=config.trust_remote_code,
    )

    print("Processor loaded successfully.")

    print()
    print("Loading train dataset...")

    train_dataset = VinTextFlorenceDataset(
        jsonl_path=config.train_jsonl,
        data_root=config.data_root,
        default_prompt=config.default_prompt,
        validate_image_paths=False,
    )

    print("Train dataset loaded.")

    print()
    print("Dataset summary:")

    for key, value in train_dataset.summary().items():
        print(f"{key}: {value}")

    print()
    print("First sample metadata:")
    print(train_dataset.get_metadata(0))

    first_sample = train_dataset[0]

    print()
    print("First loaded sample:")
    print(f"Image path: {first_sample['image_path']}")
    print(f"Image mode: {first_sample['image'].mode}")
    print(f"Image size: {first_sample['image'].size}")
    print(f"Prompt: {first_sample['prompt']!r}")
    print(f"Target: {first_sample['target']!r}")

    collator = FlorenceOCRCollator(
        processor=processor,
        max_target_length=config.max_target_length,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            config.pin_memory
            and config.device.type == "cuda"
        ),
        collate_fn=collator,
    )

    print()
    print("Reading first batch...")

    batch = next(iter(train_loader))

    print()
    print("=" * 70)
    print("BATCH INFORMATION")
    print("=" * 70)

    print_tensor_information(
        "input_ids",
        batch["input_ids"],
    )

    print_tensor_information(
        "attention_mask",
        batch["attention_mask"],
    )

    print_tensor_information(
        "pixel_values",
        batch["pixel_values"],
    )

    print_tensor_information(
        "labels",
        batch["labels"],
    )

    decoded_targets = decode_labels(
        processor=processor,
        labels=batch["labels"],
    )

    print()
    print("Batch samples:")

    for index, (
        original_target,
        decoded_target,
        image_path,
    ) in enumerate(
        zip(
            batch["targets"],
            decoded_targets,
            batch["image_paths"],
        ),
        start=1,
    ):
        print()
        print(f"Sample {index}:")
        print(f"  Image: {image_path}")
        print(f"  Prompt: {batch['prompts'][index - 1]!r}")
        print(f"  Original target: {original_target!r}")
        print(f"  Decoded target: {decoded_target!r}")

        if original_target != decoded_target:
            raise RuntimeError(
                "Target sau khi encode/decode không khớp. "
                f"Original={original_target!r}, "
                f"decoded={decoded_target!r}"
            )

    if batch["pixel_values"].shape[0] != 2:
        raise RuntimeError(
            "Batch size thực tế không bằng 2."
        )

    if not torch.isfinite(
        batch["pixel_values"]
    ).all():
        raise RuntimeError(
            "pixel_values chứa NaN hoặc Infinity."
        )

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)
    print("config.py hoạt động.")
    print("dataset.py hoạt động.")
    print("collator.py hoạt động.")
    print("model.py hoạt động.")
    print("utils.py hoạt động.")
    print("Các module đã sẵn sàng cho train.py.")


if __name__ == "__main__":
    main()