from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.collator import FlorenceOCRCollator  # noqa: E402
from src.config import get_default_config  # noqa: E402
from src.dataset import VinTextFlorenceDataset  # noqa: E402
from src.model import (  # noqa: E402
    load_florence_model_with_lora,
    load_florence_processor,
    move_batch_to_device,
    print_parameter_summary,
)
from src.utils import (  # noqa: E402
    print_device_information,
    print_gpu_memory,
    set_seed,
)


TEST_LEARNING_RATE = 1e-4
TEST_BATCH_SIZE = 1


def calculate_gradient_norm(
    model: torch.nn.Module,
) -> float:
    """
    Tính L2 norm của toàn bộ gradient trainable.
    """

    squared_norm = 0.0

    for parameter in model.parameters():
        if (
            parameter.requires_grad
            and parameter.grad is not None
        ):
            gradient_norm = parameter.grad.detach().norm(2)

            squared_norm += gradient_norm.item() ** 2

    return squared_norm ** 0.5


def main() -> None:
    print("=" * 70)
    print("TEST FLORENCE-2 LORA TRAIN STEP")
    print("=" * 70)

    config = get_default_config()

    set_seed(config.seed)

    device = config.device
    dtype = config.model_dtype

    print(f"Model: {config.model_name}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"LoRA rank: {config.lora_rank}")
    print(f"LoRA alpha: {config.lora_alpha}")
    print(f"LoRA dropout: {config.lora_dropout}")
    print(
        f"Requested target modules: "
        f"{config.lora_target_modules}"
    )

    print_device_information(device)

    print()
    print("Loading processor...")

    processor = load_florence_processor(
        model_name=config.model_name,
        trust_remote_code=config.trust_remote_code,
    )

    print("Processor loaded.")

    print()
    print("Loading dataset...")

    dataset = VinTextFlorenceDataset(
        jsonl_path=config.train_jsonl,
        data_root=config.data_root,
        default_prompt=config.default_prompt,
    )

    collator = FlorenceOCRCollator(
        processor=processor,
        max_target_length=config.max_target_length,
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    print(f"Dataset samples: {len(dataset)}")

    print()
    print("Loading Florence-2 and applying LoRA...")

    model = load_florence_model_with_lora(
        model_name=config.model_name,
        device=device,
        dtype=dtype,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        trust_remote_code=config.trust_remote_code,
    )

    print("Model and LoRA loaded.")

    print_parameter_summary(model)

    optimizer = AdamW(
        params=[
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=TEST_LEARNING_RATE,
        weight_decay=config.weight_decay,
    )

    print()
    print("Reading one batch...")

    batch = next(iter(data_loader))

    print(f"Image: {batch['image_paths'][0]}")
    print(f"Target: {batch['targets'][0]!r}")

    model_batch = move_batch_to_device(
        batch=batch,
        device=device,
        pixel_dtype=dtype,
    )

    optimizer.zero_grad(set_to_none=True)

    print_gpu_memory()

    print()
    print("Running forward pass...")

    outputs = model(
        input_ids=model_batch["input_ids"],
        attention_mask=model_batch["attention_mask"],
        pixel_values=model_batch["pixel_values"],
        labels=model_batch["labels"],
    )

    loss = outputs.loss

    print(f"Loss before backward: {loss.item():.6f}")

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Loss không hữu hạn: {loss.item()}"
        )

    print()
    print("Running backward pass...")

    loss.backward()

    gradient_norm = calculate_gradient_norm(model)

    print(f"Gradient norm: {gradient_norm:.6f}")

    if gradient_norm <= 0.0:
        raise RuntimeError(
            "Gradient norm bằng 0. "
            "LoRA adapter có thể chưa nhận gradient."
        )

    if not torch.isfinite(
        torch.tensor(gradient_norm)
    ):
        raise RuntimeError(
            "Gradient norm là NaN hoặc Infinity."
        )

    clipped_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=config.max_grad_norm,
    )

    print(
        f"Gradient norm before clipping: "
        f"{float(clipped_norm):.6f}"
    )

    print()
    print("Running optimizer step...")

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    print_gpu_memory()

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)
    print("Florence-2 đã load.")
    print("LoRA đã được gắn vào model.")
    print("Forward pass thành công.")
    print("Backward pass thành công.")
    print("LoRA parameters đã nhận gradient.")
    print("Optimizer step thành công.")
    print("Model đã sẵn sàng cho training loop hoàn chỉnh.")


if __name__ == "__main__":
    main()