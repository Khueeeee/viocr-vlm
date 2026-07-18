from __future__ import annotations

from typing import Any

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)


def load_florence_processor(
    model_name: str,
    trust_remote_code: bool = True,
) -> Any:
    """
    Load processor của Florence-2.
    """

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    if not hasattr(processor, "tokenizer"):
        raise RuntimeError(
            "Florence processor không chứa tokenizer."
        )

    return processor


def load_florence_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool = True,
    training: bool = False,
) -> torch.nn.Module:
    """
    Load Florence-2 base model và đưa lên device.
    """

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
    )

    model.to(device)

    if training:
        model.train()
    else:
        model.eval()

    return model


def find_matching_lora_modules(
    model: torch.nn.Module,
    requested_module_names: tuple[str, ...],
) -> list[str]:
    """
    Tìm các tên module LoRA thực sự tồn tại trong model.

    Hàm này giúp phát hiện sớm trường hợp tên target module
    không khớp với kiến trúc Florence-2.
    """

    available_suffixes: set[str] = set()

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = module_name.split(".")[-1]
            available_suffixes.add(suffix)

    matched_modules = [
        module_name
        for module_name in requested_module_names
        if module_name in available_suffixes
    ]

    if not matched_modules:
        available_text = ", ".join(
            sorted(available_suffixes)
        )

        raise RuntimeError(
            "Không tìm thấy target module LoRA phù hợp.\n"
            f"Requested: {requested_module_names}\n"
            f"Linear module suffixes hiện có: {available_text}"
        )

    return matched_modules


def apply_lora(
    model: torch.nn.Module,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: tuple[str, ...],
) -> PeftModel:
    """
    Gắn LoRA adapter vào Florence-2.
    """

    matched_modules = find_matching_lora_modules(
        model=model,
        requested_module_names=target_modules,
    )

    print()
    print("=" * 70)
    print("LORA TARGET MODULES")
    print("=" * 70)

    for module_name in matched_modules:
        print(f"- {module_name}")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=matched_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(
        model,
        lora_config,
    )

    return peft_model


def load_florence_model_with_lora(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: tuple[str, ...],
    trust_remote_code: bool = True,
) -> PeftModel:
    """
    Load Florence-2 rồi gắn LoRA adapter.
    """

    base_model = load_florence_model(
        model_name=model_name,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        training=True,
    )

    model = apply_lora(
        model=base_model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )

    model.to(device)
    model.train()

    return model


def count_parameters(
    model: torch.nn.Module,
) -> dict[str, int | float]:
    """
    Đếm tổng, trainable và frozen parameters.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = (
        total_parameters - trainable_parameters
    )

    trainable_percentage = (
        100.0 * trainable_parameters / total_parameters
        if total_parameters > 0
        else 0.0
    )

    return {
        "total": total_parameters,
        "trainable": trainable_parameters,
        "frozen": frozen_parameters,
        "trainable_percentage": trainable_percentage,
    }


def print_parameter_summary(
    model: torch.nn.Module,
) -> None:
    """
    In thống kê parameter.
    """

    information = count_parameters(model)

    print()
    print("=" * 70)
    print("MODEL PARAMETERS")
    print("=" * 70)

    print(
        f"Total parameters: "
        f"{information['total']:,}"
    )

    print(
        f"Trainable parameters: "
        f"{information['trainable']:,}"
    )

    print(
        f"Frozen parameters: "
        f"{information['frozen']:,}"
    )

    print(
        "Trainable percentage: "
        f"{information['trainable_percentage']:.6f}%"
    )


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
    pixel_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """
    Đưa batch tensor lên device.
    """

    required_fields = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "labels",
    }

    missing_fields = required_fields - set(batch.keys())

    if missing_fields:
        raise ValueError(
            f"Batch thiếu field: {sorted(missing_fields)}"
        )

    return {
        "input_ids": batch["input_ids"].to(
            device=device,
            non_blocking=True,
        ),
        "attention_mask": batch[
            "attention_mask"
        ].to(
            device=device,
            non_blocking=True,
        ),
        "pixel_values": batch[
            "pixel_values"
        ].to(
            device=device,
            dtype=pixel_dtype,
            non_blocking=True,
        ),
        "labels": batch["labels"].to(
            device=device,
            non_blocking=True,
        ),
    }