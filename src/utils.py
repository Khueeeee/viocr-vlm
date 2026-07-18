from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Thiết lập seed để kết quả có khả năng tái lập.
    """

    if seed < 0:
        raise ValueError("Seed không được âm.")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


def print_device_information(
    device: torch.device,
) -> None:
    """
    In thông tin môi trường PyTorch và GPU.
    """

    print()
    print("=" * 70)
    print("DEVICE INFORMATION")
    print("=" * 70)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Selected device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if device.type == "cuda":
        device_index = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )

        properties = torch.cuda.get_device_properties(
            device_index
        )

        total_memory_gb = (
            properties.total_memory / 1024**3
        )

        print(
            f"GPU name: "
            f"{torch.cuda.get_device_name(device_index)}"
        )
        print(
            f"GPU memory: {total_memory_gb:.2f} GB"
        )
        print(
            f"CUDA version: {torch.version.cuda}"
        )


def print_tensor_information(
    name: str,
    tensor: torch.Tensor,
) -> None:
    """
    In shape, dtype và device của tensor.
    """

    print(
        f"{name}: "
        f"shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, "
        f"device={tensor.device}"
    )


def save_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """
    Lưu checkpoint cơ bản.

    Phần này sẽ được sử dụng khi viết training loop.
    """

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if extra_state is not None:
        state["extra_state"] = extra_state

    torch.save(state, checkpoint_path)


def get_gpu_memory_information() -> dict[str, float]:
    """
    Trả thông tin bộ nhớ GPU theo GB.
    """

    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
        }

    return {
        "allocated_gb": (
            torch.cuda.memory_allocated() / 1024**3
        ),
        "reserved_gb": (
            torch.cuda.memory_reserved() / 1024**3
        ),
        "max_allocated_gb": (
            torch.cuda.max_memory_allocated() / 1024**3
        ),
    }


def print_gpu_memory() -> None:
    """
    In mức sử dụng VRAM hiện tại.
    """

    information = get_gpu_memory_information()

    print()
    print("=" * 70)
    print("GPU MEMORY")
    print("=" * 70)
    print(
        f"Allocated: "
        f"{information['allocated_gb']:.2f} GB"
    )
    print(
        f"Reserved: "
        f"{information['reserved_gb']:.2f} GB"
    )
    print(
        f"Maximum allocated: "
        f"{information['max_allocated_gb']:.2f} GB"
    )