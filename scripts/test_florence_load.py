from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL_ID = "microsoft/Florence-2-base-ft"


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    print(f"Device: {device}")
    print(f"Dtype:  {dtype}")
    print(f"Model:  {MODEL_ID}")

    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    model = model.to(device)
    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Florence-2 loaded successfully.")
    print(f"Processor:  {type(processor).__name__}")
    print(f"Model class: {type(model).__name__}")
    print(f"Parameters: {parameter_count:,}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)

        print(f"CUDA allocated: {allocated:.2f} GB")
        print(f"CUDA reserved:  {reserved:.2f} GB")


if __name__ == "__main__":
    main()