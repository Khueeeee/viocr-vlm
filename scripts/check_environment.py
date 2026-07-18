import platform
import sys

import torch
import transformers


def main():
    print("=" * 60)
    print("ViOCR-VLM Environment")
    print("=" * 60)

    print(f"Python        : {sys.version}")
    print(f"Platform      : {platform.platform()}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Transformers  : {transformers.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA Runtime  : {torch.version.cuda}")
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()