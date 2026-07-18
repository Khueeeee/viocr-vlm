from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainingConfig:
    """
    Cấu hình trung tâm cho quá trình fine-tune Florence-2 trên VinText.

    File này quản lý:

    - đường dẫn dữ liệu;
    - cấu hình model;
    - DataLoader;
    - LoRA/PEFT;
    - optimizer;
    - scheduler;
    - mixed precision;
    - gradient accumulation;
    - validation;
    - checkpoint;
    - logging;
    - reproducibility.

    Tất cả đường dẫn mặc định được xây dựng từ PROJECT_ROOT nên không
    phụ thuộc vào thư mục hiện tại khi chạy script.
    """

    # ========================================================
    # MODEL
    # ========================================================

    model_name: str = "microsoft/Florence-2-large-ft"

    trust_remote_code: bool = True

    default_prompt: str = "<OCR>"

    # Số token tối đa của target OCR.
    # Với crop từng từ của VinText, 64 là đủ an toàn.
    max_target_length: int = 64

    # ========================================================
    # DATA PATHS
    # ========================================================

    data_root: Path = field(
        default_factory=lambda: (
            PROJECT_ROOT
            / "data"
            / "florence"
        )
    )

    train_jsonl: Path = field(
        default_factory=lambda: (
            PROJECT_ROOT
            / "data"
            / "florence"
            / "train.jsonl"
        )
    )

    val_jsonl: Path = field(
        default_factory=lambda: (
            PROJECT_ROOT
            / "data"
            / "florence"
            / "val.jsonl"
        )
    )

    # ========================================================
    # DATALOADER
    # ========================================================

    # RTX 3050 6 GB nên bắt đầu với batch size 1.
    train_batch_size: int = 1
    val_batch_size: int = 1

    # Windows thường ổn định nhất với 0 ở giai đoạn đầu.
    # Khi chạy trên Colab/Linux có thể tăng lên 2 hoặc 4.
    num_workers: int = 0

    pin_memory: bool = True

    # persistent_workers chỉ dùng được khi num_workers > 0.
    persistent_workers: bool = False

    # Không bỏ batch cuối.
    drop_last_train_batch: bool = False

    # ========================================================
    # LORA / PEFT
    # ========================================================

    use_lora: bool = True

    # Cấu hình mặc định cân bằng giữa bộ nhớ và khả năng học.
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Không huấn luyện bias.
    lora_bias: str = "none"

    # Các suffix module phổ biến trong Florence-2.
    # model.py sẽ kiểm tra những tên nào thực sự tồn tại.
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    )

    # ========================================================
    # TRAINING
    # ========================================================

    num_epochs: int = 5

    # LoRA thường dùng LR cao hơn full fine-tuning.
    learning_rate: float = 1e-4

    weight_decay: float = 0.01

    # Batch hiệu dụng:
    #
    # train_batch_size * gradient_accumulation_steps
    #
    # Với mặc định:
    # 1 * 8 = 8
    gradient_accumulation_steps: int = 8

    max_grad_norm: float = 1.0

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer_name: str = "adamw"

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler_name: str = "linear"

    warmup_ratio: float = 0.05

    # Nếu warmup_steps > 0, training script có thể ưu tiên giá trị này
    # thay cho warmup_ratio.
    warmup_steps: int = 0

    # ========================================================
    # MIXED PRECISION
    # ========================================================

    # RTX 3050 hỗ trợ FP16.
    use_fp16: bool = True

    # Không bật BF16 trên RTX 3050.
    use_bf16: bool = False

    # Dùng AMP khi chạy CUDA và precision phù hợp.
    use_amp: bool = True

    # GradScaler cho FP16. Scale khởi tạo thấp giúp Florence-2 ổn định
    # hơn trên GPU T4/RTX khi bắt đầu huấn luyện.
    grad_scaler_init_scale: float = 256.0
    grad_scaler_growth_factor: float = 2.0
    grad_scaler_backoff_factor: float = 0.5
    grad_scaler_growth_interval: int = 2000

    # Khi FP16 overflow, bỏ qua optimizer step đó thay vì dừng toàn bộ.
    skip_non_finite_steps: bool = True

    # ========================================================
    # MEMORY OPTIMIZATION
    # ========================================================

    # Có thể giúp giảm VRAM khi training.
    gradient_checkpointing: bool = True

    # Tắt cache decoder khi training để tương thích
    # gradient checkpointing.
    disable_model_cache_during_training: bool = True

    # Gọi empty_cache không nên quá thường xuyên.
    empty_cuda_cache_every_steps: int = 0

    # ========================================================
    # VALIDATION
    # ========================================================

    run_validation: bool = True

    # Validate sau mỗi N optimizer steps.
    validate_every_steps: int = 1000

    # Giới hạn số batch validation để test nhanh.
    # None nghĩa là chạy toàn bộ validation set.
    max_validation_batches: int | None = 500

    # ========================================================
    # LOGGING
    # ========================================================

    log_every_steps: int = 20

    log_dir: Path = field(
        default_factory=lambda: Path(
            "/content/drive/MyDrive/ViOCR/logs"
        )
    )

    # ========================================================
    # CHECKPOINTS
    # ========================================================

    output_dir: Path = field(
        default_factory=lambda: Path(
            "/content/drive/MyDrive/ViOCR/checkpoints"
        )
    )

    save_every_steps: int = 500

    save_every_epoch: bool = True

    # Số checkpoint gần nhất muốn giữ lại.
    # Việc xóa checkpoint cũ sẽ do train.py xử lý.
    max_checkpoints_to_keep: int = 3

    save_best_model: bool = True

    # Tiêu chí mặc định là validation loss.
    metric_for_best_model: str = "val_loss"

    # Với loss thì nhỏ hơn là tốt hơn.
    greater_is_better: bool = False

    # ========================================================
    # RESUME TRAINING
    # ========================================================

    resume_from_checkpoint: Path | None = Path(
    "/content/drive/MyDrive/ViOCR/checkpoints/interrupted_checkpoint"
    )

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    seed: int = 42

    deterministic: bool = False

    # ========================================================
    # DEBUG / DEVELOPMENT
    # ========================================================

    debug: bool = False

    # None nghĩa là dùng toàn bộ dữ liệu.
    max_train_samples: int | None = None
    max_val_samples: int | None = None


    # ========================================================
    # INITIALIZATION AND VALIDATION
    # ========================================================

    def __post_init__(self) -> None:
        """
        Chuẩn hóa kiểu dữ liệu và kiểm tra tính hợp lệ của cấu hình.
        """

        self.data_root = Path(self.data_root)
        self.train_jsonl = Path(self.train_jsonl)
        self.val_jsonl = Path(self.val_jsonl)
        self.output_dir = Path(self.output_dir)
        self.log_dir = Path(self.log_dir)

        if self.resume_from_checkpoint is not None:
            self.resume_from_checkpoint = Path(
                self.resume_from_checkpoint
            )

        self._validate_model_config()
        self._validate_data_config()
        self._validate_dataloader_config()
        self._validate_lora_config()
        self._validate_training_config()
        self._validate_precision_config()
        self._validate_validation_config()
        self._validate_logging_and_checkpoint_config()
        self._validate_debug_config()

    # ========================================================
    # VALIDATION HELPERS
    # ========================================================

    def _validate_model_config(self) -> None:
        if not self.model_name.strip():
            raise ValueError(
                "model_name không được rỗng."
            )

        if not self.default_prompt.strip():
            raise ValueError(
                "default_prompt không được rỗng."
            )

        if self.max_target_length <= 0:
            raise ValueError(
                "max_target_length phải lớn hơn 0."
            )

    def _validate_data_config(self) -> None:
        if not str(self.data_root):
            raise ValueError(
                "data_root không hợp lệ."
            )

        if not str(self.train_jsonl):
            raise ValueError(
                "train_jsonl không hợp lệ."
            )

        if not str(self.val_jsonl):
            raise ValueError(
                "val_jsonl không hợp lệ."
            )

    def _validate_dataloader_config(self) -> None:
        if self.train_batch_size <= 0:
            raise ValueError(
                "train_batch_size phải lớn hơn 0."
            )

        if self.val_batch_size <= 0:
            raise ValueError(
                "val_batch_size phải lớn hơn 0."
            )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers không được âm."
            )

        if (
            self.persistent_workers
            and self.num_workers == 0
        ):
            raise ValueError(
                "persistent_workers=True yêu cầu "
                "num_workers > 0."
            )

    def _validate_lora_config(self) -> None:
        if not self.use_lora:
            return

        if self.lora_rank <= 0:
            raise ValueError(
                "lora_rank phải lớn hơn 0."
            )

        if self.lora_alpha <= 0:
            raise ValueError(
                "lora_alpha phải lớn hơn 0."
            )

        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(
                "lora_dropout phải thuộc khoảng [0, 1)."
            )

        valid_bias_values = {
            "none",
            "all",
            "lora_only",
        }

        if self.lora_bias not in valid_bias_values:
            raise ValueError(
                "lora_bias phải là một trong: "
                "'none', 'all', 'lora_only'."
            )

        if not self.lora_target_modules:
            raise ValueError(
                "lora_target_modules không được rỗng."
            )

        for module_name in self.lora_target_modules:
            if not module_name.strip():
                raise ValueError(
                    "lora_target_modules chứa tên rỗng."
                )

    def _validate_training_config(self) -> None:
        if self.num_epochs <= 0:
            raise ValueError(
                "num_epochs phải lớn hơn 0."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate phải lớn hơn 0."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay không được âm."
            )

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps "
                "phải lớn hơn 0."
            )

        if self.max_grad_norm <= 0:
            raise ValueError(
                "max_grad_norm phải lớn hơn 0."
            )

        if not 0.0 <= self.adam_beta1 < 1.0:
            raise ValueError(
                "adam_beta1 phải thuộc khoảng [0, 1)."
            )

        if not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError(
                "adam_beta2 phải thuộc khoảng [0, 1)."
            )

        if self.adam_epsilon <= 0:
            raise ValueError(
                "adam_epsilon phải lớn hơn 0."
            )

        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(
                "warmup_ratio phải thuộc khoảng [0, 1)."
            )

        if self.warmup_steps < 0:
            raise ValueError(
                "warmup_steps không được âm."
            )

        if self.empty_cuda_cache_every_steps < 0:
            raise ValueError(
                "empty_cuda_cache_every_steps "
                "không được âm."
            )

    def _validate_precision_config(self) -> None:
        if self.use_fp16 and self.use_bf16:
            raise ValueError(
                "Không được bật đồng thời "
                "use_fp16 và use_bf16."
            )

        if self.grad_scaler_init_scale <= 0:
            raise ValueError(
                "grad_scaler_init_scale phải lớn hơn 0."
            )

        if self.grad_scaler_growth_factor <= 1.0:
            raise ValueError(
                "grad_scaler_growth_factor phải lớn hơn 1."
            )

        if not 0.0 < self.grad_scaler_backoff_factor < 1.0:
            raise ValueError(
                "grad_scaler_backoff_factor phải thuộc khoảng (0, 1)."
            )

        if self.grad_scaler_growth_interval <= 0:
            raise ValueError(
                "grad_scaler_growth_interval phải lớn hơn 0."
            )

    def _validate_validation_config(self) -> None:
        if (
            self.run_validation
            and self.validate_every_steps <= 0
        ):
            raise ValueError(
                "validate_every_steps phải lớn hơn 0 "
                "khi run_validation=True."
            )

        if (
            self.max_validation_batches is not None
            and self.max_validation_batches <= 0
        ):
            raise ValueError(
                "max_validation_batches phải lớn hơn 0 "
                "hoặc bằng None."
            )

    def _validate_logging_and_checkpoint_config(
        self,
    ) -> None:
        if self.log_every_steps <= 0:
            raise ValueError(
                "log_every_steps phải lớn hơn 0."
            )

        if self.save_every_steps <= 0:
            raise ValueError(
                "save_every_steps phải lớn hơn 0."
            )

        if self.max_checkpoints_to_keep <= 0:
            raise ValueError(
                "max_checkpoints_to_keep "
                "phải lớn hơn 0."
            )

        if not self.metric_for_best_model.strip():
            raise ValueError(
                "metric_for_best_model không được rỗng."
            )

    def _validate_debug_config(self) -> None:
        if (
            self.max_train_samples is not None
            and self.max_train_samples <= 0
        ):
            raise ValueError(
                "max_train_samples phải lớn hơn 0 "
                "hoặc bằng None."
            )

        if (
            self.max_val_samples is not None
            and self.max_val_samples <= 0
        ):
            raise ValueError(
                "max_val_samples phải lớn hơn 0 "
                "hoặc bằng None."
            )

    # ========================================================
    # COMPUTED PROPERTIES
    # ========================================================

    @property
    def device(self) -> torch.device:
        """
        Thiết bị mặc định của quá trình huấn luyện.
        """

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    @property
    def model_dtype(self) -> torch.dtype:
        """
        Dtype dùng khi load model và chuyển pixel_values.
        """

        if self.device.type != "cuda":
            return torch.float32

        if self.use_bf16:
            return torch.bfloat16

        if self.use_fp16:
            return torch.float16

        return torch.float32

    @property
    def amp_enabled(self) -> bool:
        """
        AMP chỉ được bật khi chạy CUDA và precision thấp được chọn.
        """

        return (
            self.use_amp
            and self.device.type == "cuda"
            and (
                self.use_fp16
                or self.use_bf16
            )
        )

    @property
    def amp_dtype(self) -> torch.dtype:
        """
        Dtype dùng cho torch.autocast.
        """

        if self.use_bf16:
            return torch.bfloat16

        return torch.float16

    @property
    def use_grad_scaler(self) -> bool:
        """
        GradScaler cần thiết cho FP16 nhưng thường không cần cho BF16.
        """

        return (
            self.amp_enabled
            and self.use_fp16
            and not self.use_bf16
        )

    @property
    def effective_train_batch_size(self) -> int:
        """
        Batch size hiệu dụng sau gradient accumulation.
        """

        return (
            self.train_batch_size
            * self.gradient_accumulation_steps
        )

    @property
    def resolved_pin_memory(self) -> bool:
        """
        pin_memory chỉ có ý nghĩa khi huấn luyện trên CUDA.
        """

        return (
            self.pin_memory
            and self.device.type == "cuda"
        )

    @property
    def resolved_persistent_workers(self) -> bool:
        """
        persistent_workers chỉ hợp lệ khi num_workers > 0.
        """

        return (
            self.persistent_workers
            and self.num_workers > 0
        )

    # ========================================================
    # PATH OPERATIONS
    # ========================================================

    def validate_paths(self) -> None:
        """
        Kiểm tra các đường dẫn dữ liệu bắt buộc.
        """

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Không tìm thấy data root: "
                f"{self.data_root}"
            )

        if not self.data_root.is_dir():
            raise NotADirectoryError(
                f"data_root không phải thư mục: "
                f"{self.data_root}"
            )

        if not self.train_jsonl.exists():
            raise FileNotFoundError(
                f"Không tìm thấy train JSONL: "
                f"{self.train_jsonl}"
            )

        if not self.train_jsonl.is_file():
            raise ValueError(
                f"train_jsonl không phải file: "
                f"{self.train_jsonl}"
            )

        if self.run_validation:
            if not self.val_jsonl.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy validation JSONL: "
                    f"{self.val_jsonl}"
                )

            if not self.val_jsonl.is_file():
                raise ValueError(
                    f"val_jsonl không phải file: "
                    f"{self.val_jsonl}"
                )

        if self.resume_from_checkpoint is not None:
            if not self.resume_from_checkpoint.exists():
                raise FileNotFoundError(
                    "Không tìm thấy checkpoint để resume: "
                    f"{self.resume_from_checkpoint}"
                )

    def create_output_directories(self) -> None:
        """
        Tạo thư mục log và checkpoint.
        """

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # SERIALIZATION AND DISPLAY
    # ========================================================

    def to_dict(self) -> dict[str, Any]:
        """
        Chuyển cấu hình thành dictionary có thể ghi JSON.
        """

        config_dict = asdict(self)

        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)

            elif isinstance(value, tuple):
                config_dict[key] = list(value)

        config_dict["device"] = str(self.device)
        config_dict["model_dtype"] = str(
            self.model_dtype
        )
        config_dict["amp_enabled"] = (
            self.amp_enabled
        )
        config_dict["use_grad_scaler"] = (
            self.use_grad_scaler
        )
        config_dict["effective_train_batch_size"] = (
            self.effective_train_batch_size
        )

        return config_dict

    def print_summary(self) -> None:
        """
        In các cấu hình quan trọng trước khi huấn luyện.
        """

        print()
        print("=" * 70)
        print("TRAINING CONFIGURATION")
        print("=" * 70)

        print(f"Project root: {PROJECT_ROOT}")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Model dtype: {self.model_dtype}")

        print()
        print("Data:")
        print(f"  Data root: {self.data_root}")
        print(f"  Train JSONL: {self.train_jsonl}")
        print(f"  Validation JSONL: {self.val_jsonl}")
        print(
            f"  Max target length: "
            f"{self.max_target_length}"
        )

        print()
        print("Batching:")
        print(
            f"  Train batch size: "
            f"{self.train_batch_size}"
        )
        print(
            f"  Validation batch size: "
            f"{self.val_batch_size}"
        )
        print(
            f"  Gradient accumulation: "
            f"{self.gradient_accumulation_steps}"
        )
        print(
            f"  Effective train batch size: "
            f"{self.effective_train_batch_size}"
        )
        print(f"  Num workers: {self.num_workers}")

        print()
        print("Training:")
        print(f"  Epochs: {self.num_epochs}")
        print(
            f"  Learning rate: "
            f"{self.learning_rate}"
        )
        print(
            f"  Weight decay: "
            f"{self.weight_decay}"
        )
        print(
            f"  Max grad norm: "
            f"{self.max_grad_norm}"
        )

        print()
        print("Precision:")
        print(f"  FP16: {self.use_fp16}")
        print(f"  BF16: {self.use_bf16}")
        print(f"  AMP enabled: {self.amp_enabled}")
        print(
            f"  Grad scaler: "
            f"{self.use_grad_scaler}"
        )

        print()
        print("LoRA:")
        print(f"  Enabled: {self.use_lora}")
        print(f"  Rank: {self.lora_rank}")
        print(f"  Alpha: {self.lora_alpha}")
        print(f"  Dropout: {self.lora_dropout}")
        print(
            "  Target modules: "
            f"{self.lora_target_modules}"
        )

        print()
        print("Memory:")
        print(
            "  Gradient checkpointing: "
            f"{self.gradient_checkpointing}"
        )

        print()
        print("Output:")
        print(f"  Checkpoints: {self.output_dir}")
        print(f"  Logs: {self.log_dir}")


def get_default_config(
    validate_paths: bool = True,
    create_directories: bool = True,
) -> TrainingConfig:
    """
    Tạo cấu hình mặc định của dự án.

    Args:
        validate_paths:
            Kiểm tra train.jsonl, val.jsonl và data_root.

        create_directories:
            Tự động tạo thư mục checkpoints và logs.
    """

    config = TrainingConfig()

    if validate_paths:
        config.validate_paths()

    if create_directories:
        config.create_output_directories()

    return config
