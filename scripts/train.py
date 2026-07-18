from __future__ import annotations

import csv
import gc
import json
import math
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from peft import PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import get_scheduler


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.collator import FlorenceOCRCollator  # noqa: E402
from src.config import TrainingConfig, get_default_config  # noqa: E402
from src.dataset import VinTextFlorenceDataset  # noqa: E402
from src.model import (  # noqa: E402
    load_florence_model,
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


# ============================================================
# TRAINING STATE
# ============================================================

@dataclass
class TrainingState:
    """
    Trạng thái cần thiết để theo dõi hoặc tiếp tục huấn luyện.
    """

    start_epoch: int = 1
    global_step: int = 0
    micro_step: int = 0

    # Batch cuối cùng đã hoàn tất an toàn trong epoch hiện tại.
    # Chỉ cập nhật sau optimizer step thành công.
    resume_batch_index: int = 0

    best_val_loss: float = math.inf

    total_optimizer_steps: int = 0
    warmup_steps: int = 0

    last_train_loss: float | None = None
    last_val_loss: float | None = None


# ============================================================
# CSV LOGGER
# ============================================================

class CSVTrainingLogger:
    """
    Ghi lịch sử train và validation vào file CSV.

    File mặc định:

        logs/train_log.csv
    """

    FIELDNAMES = [
        "timestamp",
        "event",
        "epoch",
        "micro_step",
        "global_step",
        "train_loss",
        "val_loss",
        "learning_rate",
        "grad_norm",
        "gpu_allocated_gb",
        "gpu_reserved_gb",
        "elapsed_seconds",
    ]

    def __init__(
        self,
        log_path: str | Path,
        append: bool = False,
    ) -> None:
        self.log_path = Path(log_path)

        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_exists = self.log_path.exists()
        file_has_content = (
            file_exists
            and self.log_path.stat().st_size > 0
        )

        mode = "a" if append else "w"

        self.file = self.log_path.open(
            mode=mode,
            encoding="utf-8",
            newline="",
        )

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self.FIELDNAMES,
        )

        if not append or not file_has_content:
            self.writer.writeheader()
            self.file.flush()

    def log(
        self,
        *,
        event: str,
        epoch: int,
        micro_step: int,
        global_step: int,
        train_loss: float | None,
        val_loss: float | None,
        learning_rate: float | None,
        grad_norm: float | None,
        elapsed_seconds: float,
    ) -> None:
        allocated_gb = 0.0
        reserved_gb = 0.0

        if torch.cuda.is_available():
            allocated_gb = (
                torch.cuda.memory_allocated() / 1024**3
            )

            reserved_gb = (
                torch.cuda.memory_reserved() / 1024**3
            )

        row = {
            "timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "event": event,
            "epoch": epoch,
            "micro_step": micro_step,
            "global_step": global_step,
            "train_loss": (
                f"{train_loss:.8f}"
                if train_loss is not None
                else ""
            ),
            "val_loss": (
                f"{val_loss:.8f}"
                if val_loss is not None
                else ""
            ),
            "learning_rate": (
                f"{learning_rate:.12g}"
                if learning_rate is not None
                else ""
            ),
            "grad_norm": (
                f"{grad_norm:.8f}"
                if grad_norm is not None
                else ""
            ),
            "gpu_allocated_gb": (
                f"{allocated_gb:.4f}"
            ),
            "gpu_reserved_gb": (
                f"{reserved_gb:.4f}"
            ),
            "elapsed_seconds": (
                f"{elapsed_seconds:.2f}"
            ),
        }

        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.flush()
            self.file.close()

    def __enter__(self) -> CSVTrainingLogger:
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()


# ============================================================
# DATASET HELPERS
# ============================================================

def limit_dataset(
    dataset: Dataset,
    max_samples: int | None,
) -> Dataset:
    """
    Chỉ lấy các sample đầu tiên khi chạy debug.
    """

    if max_samples is None:
        return dataset

    number_of_samples = min(
        max_samples,
        len(dataset),
    )

    return Subset(
        dataset,
        range(number_of_samples),
    )


def build_dataloaders(
    config: TrainingConfig,
    processor: Any,
) -> tuple[DataLoader, DataLoader | None]:
    """
    Tạo train và validation DataLoader.
    """

    print()
    print("=" * 70)
    print("BUILDING DATASETS")
    print("=" * 70)

    train_dataset: Dataset = VinTextFlorenceDataset(
        jsonl_path=config.train_jsonl,
        data_root=config.data_root,
        default_prompt=config.default_prompt,
        validate_image_paths=False,
    )

    train_dataset = limit_dataset(
        dataset=train_dataset,
        max_samples=config.max_train_samples,
    )

    val_dataset: Dataset | None = None

    if config.run_validation:
        val_dataset = VinTextFlorenceDataset(
            jsonl_path=config.val_jsonl,
            data_root=config.data_root,
            default_prompt=config.default_prompt,
            validate_image_paths=False,
        )

        val_dataset = limit_dataset(
            dataset=val_dataset,
            max_samples=config.max_val_samples,
        )

    collator = FlorenceOCRCollator(
        processor=processor,
        max_target_length=config.max_target_length,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=config.num_workers,
        pin_memory=config.resolved_pin_memory,
        persistent_workers=(
            config.resolved_persistent_workers
        ),
        drop_last=config.drop_last_train_batch,
        collate_fn=collator,
    )

    val_loader: DataLoader | None = None

    if val_dataset is not None:
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.resolved_pin_memory,
            persistent_workers=(
                config.resolved_persistent_workers
            ),
            drop_last=False,
            collate_fn=collator,
        )

    print(
        f"Train samples: {len(train_dataset):,}"
    )
    print(
        f"Train batches per epoch: "
        f"{len(train_loader):,}"
    )

    if val_dataset is not None and val_loader is not None:
        print(
            f"Validation samples: "
            f"{len(val_dataset):,}"
        )
        print(
            f"Validation batches: "
            f"{len(val_loader):,}"
        )
    else:
        print("Validation disabled.")

    return train_loader, val_loader


# ============================================================
# MODEL HELPERS
# ============================================================

def enable_training_memory_optimizations(
    model: torch.nn.Module,
    config: TrainingConfig,
) -> None:
    """
    Bật gradient checkpointing và tắt decoder cache.
    """

    if config.disable_model_cache_during_training:
        model_config = getattr(model, "config", None)

        if model_config is not None:
            model_config.use_cache = False

        base_model = getattr(
            model,
            "base_model",
            None,
        )

        if base_model is not None:
            base_config = getattr(
                base_model,
                "config",
                None,
            )

            if base_config is not None:
                base_config.use_cache = False

    if not config.gradient_checkpointing:
        return

    checkpointing_enabled = False

    candidates = [
        model,
        getattr(model, "base_model", None),
        getattr(model, "model", None),
    ]

    for candidate in candidates:
        if candidate is None:
            continue

        enable_function = getattr(
            candidate,
            "gradient_checkpointing_enable",
            None,
        )

        if callable(enable_function):
            enable_function()
            checkpointing_enabled = True
            break

    if checkpointing_enabled:
        print("Gradient checkpointing enabled.")
    else:
        print(
            "WARNING: Model không cung cấp "
            "gradient_checkpointing_enable()."
        )


def build_model(
    config: TrainingConfig,
) -> torch.nn.Module:
    """
    Load Florence-2 và LoRA.

    Nếu resume_from_checkpoint được đặt, adapter đã lưu sẽ được
    nạp lại ở chế độ trainable.
    """

    print()
    print("=" * 70)
    print("BUILDING MODEL")
    print("=" * 70)

    if config.resume_from_checkpoint is None:
        if not config.use_lora:
            model = load_florence_model(
                model_name=config.model_name,
                device=config.device,
                dtype=config.model_dtype,
                trust_remote_code=(
                    config.trust_remote_code
                ),
                training=True,
            )
        else:
            model = load_florence_model_with_lora(
                model_name=config.model_name,
                device=config.device,
                dtype=config.model_dtype,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=(
                    config.lora_target_modules
                ),
                trust_remote_code=(
                    config.trust_remote_code
                ),
            )
    else:
        if not config.use_lora:
            raise NotImplementedError(
                "Resume hiện được thiết kế cho LoRA. "
                "Full fine-tuning chưa được hỗ trợ."
            )

        checkpoint_dir = (
            config.resume_from_checkpoint
        )

        adapter_dir = checkpoint_dir / "adapter"

        if not adapter_dir.exists():
            raise FileNotFoundError(
                "Không tìm thấy adapter trong checkpoint: "
                f"{adapter_dir}"
            )

        print(
            "Loading base Florence-2 model for resume..."
        )

        base_model = load_florence_model(
            model_name=config.model_name,
            device=config.device,
            dtype=config.model_dtype,
            trust_remote_code=(
                config.trust_remote_code
            ),
            training=True,
        )

        print(
            f"Loading LoRA adapter: {adapter_dir}"
        )

        model = PeftModel.from_pretrained(
            base_model,
            adapter_dir,
            is_trainable=True,
        )

        model.to(config.device)
        model.train()

    enable_training_memory_optimizations(
        model=model,
        config=config,
    )

    print_parameter_summary(model)

    return model


# ============================================================
# OPTIMIZER AND SCHEDULER
# ============================================================

def build_optimizer(
    model: torch.nn.Module,
    config: TrainingConfig,
) -> AdamW:
    """
    Tạo AdamW chỉ cho các parameter có requires_grad=True.
    """

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "Không có parameter nào có thể huấn luyện."
        )

    optimizer = AdamW(
        params=trainable_parameters,
        lr=config.learning_rate,
        betas=(
            config.adam_beta1,
            config.adam_beta2,
        ),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )

    return optimizer


def calculate_training_steps(
    train_loader: DataLoader,
    config: TrainingConfig,
) -> tuple[int, int, int]:
    """
    Tính số optimizer step trong mỗi epoch, tổng step và warmup.
    """

    optimizer_steps_per_epoch = math.ceil(
        len(train_loader)
        / config.gradient_accumulation_steps
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        * config.num_epochs
    )

    if config.warmup_steps > 0:
        warmup_steps = config.warmup_steps
    else:
        warmup_steps = int(
            total_optimizer_steps
            * config.warmup_ratio
        )

    return (
        optimizer_steps_per_epoch,
        total_optimizer_steps,
        warmup_steps,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    total_optimizer_steps: int,
    warmup_steps: int,
) -> Any:
    """
    Tạo learning-rate scheduler của Transformers.
    """

    scheduler = get_scheduler(
        name=config.scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    return scheduler


# ============================================================
# AMP HELPERS
# ============================================================

def build_grad_scaler(
    config: TrainingConfig,
) -> torch.amp.GradScaler:
    """
    GradScaler chỉ thực sự bật khi dùng CUDA FP16.
    """

    return torch.amp.GradScaler(
        device="cuda",
        enabled=config.use_grad_scaler,
        init_scale=config.grad_scaler_init_scale,
        growth_factor=config.grad_scaler_growth_factor,
        backoff_factor=config.grad_scaler_backoff_factor,
        growth_interval=config.grad_scaler_growth_interval,
    )


def autocast_context(
    config: TrainingConfig,
) -> Any:
    """
    Trả context manager phù hợp cho AMP.
    """

    if not config.amp_enabled:
        return nullcontext()

    return torch.amp.autocast(
        device_type=config.device.type,
        dtype=config.amp_dtype,
        enabled=True,
    )


# ============================================================
# TRAINING HELPERS
# ============================================================

def get_current_learning_rate(
    optimizer: torch.optim.Optimizer,
) -> float:
    return float(
        optimizer.param_groups[0]["lr"]
    )


def calculate_group_size(
    batch_index: int,
    number_of_batches: int,
    accumulation_steps: int,
) -> int:
    """
    Trả số micro-batch trong nhóm gradient accumulation hiện tại.

    Việc này giúp nhóm cuối epoch không bị chia loss sai khi số batch
    không chia hết cho gradient_accumulation_steps.
    """

    zero_based_index = batch_index - 1

    group_start = (
        zero_based_index
        // accumulation_steps
    ) * accumulation_steps

    remaining_from_group_start = (
        number_of_batches - group_start
    )

    return min(
        accumulation_steps,
        remaining_from_group_start,
    )


def should_optimizer_step(
    batch_index: int,
    number_of_batches: int,
    accumulation_steps: int,
) -> bool:
    return (
        batch_index % accumulation_steps == 0
        or batch_index == number_of_batches
    )


def clear_cuda_cache() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    config: TrainingConfig,
) -> float:
    """
    Tính validation loss trung bình theo số sample.
    """

    model.eval()

    total_weighted_loss = 0.0
    total_samples = 0

    number_of_batches = len(val_loader)

    if config.max_validation_batches is not None:
        number_of_batches = min(
            number_of_batches,
            config.max_validation_batches,
        )

    print()
    print("-" * 70)
    print(
        f"VALIDATION: processing "
        f"{number_of_batches:,} batch(es)"
    )
    print("-" * 70)

    for batch_index, batch in enumerate(
        val_loader,
        start=1,
    ):
        if (
            config.max_validation_batches is not None
            and batch_index
            > config.max_validation_batches
        ):
            break

        model_batch = move_batch_to_device(
            batch=batch,
            device=config.device,
            pixel_dtype=config.model_dtype,
        )

        with autocast_context(config):
            outputs = model(
                input_ids=model_batch["input_ids"],
                attention_mask=(
                    model_batch["attention_mask"]
                ),
                pixel_values=(
                    model_batch["pixel_values"]
                ),
                labels=model_batch["labels"],
            )

            loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Validation loss là NaN hoặc Infinity "
                f"tại batch {batch_index}."
            )

        batch_size = int(
            model_batch["input_ids"].shape[0]
        )

        total_weighted_loss += (
            float(loss.detach().item())
            * batch_size
        )

        total_samples += batch_size

        if (
            batch_index % 100 == 0
            or batch_index == number_of_batches
        ):
            running_loss = (
                total_weighted_loss
                / max(total_samples, 1)
            )

            print(
                f"Validation batch "
                f"{batch_index:,}/{number_of_batches:,} "
                f"| loss={running_loss:.6f}"
            )

        del outputs
        del loss
        del model_batch

    if total_samples == 0:
        raise RuntimeError(
            "Validation không xử lý được sample nào."
        )

    average_loss = (
        total_weighted_loss / total_samples
    )

    model.train()

    print(
        f"Validation loss: {average_loss:.6f}"
    )

    return average_loss


# ============================================================
# CHECKPOINT OPERATIONS
# ============================================================

def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def save_checkpoint(
    *,
    checkpoint_dir: Path,
    model: torch.nn.Module,
    processor: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    state: TrainingState,
    config: TrainingConfig,
    current_epoch: int,
    epoch_completed: bool = False,
) -> None:
    """
    Lưu adapter, processor và trainer state.
    """

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    adapter_dir = checkpoint_dir / "adapter"
    processor_dir = checkpoint_dir / "processor"

    model.save_pretrained(
        adapter_dir,
        safe_serialization=True,
    )

    processor.save_pretrained(
        processor_dir,
    )

    trainer_state = {
        "epoch": current_epoch,
        "epoch_completed": epoch_completed,
        "start_epoch": state.start_epoch,
        "global_step": state.global_step,
        "micro_step": state.micro_step,
        "resume_batch_index": state.resume_batch_index,
        "best_val_loss": state.best_val_loss,
        "last_train_loss": state.last_train_loss,
        "last_val_loss": state.last_val_loss,
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
        ),
        "scaler_state_dict": (
            scaler.state_dict()
        ),
    }

    torch.save(
        trainer_state,
        checkpoint_dir / "trainer_state.pt",
    )

    save_json(
        config.to_dict(),
        checkpoint_dir / "training_config.json",
    )

    print(
        f"Checkpoint saved: {checkpoint_dir}"
    )


def load_training_state(
    *,
    checkpoint_dir: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    number_of_batches: int,
) -> TrainingState:
    """
    Nạp optimizer, scheduler, scaler và các bộ đếm.
    """

    trainer_state_path = (
        checkpoint_dir / "trainer_state.pt"
    )

    if not trainer_state_path.exists():
        raise FileNotFoundError(
            "Không tìm thấy trainer_state.pt: "
            f"{trainer_state_path}"
        )

    saved_state = torch.load(
        trainer_state_path,
        map_location=device,
        weights_only=False,
    )

    optimizer.load_state_dict(
        saved_state["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        saved_state["scheduler_state_dict"]
    )

    scaler_state = saved_state.get(
        "scaler_state_dict"
    )

    if scaler_state:
        scaler.load_state_dict(scaler_state)

    saved_epoch = max(
        int(saved_state.get("epoch", 1)),
        1,
    )

    epoch_completed = bool(
        saved_state.get("epoch_completed", False)
    )

    saved_micro_step = int(
        saved_state.get("micro_step", 0)
    )

    saved_resume_batch_index = saved_state.get(
        "resume_batch_index"
    )

    # Tương thích checkpoint cũ chưa lưu resume_batch_index.
    if saved_resume_batch_index is None:
        saved_resume_batch_index = (
            saved_micro_step % number_of_batches
            if number_of_batches > 0
            else 0
        )

    saved_resume_batch_index = int(
        saved_resume_batch_index
    )

    if epoch_completed:
        start_epoch = saved_epoch + 1
        saved_resume_batch_index = 0
    else:
        start_epoch = saved_epoch

    if not 0 <= saved_resume_batch_index <= number_of_batches:
        raise ValueError(
            "resume_batch_index không hợp lệ: "
            f"{saved_resume_batch_index}; "
            f"train loader có {number_of_batches} batch."
        )

    state = TrainingState(
        start_epoch=start_epoch,
        global_step=int(
            saved_state.get("global_step", 0)
        ),
        micro_step=saved_micro_step,
        resume_batch_index=(
            saved_resume_batch_index
        ),
        best_val_loss=float(
            saved_state.get(
                "best_val_loss",
                math.inf,
            )
        ),
        last_train_loss=(
            saved_state.get("last_train_loss")
        ),
        last_val_loss=(
            saved_state.get("last_val_loss")
        ),
    )

    print()
    print("=" * 70)
    print("RESUME STATE")
    print("=" * 70)
    print(f"Saved epoch: {saved_epoch}")
    print(f"Epoch completed: {epoch_completed}")
    print(
        f"Continue from epoch: {state.start_epoch}"
    )
    print(
        "Resume after batch: "
        f"{state.resume_batch_index:,}/"
        f"{number_of_batches:,}"
    )
    print(
        f"Global optimizer step: "
        f"{state.global_step}"
    )
    print(
        f"Best validation loss: "
        f"{state.best_val_loss}"
    )

    return state


def find_step_checkpoints(
    output_dir: Path,
) -> list[Path]:
    checkpoints = [
        path
        for path in output_dir.glob(
            "checkpoint-step-*"
        )
        if path.is_dir()
    ]

    def extract_step(path: Path) -> int:
        try:
            return int(
                path.name.rsplit("-", maxsplit=1)[-1]
            )
        except ValueError:
            return -1

    return sorted(
        checkpoints,
        key=extract_step,
    )


def remove_old_checkpoints(
    output_dir: Path,
    max_checkpoints_to_keep: int,
) -> None:
    """
    Chỉ giới hạn checkpoint-step-*.

    best_model và last_checkpoint không bị xóa.
    """

    checkpoints = find_step_checkpoints(
        output_dir
    )

    number_to_remove = (
        len(checkpoints)
        - max_checkpoints_to_keep
    )

    if number_to_remove <= 0:
        return

    for checkpoint_path in checkpoints[
        :number_to_remove
    ]:
        print(
            f"Removing old checkpoint: "
            f"{checkpoint_path}"
        )

        shutil.rmtree(
            checkpoint_path,
            ignore_errors=False,
        )


def replace_directory(
    source_dir: Path,
    destination_dir: Path,
) -> None:
    """
    Sao chép checkpoint vào alias như best_model hoặc last_checkpoint.
    """

    if destination_dir.exists():
        shutil.rmtree(destination_dir)

    shutil.copytree(
        source_dir,
        destination_dir,
    )


# ============================================================
# PERIODIC VALIDATION
# ============================================================

def run_validation_and_checkpoint_if_needed(
    *,
    model: torch.nn.Module,
    processor: Any,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    config: TrainingConfig,
    state: TrainingState,
    current_epoch: int,
    csv_logger: CSVTrainingLogger,
    training_start_time: float,
) -> float | None:
    if not config.run_validation:
        return None

    if val_loader is None:
        return None

    val_loss = validate(
        model=model,
        val_loader=val_loader,
        config=config,
    )

    state.last_val_loss = val_loss

    csv_logger.log(
        event="validation",
        epoch=current_epoch,
        micro_step=state.micro_step,
        global_step=state.global_step,
        train_loss=state.last_train_loss,
        val_loss=val_loss,
        learning_rate=get_current_learning_rate(
            optimizer
        ),
        grad_norm=None,
        elapsed_seconds=(
            time.perf_counter()
            - training_start_time
        ),
    )

    is_better = (
        val_loss > state.best_val_loss
        if config.greater_is_better
        else val_loss < state.best_val_loss
    )

    if is_better:
        previous_best = state.best_val_loss
        state.best_val_loss = val_loss

        print()
        print(
            "New best validation result: "
            f"{previous_best:.6f} -> "
            f"{val_loss:.6f}"
        )

        best_dir = (
            config.output_dir / "best_model"
        )

        save_checkpoint(
            checkpoint_dir=best_dir,
            model=model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            state=state,
            config=config,
            current_epoch=current_epoch,
        )

    return val_loss


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    *,
    model: torch.nn.Module,
    processor: Any,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    config: TrainingConfig,
    state: TrainingState,
    epoch: int,
    csv_logger: CSVTrainingLogger,
    training_start_time: float,
) -> float:
    """
    Huấn luyện một epoch.
    """

    model.train()

    optimizer.zero_grad(set_to_none=True)

    total_weighted_loss = 0.0
    total_samples = 0

    interval_loss_sum = 0.0
    interval_sample_count = 0

    number_of_batches = len(train_loader)

    # Dùng permutation cố định cho từng epoch.
    loader_generator = getattr(
        train_loader,
        "generator",
        None,
    )

    if loader_generator is not None:
        loader_generator.manual_seed(
            config.seed + epoch
        )

    resume_batch_index = 0

    if epoch == state.start_epoch:
        resume_batch_index = (
            state.resume_batch_index
        )
    else:
        state.resume_batch_index = 0

    if resume_batch_index > 0:
        print("=" * 70)
        print(f"RESUME EPOCH {epoch}")
        print(
            f"Skip {resume_batch_index:,} "
            "batch(es) already completed."
        )
        print("=" * 70)

    print()
    print("=" * 70)
    print(
        f"EPOCH {epoch}/{config.num_epochs}"
    )
    print("=" * 70)

    epoch_start_time = time.perf_counter()

    for batch_index, batch in enumerate(
        train_loader,
        start=1,
    ):
        if batch_index <= resume_batch_index:
            continue
        state.micro_step += 1

        group_size = calculate_group_size(
            batch_index=batch_index,
            number_of_batches=number_of_batches,
            accumulation_steps=(
                config.gradient_accumulation_steps
            ),
        )

        model_batch = move_batch_to_device(
            batch=batch,
            device=config.device,
            pixel_dtype=config.model_dtype,
        )

        batch_size = int(
            model_batch["input_ids"].shape[0]
        )

        with autocast_context(config):
            outputs = model(
                input_ids=model_batch["input_ids"],
                attention_mask=(
                    model_batch["attention_mask"]
                ),
                pixel_values=(
                    model_batch["pixel_values"]
                ),
                labels=model_batch["labels"],
            )

            raw_loss = outputs.loss

            loss_for_backward = (
                raw_loss / group_size
            )

        if not torch.isfinite(raw_loss):
            optimizer.zero_grad(set_to_none=True)

            message = (
                "Training loss là NaN hoặc Infinity "
                f"| epoch {epoch} "
                f"| batch {batch_index} "
                f"| images {batch.get('image_paths')}"
            )

            del outputs
            del raw_loss
            del loss_for_backward
            del model_batch

            if config.skip_non_finite_steps:
                print(f"WARNING: {message}. Skipping batch.")
                continue

            raise RuntimeError(message)

        raw_loss_value = float(
            raw_loss.detach().item()
        )

        total_weighted_loss += (
            raw_loss_value * batch_size
        )
        total_samples += batch_size

        interval_loss_sum += (
            raw_loss_value * batch_size
        )
        interval_sample_count += batch_size

        scaler.scale(
            loss_for_backward
        ).backward()

        perform_step = should_optimizer_step(
            batch_index=batch_index,
            number_of_batches=number_of_batches,
            accumulation_steps=(
                config.gradient_accumulation_steps
            ),
        )

        grad_norm_value: float | None = None

        if perform_step:
            scaler.unscale_(optimizer)

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.max_grad_norm,
                )
            )

            grad_norm_value = float(
                grad_norm.detach().item()
                if isinstance(
                    grad_norm,
                    torch.Tensor,
                )
                else grad_norm
            )

            gradient_is_finite = math.isfinite(
                grad_norm_value
            )

            if (
                not gradient_is_finite
                and not config.use_grad_scaler
            ):
                optimizer.zero_grad(set_to_none=True)

                message = (
                    "Gradient norm là NaN hoặc Infinity "
                    f"| epoch {epoch} "
                    f"| batch {batch_index}"
                )

                if config.skip_non_finite_steps:
                    print(f"WARNING: {message}. Skipping step.")
                    del outputs
                    del raw_loss
                    del loss_for_backward
                    del model_batch
                    continue

                raise RuntimeError(message)

            previous_scale = float(
                scaler.get_scale()
            )

            # Khi GradScaler phát hiện Inf/NaN, scaler.step() tự động
            # bỏ qua optimizer.step(). scaler.update() sau đó giảm scale.
            scaler.step(optimizer)
            scaler.update()

            current_scale = float(
                scaler.get_scale()
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            optimizer_step_skipped = (
                config.use_grad_scaler
                and current_scale < previous_scale
            )

            if optimizer_step_skipped:
                message = (
                    "FP16 overflow; optimizer step was skipped "
                    f"| epoch {epoch} "
                    f"| batch {batch_index} "
                    f"| grad {grad_norm_value} "
                    f"| scale {previous_scale:g} -> {current_scale:g}"
                )

                if config.skip_non_finite_steps:
                    print(f"WARNING: {message}")
                    del outputs
                    del raw_loss
                    del loss_for_backward
                    del model_batch
                    continue

                raise RuntimeError(message)

            if not gradient_is_finite:
                message = (
                    "Gradient norm không hữu hạn nhưng GradScaler "
                    "không báo đã bỏ qua optimizer step "
                    f"| epoch {epoch} "
                    f"| batch {batch_index}"
                )

                if config.skip_non_finite_steps:
                    print(f"WARNING: {message}")
                    del outputs
                    del raw_loss
                    del loss_for_backward
                    del model_batch
                    continue

                raise RuntimeError(message)

            scheduler.step()

            state.global_step += 1
            state.resume_batch_index = (
                batch_index
            )
            state.last_train_loss = (
                raw_loss_value
            )

            if (
                state.global_step
                % config.log_every_steps
                == 0
            ):
                interval_average_loss = (
                    interval_loss_sum
                    / max(
                        interval_sample_count,
                        1,
                    )
                )

                learning_rate = (
                    get_current_learning_rate(
                        optimizer
                    )
                )

                elapsed = (
                    time.perf_counter()
                    - training_start_time
                )

                print(
                    f"Epoch {epoch}/{config.num_epochs} "
                    f"| batch "
                    f"{batch_index:,}/"
                    f"{number_of_batches:,} "
                    f"| step "
                    f"{state.global_step:,} "
                    f"| loss "
                    f"{interval_average_loss:.6f} "
                    f"| lr "
                    f"{learning_rate:.3e} "
                    f"| grad "
                    f"{grad_norm_value:.4f} "
                    f"| elapsed "
                    f"{elapsed / 60:.1f} min"
                )

                csv_logger.log(
                    event="train",
                    epoch=epoch,
                    micro_step=(
                        state.micro_step
                    ),
                    global_step=(
                        state.global_step
                    ),
                    train_loss=(
                        interval_average_loss
                    ),
                    val_loss=None,
                    learning_rate=(
                        learning_rate
                    ),
                    grad_norm=(
                        grad_norm_value
                    ),
                    elapsed_seconds=elapsed,
                )

                interval_loss_sum = 0.0
                interval_sample_count = 0

            if (
                config.save_every_steps > 0
                and state.global_step
                % config.save_every_steps
                == 0
            ):
                checkpoint_dir = (
                    config.output_dir
                    / (
                        "checkpoint-step-"
                        f"{state.global_step}"
                    )
                )

                save_checkpoint(
                    checkpoint_dir=(
                        checkpoint_dir
                    ),
                    model=model,
                    processor=processor,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    state=state,
                    config=config,
                    current_epoch=epoch,
                )

                remove_old_checkpoints(
                    output_dir=(
                        config.output_dir
                    ),
                    max_checkpoints_to_keep=(
                        config
                        .max_checkpoints_to_keep
                    ),
                )

            if (
                config.run_validation
                and config.validate_every_steps
                > 0
                and state.global_step
                % config.validate_every_steps
                == 0
            ):
                run_validation_and_checkpoint_if_needed(
                    model=model,
                    processor=processor,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    state=state,
                    current_epoch=epoch,
                    csv_logger=csv_logger,
                    training_start_time=(
                        training_start_time
                    ),
                )

        if (
            config.empty_cuda_cache_every_steps
            > 0
            and state.micro_step
            % config.empty_cuda_cache_every_steps
            == 0
        ):
            clear_cuda_cache()

        del outputs
        del raw_loss
        del loss_for_backward
        del model_batch

    if total_samples == 0:
        raise RuntimeError(
            "Epoch không xử lý được sample nào."
        )

    average_epoch_loss = (
        total_weighted_loss / total_samples
    )

    epoch_elapsed = (
        time.perf_counter() - epoch_start_time
    )

    state.last_train_loss = average_epoch_loss
    state.resume_batch_index = 0
    state.start_epoch = epoch + 1

    print()
    print(
        f"Epoch {epoch} completed "
        f"| train loss={average_epoch_loss:.6f} "
        f"| time={epoch_elapsed / 60:.2f} min"
    )

    csv_logger.log(
        event="epoch_end",
        epoch=epoch,
        micro_step=state.micro_step,
        global_step=state.global_step,
        train_loss=average_epoch_loss,
        val_loss=state.last_val_loss,
        learning_rate=get_current_learning_rate(
            optimizer
        ),
        grad_norm=None,
        elapsed_seconds=(
            time.perf_counter()
            - training_start_time
        ),
    )

    return average_epoch_loss


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train(
    config: TrainingConfig,
) -> None:
    """
    Điều phối toàn bộ quá trình fine-tune.
    """

    config.validate_paths()
    config.create_output_directories()

    set_seed(config.seed)

    if config.deterministic:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print("VIOCR-VLM FLORENCE-2 TRAINING")
    print("=" * 70)

    config.print_summary()
    print_device_information(config.device)

    print()
    print("Loading Florence-2 processor...")

    processor = load_florence_processor(
        model_name=config.model_name,
        trust_remote_code=config.trust_remote_code,
    )

    print("Processor loaded.")

    train_loader, val_loader = (
        build_dataloaders(
            config=config,
            processor=processor,
        )
    )

    (
        optimizer_steps_per_epoch,
        total_optimizer_steps,
        warmup_steps,
    ) = calculate_training_steps(
        train_loader=train_loader,
        config=config,
    )

    print()
    print("=" * 70)
    print("TRAINING STEPS")
    print("=" * 70)
    print(
        f"Optimizer steps per epoch: "
        f"{optimizer_steps_per_epoch:,}"
    )
    print(
        f"Total optimizer steps: "
        f"{total_optimizer_steps:,}"
    )
    print(
        f"Warmup steps: {warmup_steps:,}"
    )

    model = build_model(config)

    optimizer = build_optimizer(
        model=model,
        config=config,
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        config=config,
        total_optimizer_steps=(
            total_optimizer_steps
        ),
        warmup_steps=warmup_steps,
    )

    scaler = build_grad_scaler(config)

    state = TrainingState(
        total_optimizer_steps=(
            total_optimizer_steps
        ),
        warmup_steps=warmup_steps,
    )

    if config.resume_from_checkpoint is not None:
        state = load_training_state(
            checkpoint_dir=(
                config.resume_from_checkpoint
            ),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=config.device,
            number_of_batches=len(train_loader),
        )

        state.total_optimizer_steps = (
            total_optimizer_steps
        )
        state.warmup_steps = warmup_steps

    if state.start_epoch > config.num_epochs:
        print(
            "Checkpoint đã hoàn thành đủ số epoch. "
            "Không còn epoch để huấn luyện."
        )
        return

    config_snapshot_path = (
        config.log_dir
        / "training_config.json"
    )

    save_json(
        config.to_dict(),
        config_snapshot_path,
    )

    log_path = (
        config.log_dir / "train_log.csv"
    )

    append_log = (
        config.resume_from_checkpoint
        is not None
    )

    training_start_time = time.perf_counter()

    print_gpu_memory()

    current_epoch = state.start_epoch

    try:
        with CSVTrainingLogger(
            log_path=log_path,
            append=append_log,
        ) as csv_logger:
            for epoch in range(
                state.start_epoch,
                config.num_epochs + 1,
            ):
                current_epoch = epoch

                train_one_epoch(
                    model=model,
                    processor=processor,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    state=state,
                    epoch=epoch,
                    csv_logger=csv_logger,
                    training_start_time=(
                        training_start_time
                    ),
                )

                if (
                    config.run_validation
                    and val_loader is not None
                ):
                    run_validation_and_checkpoint_if_needed(
                        model=model,
                        processor=processor,
                        val_loader=val_loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=config,
                        state=state,
                        current_epoch=epoch,
                        csv_logger=csv_logger,
                        training_start_time=(
                            training_start_time
                        ),
                    )

                if config.save_every_epoch:
                    epoch_checkpoint_dir = (
                        config.output_dir
                        / f"checkpoint-epoch-{epoch}"
                    )

                    save_checkpoint(
                        checkpoint_dir=(
                            epoch_checkpoint_dir
                        ),
                        model=model,
                        processor=processor,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        state=state,
                        config=config,
                        current_epoch=epoch,
                    )

                    last_checkpoint_dir = (
                        config.output_dir
                        / "last_checkpoint"
                    )

                    replace_directory(
                        source_dir=(
                            epoch_checkpoint_dir
                        ),
                        destination_dir=(
                            last_checkpoint_dir
                        ),
                    )

                clear_cuda_cache()

    except KeyboardInterrupt:
        print()
        print(
            "Training bị dừng bằng bàn phím. "
            "Đang lưu emergency checkpoint..."
        )

        emergency_dir = (
            config.output_dir
            / "interrupted_checkpoint"
        )

        save_checkpoint(
            checkpoint_dir=emergency_dir,
            model=model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            state=state,
            config=config,
            current_epoch=max(
                current_epoch,
                1,
            ),
            epoch_completed=False,
        )

        raise

    total_elapsed = (
        time.perf_counter()
        - training_start_time
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(
        f"Global optimizer steps: "
        f"{state.global_step:,}"
    )
    print(
        f"Best validation loss: "
        f"{state.best_val_loss:.6f}"
    )
    print(
        f"Last training loss: "
        f"{state.last_train_loss}"
    )
    print(
        f"Last validation loss: "
        f"{state.last_val_loss}"
    )
    print(
        f"Total time: "
        f"{total_elapsed / 3600:.2f} hours"
    )
    print(
        f"Checkpoints: {config.output_dir}"
    )
    print(f"Training log: {log_path}")

    print_gpu_memory()


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    config = get_default_config(
        validate_paths=True,
        create_directories=True,
    )

    train(config)


if __name__ == "__main__":
    main()
