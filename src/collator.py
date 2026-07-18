from __future__ import annotations

from typing import Any

import torch


class FlorenceOCRCollator:
    """
    Collator dành cho Florence-2 OCR.

    Một batch đầu vào:

    [
        {
            "image": PIL.Image,
            "prompt": "<OCR>",
            "target": "SỞ"
        },
        ...
    ]

    Batch đầu ra:

    {
        "input_ids": Tensor,
        "attention_mask": Tensor,
        "pixel_values": Tensor,
        "labels": Tensor,
        "targets": list[str],
        "prompts": list[str],
        "image_paths": list[str]
    }
    """

    def __init__(
        self,
        processor: Any,
        max_target_length: int = 64,
    ) -> None:
        self.processor = processor
        self.max_target_length = max_target_length

        if self.max_target_length <= 0:
            raise ValueError(
                "max_target_length phải lớn hơn 0."
            )

        if not hasattr(self.processor, "tokenizer"):
            raise AttributeError(
                "Processor không có tokenizer."
            )

        tokenizer = self.processor.tokenizer

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError(
                    "Tokenizer không có pad_token_id "
                    "và cũng không có eos_token_id."
                )

            tokenizer.pad_token = tokenizer.eos_token

    def __call__(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not samples:
            raise ValueError("Batch sample đang rỗng.")

        images = []
        prompts: list[str] = []
        targets: list[str] = []
        image_paths: list[str] = []
        relative_image_paths: list[str] = []
        indices: list[int] = []

        for batch_index, sample in enumerate(samples):
            required_fields = {
                "image",
                "prompt",
                "target",
            }

            missing_fields = (
                required_fields - set(sample.keys())
            )

            if missing_fields:
                raise ValueError(
                    f"Sample {batch_index} thiếu field: "
                    f"{sorted(missing_fields)}"
                )

            prompt = str(sample["prompt"]).strip()
            target = str(sample["target"]).strip()

            if not prompt:
                raise ValueError(
                    f"Prompt rỗng tại sample {batch_index}."
                )

            if not target:
                raise ValueError(
                    f"Target rỗng tại sample {batch_index}."
                )

            images.append(sample["image"])
            prompts.append(prompt)
            targets.append(target)

            image_paths.append(
                str(sample.get("image_path", ""))
            )

            relative_image_paths.append(
                str(sample.get("relative_image_path", ""))
            )

            indices.append(
                int(sample.get("index", batch_index))
            )

        model_inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        required_model_fields = {
            "input_ids",
            "attention_mask",
            "pixel_values",
        }

        missing_model_fields = (
            required_model_fields - set(model_inputs.keys())
        )

        if missing_model_fields:
            raise RuntimeError(
                "Processor output thiếu field: "
                f"{sorted(missing_model_fields)}"
            )

        target_encoding = self.processor.tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            add_special_tokens=True,
        )

        labels = target_encoding["input_ids"].clone()

        pad_token_id = (
            self.processor.tokenizer.pad_token_id
        )

        labels[labels == pad_token_id] = -100

        batch = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs[
                "attention_mask"
            ],
            "pixel_values": model_inputs[
                "pixel_values"
            ],
            "labels": labels,
            "targets": targets,
            "prompts": prompts,
            "image_paths": image_paths,
            "relative_image_paths": relative_image_paths,
            "indices": indices,
        }

        self._validate_tensor_batch(batch)

        return batch

    @staticmethod
    def _validate_tensor_batch(
        batch: dict[str, Any],
    ) -> None:
        """
        Kiểm tra nhanh tensor sau khi collate.
        """

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids không phải Tensor.")

        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask không phải Tensor.")

        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError("pixel_values không phải Tensor.")

        if not isinstance(labels, torch.Tensor):
            raise TypeError("labels không phải Tensor.")

        if input_ids.ndim != 2:
            raise RuntimeError(
                "input_ids phải có shape "
                "[batch_size, sequence_length]."
            )

        if attention_mask.ndim != 2:
            raise RuntimeError(
                "attention_mask phải có shape "
                "[batch_size, sequence_length]."
            )

        if pixel_values.ndim != 4:
            raise RuntimeError(
                "pixel_values phải có shape "
                "[batch_size, channels, height, width]."
            )

        if labels.ndim != 2:
            raise RuntimeError(
                "labels phải có shape "
                "[batch_size, target_length]."
            )

        batch_size = input_ids.shape[0]

        if attention_mask.shape[0] != batch_size:
            raise RuntimeError(
                "Batch size của attention_mask không khớp."
            )

        if pixel_values.shape[0] != batch_size:
            raise RuntimeError(
                "Batch size của pixel_values không khớp."
            )

        if labels.shape[0] != batch_size:
            raise RuntimeError(
                "Batch size của labels không khớp."
            )

        if not torch.isfinite(pixel_values).all():
            raise RuntimeError(
                "pixel_values chứa NaN hoặc Infinity."
            )


def decode_labels(
    processor: Any,
    labels: torch.Tensor,
) -> list[str]:
    """
    Decode tensor labels về chuỗi để kiểm tra.

    Các vị trí -100 được đổi lại thành padding trước khi decode.
    """

    labels_for_decode = labels.detach().clone()

    pad_token_id = processor.tokenizer.pad_token_id

    labels_for_decode[
        labels_for_decode == -100
    ] = pad_token_id

    decoded = processor.tokenizer.batch_decode(
        labels_for_decode,
        skip_special_tokens=True,
    )

    return [
        text.strip()
        for text in decoded
    ]