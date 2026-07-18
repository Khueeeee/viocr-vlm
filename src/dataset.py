from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


class VinTextFlorenceDataset(Dataset):
    """
    Dataset OCR cho Florence-2.

    Mỗi dòng JSONL có dạng:

    {
        "image": "train/images/00000001.jpg",
        "prompt": "<OCR>",
        "target": "SỞ"
    }

    Dataset chỉ chịu trách nhiệm:

    1. Đọc metadata JSONL.
    2. Kiểm tra cấu trúc record.
    3. Đọc ảnh từ ổ đĩa.
    4. Chuyển ảnh sang RGB.
    5. Trả về ảnh, prompt và target.

    Việc chuyển ảnh và văn bản thành tensor được thực hiện trong
    FlorenceOCRCollator.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        data_root: str | Path,
        default_prompt: str = "<OCR>",
        validate_image_paths: bool = False,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.data_root = Path(data_root)
        self.default_prompt = default_prompt
        self.validate_image_paths = validate_image_paths

        if not self.jsonl_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy JSONL: {self.jsonl_path}"
            )

        if not self.jsonl_path.is_file():
            raise ValueError(
                f"Đường dẫn JSONL không phải file: {self.jsonl_path}"
            )

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Không tìm thấy data root: {self.data_root}"
            )

        if not self.data_root.is_dir():
            raise ValueError(
                f"Data root không phải thư mục: {self.data_root}"
            )

        if not self.default_prompt.strip():
            raise ValueError("default_prompt không được rỗng.")

        self.samples = self._load_jsonl()

        if not self.samples:
            raise RuntimeError(
                f"Không tìm thấy sample hợp lệ trong {self.jsonl_path}"
            )

        if self.validate_image_paths:
            self._validate_all_image_paths()

    def _load_jsonl(self) -> list[dict[str, str]]:
        """
        Đọc toàn bộ metadata JSONL.

        Chỉ metadata được giữ trong RAM. Ảnh chỉ được đọc khi
        __getitem__ được gọi.
        """

        samples: list[dict[str, str]] = []

        with self.jsonl_path.open(
            mode="r",
            encoding="utf-8",
        ) as jsonl_file:
            for line_number, line in enumerate(
                jsonl_file,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "JSON không hợp lệ tại "
                        f"{self.jsonl_path}, dòng {line_number}."
                    ) from error

                if not isinstance(record, dict):
                    raise TypeError(
                        f"Record tại dòng {line_number} phải là object."
                    )

                sample = self._validate_record(
                    record=record,
                    line_number=line_number,
                )

                samples.append(sample)

        return samples

    def _validate_record(
        self,
        record: dict[str, Any],
        line_number: int,
    ) -> dict[str, str]:
        """
        Kiểm tra và chuẩn hóa một record trong JSONL.
        """

        if "image" not in record:
            raise ValueError(
                f"Dòng {line_number} thiếu field 'image'."
            )

        if "target" not in record:
            raise ValueError(
                f"Dòng {line_number} thiếu field 'target'."
            )

        image_relative_path = str(record["image"]).strip()
        target = str(record["target"]).strip()

        prompt_value = record.get(
            "prompt",
            self.default_prompt,
        )

        prompt = str(prompt_value).strip()

        if not image_relative_path:
            raise ValueError(
                f"Đường dẫn ảnh rỗng tại dòng {line_number}."
            )

        if not prompt:
            prompt = self.default_prompt

        if not target:
            raise ValueError(
                f"Target rỗng tại dòng {line_number}."
            )

        return {
            "image": image_relative_path,
            "prompt": prompt,
            "target": target,
        }

    def _validate_all_image_paths(self) -> None:
        """
        Kiểm tra trước toàn bộ đường dẫn ảnh.

        Tùy chọn này không nên bật mặc định khi huấn luyện vì có thể
        mất thời gian với dataset lớn.
        """

        missing_paths: list[Path] = []

        for sample in self.samples:
            image_path = self.data_root / sample["image"]

            if not image_path.exists():
                missing_paths.append(image_path)

                if len(missing_paths) >= 10:
                    break

        if missing_paths:
            missing_text = "\n".join(
                str(path)
                for path in missing_paths
            )

            raise FileNotFoundError(
                "Phát hiện ảnh bị thiếu. "
                "Tối đa 10 đường dẫn đầu tiên:\n"
                f"{missing_text}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        if index < 0:
            index += len(self.samples)

        if index < 0 or index >= len(self.samples):
            raise IndexError(
                f"Index ngoài phạm vi: {index}"
            )

        sample = self.samples[index]

        image_path = self.data_root / sample["image"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy ảnh: {image_path}"
            )

        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB").copy()
        except Exception as error:
            raise RuntimeError(
                f"Không thể đọc ảnh: {image_path}"
            ) from error

        return {
            "image": image,
            "prompt": sample["prompt"],
            "target": sample["target"],
            "image_path": str(image_path),
            "relative_image_path": sample["image"],
            "index": index,
        }

    def get_metadata(
        self,
        index: int,
    ) -> dict[str, str]:
        """
        Trả metadata mà không đọc ảnh.
        """

        sample = self.samples[index]

        return {
            "image": sample["image"],
            "prompt": sample["prompt"],
            "target": sample["target"],
        }

    def summary(self) -> dict[str, Any]:
        """
        Thông tin tóm tắt dataset.
        """

        return {
            "jsonl_path": str(self.jsonl_path),
            "data_root": str(self.data_root),
            "number_of_samples": len(self.samples),
            "default_prompt": self.default_prompt,
        }