import os
from typing import Dict, Any
from datasets import load_dataset, Dataset, DatasetDict
from src.constants import LABELS, LABEL_MAP


def normalize_label(label: Any) -> str:
    label = str(label).strip().lower()

    if label not in LABEL_MAP:
        raise ValueError(f"Unknown label: {label}")

    return LABEL_MAP[label]


def detect_file_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return "csv"
    if ext in [".json", ".jsonl"]:
        return "json"

    raise ValueError(f"Unsupported file type: {ext}")


def load_split(path: str) -> Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    file_type = detect_file_type(path)

    dataset = load_dataset(
        file_type,
        data_files=path,
        split="train"
    )

    required_columns = {"code", "label"}
    missing = required_columns - set(dataset.column_names)

    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    def _process(example: Dict[str, Any]) -> Dict[str, str]:
        return {
            "code": str(example["code"]).strip(),
            "label": normalize_label(example["label"]),
        }

    dataset = dataset.map(_process)
    dataset = dataset.filter(lambda x: x["code"] != "")
    dataset = dataset.filter(lambda x: x["label"] in LABELS)

    return dataset


def load_data(train_path: str, valid_path: str, test_path: str) -> DatasetDict:
    train_ds = load_split(train_path)
    valid_ds = load_split(valid_path)
    test_ds = load_split(test_path)

    return DatasetDict({
        "train": train_ds,
        "validation": valid_ds,
        "test": test_ds,
    })