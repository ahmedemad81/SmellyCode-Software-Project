import json
import os
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

from src.data.loader import load_data
from src.models.model_loader import get_tokenizer
from src.constants import LABELS, LABEL_TO_ID, ID_TO_LABEL
from src.prompts.builders import build_classifier_input


TRAIN_PATH = "data/processed/mlcq/train.csv"
VALID_PATH = "data/processed/mlcq/valid.csv"
TEST_PATH = "data/processed/mlcq/test.csv"

BASE_MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_DIR = "outputs/qwen3_0_6b_seqcls"
RESULTS_DIR = "results/mlcq_multiclass"
RESULTS_PATH = os.path.join(RESULTS_DIR, "qwen3_0_6b_seqcls_metrics.json")


def load_finetuned_model(base_model_name: str, adapter_dir: str):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model


@torch.inference_mode()
def predict_labels(model, tokenizer, dataset, max_length=1024):
    preds = []

    for example in dataset:
        text = build_classifier_input(example["code"])
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        outputs = model(**inputs)
        pred_id = int(torch.argmax(outputs.logits, dim=-1).item())
        preds.append(ID_TO_LABEL[pred_id])

    return preds


def compute_metrics(golds, preds):
    gold_ids = [LABEL_TO_ID[x] for x in golds]
    pred_ids = [LABEL_TO_ID[x] for x in preds]

    accuracy = accuracy_score(gold_ids, pred_ids)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        gold_ids, pred_ids, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        gold_ids, pred_ids, average="weighted", zero_division=0
    )

    report = classification_report(
        gold_ids,
        pred_ids,
        target_names=LABELS,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(gold_ids, pred_ids)

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "test_size": len(golds),
        "labels": LABELS,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = get_tokenizer(BASE_MODEL_NAME)

    print("Loading dataset...")
    dataset = load_data(TRAIN_PATH, VALID_PATH, TEST_PATH)
    test_dataset = dataset["test"]

    print("Loading fine-tuned model...")
    model = load_finetuned_model(BASE_MODEL_NAME, MODEL_DIR)

    print("Running inference on test set...")
    preds = predict_labels(model, tokenizer, test_dataset)

    golds = [example["label"] for example in test_dataset]

    print("Computing metrics...")
    metrics = compute_metrics(golds, preds)

    print("\nFinal Metrics:")
    print(json.dumps(metrics, indent=2))

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()