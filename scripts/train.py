import os
import numpy as np

from transformers import (
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data.loader import load_data
from src.models.model_loader import get_model, get_tokenizer
from src.constants import LABEL_TO_ID
from src.prompts.builders import build_classifier_input


TRAIN_PATH = "data/processed/mlcq/train.csv"
VALID_PATH = "data/processed/mlcq/valid.csv"
TEST_PATH = "data/processed/mlcq/test.csv"

MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "outputs/qwen3_0_6b_seqcls"


def tokenize_dataset(dataset, tokenizer, max_length=512):
    def _tokenize(example):
        text = build_classifier_input(example["code"])

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
        )

        encoded["labels"] = LABEL_TO_ID[example["label"]]
        return encoded

    return dataset.map(_tokenize, remove_columns=dataset.column_names)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = get_tokenizer(MODEL_NAME)

    print("Loading dataset...")
    dataset = load_data(TRAIN_PATH, VALID_PATH, TEST_PATH)

    print("Tokenizing dataset...")
    train_dataset = tokenize_dataset(dataset["train"], tokenizer)
    valid_dataset = tokenize_dataset(dataset["validation"], tokenizer)

    print("Loading model...")
    model = get_model(MODEL_NAME, use_4bit=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        max_grad_norm=1.0,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        report_to="none",
        fp16=False,
        bf16=False,
        seed=42,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    print("Saving best model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete.")


if __name__ == "__main__":
    main()