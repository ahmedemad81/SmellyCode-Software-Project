import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import LABELS, LABEL_MAP, INTERIM_DATA_PATH, PROCESSED_DIR, TRAIN_PATH, VALID_PATH, TEST_PATH

def normalize_label(label: str) -> str:
    label = str(label).strip().lower()

    if label not in LABEL_MAP:
        raise ValueError(f"Unknown label found: {label}")

    return LABEL_MAP[label]


def deduplicate_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate code smell instances using location + label.
    """
    dedup_columns = [
        "repository",
        "commit_hash",
        "path",
        "start_line",
        "end_line",
        "label",
    ]

    existing_cols = [col for col in dedup_columns if col in df.columns]

    if existing_cols:
        df = df.drop_duplicates(subset=existing_cols)
    else:
        df = df.drop_duplicates(subset=["code", "label"])

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"code", "label"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df.copy()

    df["code"] = df["code"].astype(str).str.strip()
    df["label"] = df["label"].apply(normalize_label)

    # Remove empty code
    df = df[df["code"] != ""]

    # Keep only allowed labels
    df = df[df["label"].isin(LABELS)]

    # Drop duplicates
    df = deduplicate_samples(df)

    return df


def main():
    print("Loading fetched dataset...")

    if not os.path.exists(INTERIM_DATA_PATH):
        raise FileNotFoundError(
            f"File not found: {INTERIM_DATA_PATH}\n"
            "Run `python scripts/fetch_mlcq_code.py` first."
        )

    df = pd.read_csv(INTERIM_DATA_PATH)

    print("Columns found:", list(df.columns))

    df = clean_dataset(df)

    print("\nDataset size after cleaning:", len(df))
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    # ========================================================
    # Stratified split
    # 80% train, 10% validation, 10% test
    # ========================================================
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["label"],
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_df[["code", "label"]].to_csv(TRAIN_PATH, index=False)
    valid_df[["code", "label"]].to_csv(VALID_PATH, index=False)
    test_df[["code", "label"]].to_csv(TEST_PATH, index=False)

    print("\nSaved processed files:")
    print(f"Train: {len(train_df)} → {TRAIN_PATH}")
    print(f"Valid: {len(valid_df)} → {VALID_PATH}")
    print(f"Test : {len(test_df)} → {TEST_PATH}")


if __name__ == "__main__":
    main()