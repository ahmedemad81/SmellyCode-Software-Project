import os

# Allowed labels for classification and the order used by the model.
LABELS = [
    "data_class",
    "feature_envy",
    "god_class",
    "long_method",
]

# Normalization map for label text used in raw data.
LABEL_MAP = {
    "data_class": "data_class",
    "data class": "data_class",

    "feature_envy": "feature_envy",
    "feature envy": "feature_envy",

    "god_class": "god_class",
    "god class": "god_class",
    "blob": "god_class",

    "long_method": "long_method",
    "long method": "long_method",
}


# Training/inference label encodings.
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for i, label in LABEL_TO_ID.items()}

# Maximum tokenizer length shared across training, evaluation, and prediction.
MAX_LENGTH = 384

# Paths for raw, interim, and processed dataset files.
RAW_METADATA_PATH = "data/raw/mlcq/mlcq.csv"
INTERIM_DIR = "data/interim/mlcq"
INTERIM_DATA_PATH = os.path.join(INTERIM_DIR, "mlcq_with_code.csv")
PROCESSED_DIR = "data/processed/mlcq"
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
VALID_PATH = os.path.join(PROCESSED_DIR, "valid.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")

# Model and output directories used by training and inference.
MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_OUTPUT_DIR = "outputs/qwen3_0_6b_seqcls"
RESULTS_DIR = "results/mlcq_multiclass"
RESULTS_PATH = os.path.join(RESULTS_DIR, "qwen3_0_6b_seqcls_metrics.json")

# Settings for the data fetcher.
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_REQUESTS = 0.2