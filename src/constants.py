LABELS = [
    "data_class",
    "feature_envy",
    "god_class",
    "long_method",
]

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

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}