import os
import time
import requests
import pandas as pd
from typing import Optional

from src.constants import LABEL_MAP, LABELS, RAW_METADATA_PATH, INTERIM_DATA_PATH, REQUEST_TIMEOUT, SLEEP_BETWEEN_REQUESTS

def normalize_label(label: str) -> str:
    label = str(label).strip().lower()
    if label not in LABEL_MAP:
        raise ValueError(f"Unknown smell label: {label}")
    return LABEL_MAP[label]


def repo_to_raw_base(repository: str) -> str:
    """
    Convert:
      git@github.com:apache/syncope.git
    into:
      apache/syncope
    """
    repo = repository.strip()

    if repo.startswith("git@github.com:"):
        repo = repo.replace("git@github.com:", "")

    if repo.endswith(".git"):
        repo = repo[:-4]

    return repo


def build_raw_url(repository: str, commit_hash: str, path: str) -> str:
    repo_name = repo_to_raw_base(repository)
    clean_path = path.lstrip("/")
    return f"https://raw.githubusercontent.com/{repo_name}/{commit_hash}/{clean_path}"


def fetch_file_text(url: str) -> Optional[str]:
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.text
        return None
    except requests.RequestException:
        return None


def extract_lines(file_text: str, start_line: int, end_line: int) -> str:
    lines = file_text.splitlines()

    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)

    snippet = lines[start_idx:end_idx]
    return "\n".join(snippet).strip()


def main():
    if not os.path.exists(RAW_METADATA_PATH):
        raise FileNotFoundError(
            f"File not found: {RAW_METADATA_PATH}"
        )

    os.makedirs(os.path.dirname(INTERIM_DATA_PATH), exist_ok=True)

    df = pd.read_csv(RAW_METADATA_PATH, sep=";")

    required_columns = [
        "smell",
        "repository",
        "commit_hash",
        "path",
        "start_line",
        "end_line",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Keep only rows with actual smell label we want
    df = df[df["smell"].notna()].copy()
    df["smell"] = df["smell"].astype(str).str.strip().str.lower()
    df = df[df["smell"].isin(LABEL_MAP.keys())].copy()

    rows = []
    success_count = 0
    fail_count = 0

    for idx, row in df.iterrows():
        try:
            label = normalize_label(row["smell"])

            raw_url = build_raw_url(
                repository=row["repository"],
                commit_hash=row["commit_hash"],
                path=row["path"],
            )

            file_text = fetch_file_text(raw_url)
            if not file_text:
                fail_count += 1
                continue

            code = extract_lines(
                file_text=file_text,
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
            )

            if not code:
                fail_count += 1
                continue

            rows.append({
                "code": code,
                "label": label,
                "repository": row["repository"],
                "commit_hash": row["commit_hash"],
                "path": row["path"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "raw_url": raw_url,
            })

            success_count += 1

            if success_count % 100 == 0:
                print(f"Fetched {success_count} samples...")

            time.sleep(SLEEP_BETWEEN_REQUESTS)

        except Exception:
            fail_count += 1
            continue

    out_df = pd.DataFrame(rows)

    # Keep only the target labels
    out_df = out_df[out_df["label"].isin(LABELS)].copy()

    out_df.to_csv(INTERIM_DATA_PATH, index=False)

    print("\nDone.")
    print(f"Saved file: {INTERIM_DATA_PATH}")
    print(f"Successful samples: {success_count}")
    print(f"Failed samples: {fail_count}")

    if not out_df.empty:
        print("\nLabel distribution:")
        print(out_df["label"].value_counts())


if __name__ == "__main__":
    main()