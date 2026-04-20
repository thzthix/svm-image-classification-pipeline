from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_settings() -> dict[str, Path]:
    """프로젝트 환경에서 데이터셋과 아티팩트 경로를 불러온다."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    data_dir = os.getenv("DATA_DIR")
    if not data_dir:
        raise ValueError("DATA_DIR is not set. Add it to .env or the environment.")

    train_csv_name = os.getenv("TRAIN_CSV_NAME", "120000_augmented.csv")
    project_root = Path(__file__).resolve().parent.parent
    artifact_dir = project_root / "artifacts"

    return {
        "data_dir": Path(data_dir),
        "train_csv_path": Path(data_dir) / train_csv_name,
        "artifact_dir": artifact_dir,
        "pca_path": artifact_dir / "pca.pkl",
        "svm_path": artifact_dir / "svm.pkl",
        "train_embeddings_path": artifact_dir / "train_embeddings.npy",
        "train_metadata_path": artifact_dir / "train_metadata.csv",
    }
