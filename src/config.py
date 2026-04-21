from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

SettingsValue = Path | str | None


def load_settings() -> dict[str, SettingsValue]:
    """프로젝트 환경에서 데이터셋과 아티팩트 경로를 불러온다."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    project_root = Path(__file__).resolve().parent.parent
    artifact_dir = project_root / "artifacts"
    data_dir_env = os.getenv("DATA_DIR")
    data_dir = Path(data_dir_env) if data_dir_env else None
    train_csv_name = os.getenv("TRAIN_CSV_NAME", "120000_augmented.csv")
    hf_repo_id = os.getenv("HF_REPO_ID", "bookbo/svm_image_classification_artifacts")
    hf_token = os.getenv("HF_TOKEN")

    return {
        "data_dir": data_dir,
        "train_csv_path": data_dir / train_csv_name if data_dir else None,
        "artifact_dir": artifact_dir,
        "pca_path": artifact_dir / "pca.pkl",
        "svm_path": artifact_dir / "svm.pkl",
        "train_embeddings_path": artifact_dir / "train_embeddings.npy",
        "train_metadata_path": artifact_dir / "train_metadata.csv",
        "hf_repo_id": hf_repo_id,
        "hf_token": hf_token,
    }
