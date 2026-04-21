from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from .config import load_settings

REQUIRED_ARTIFACT_FILENAMES = (
    "pca.pkl",
    "svm.pkl",
    "train_embeddings.npy",
    "train_metadata.csv",
)


def _download_artifact_if_needed(path: str | Path) -> Path:
    settings = load_settings()
    artifact_path = Path(path)
    if artifact_path.exists():
        return artifact_path

    hf_repo_id = settings["hf_repo_id"]
    if not isinstance(hf_repo_id, str) or not hf_repo_id:
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    artifact_dir = settings["artifact_dir"]
    if not isinstance(artifact_dir, Path):
        raise FileNotFoundError(f"Artifact directory is invalid: {artifact_dir}")

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename=artifact_path.name,
        local_dir=artifact_dir,
        token=settings["hf_token"],
    )
    return Path(downloaded_path)


def ensure_all_artifacts_available() -> None:
    """서버 실행 전에 필요한 모델 아티팩트가 모두 준비되도록 보장한다."""
    settings = load_settings()
    artifact_dir = settings["artifact_dir"]
    if not isinstance(artifact_dir, Path):
        raise FileNotFoundError(f"Artifact directory is invalid: {artifact_dir}")

    for filename in REQUIRED_ARTIFACT_FILENAMES:
        _download_artifact_if_needed(artifact_dir / filename)


def load_pickle_artifact(path: str | Path) -> Any:
    """디스크에 저장된 pickle 아티팩트를 하나 불러온다."""
    artifact_path = _download_artifact_if_needed(path)

    with artifact_path.open("rb") as file:
        return pickle.load(file)


def save_pickle_artifact(artifact: Any, path: str | Path) -> Path:
    """객체 하나를 pickle 아티팩트로 저장한다."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    with artifact_path.open("wb") as file:
        pickle.dump(artifact, file)

    return artifact_path
