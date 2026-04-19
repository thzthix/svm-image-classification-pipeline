from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def load_pickle_artifact(path: str | Path) -> Any:
    """디스크에 저장된 pickle 아티팩트를 하나 불러온다."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    with artifact_path.open("rb") as file:
        return pickle.load(file)
