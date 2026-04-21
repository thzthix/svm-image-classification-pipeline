from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import load_settings
from .features import extract_hog_feature
from .inference import load_inference_models
from .preprocessing import (
    IMAGE_SIZE,
    load_train_dataframe,
    prepare_inference_image,
    split_features_and_labels,
)

_TRAIN_EMBEDDINGS = None
_TRAIN_METADATA = None
_TRAIN_THUMBNAIL_IMAGES = None


def _load_retrieval_assets() -> tuple[np.ndarray, pd.DataFrame]:
    """저장된 train 임베딩과 메타데이터를 불러온다."""
    global _TRAIN_EMBEDDINGS, _TRAIN_METADATA

    if _TRAIN_EMBEDDINGS is None or _TRAIN_METADATA is None:
        settings = load_settings()
        _TRAIN_EMBEDDINGS = np.load(settings["train_embeddings_path"])
        _TRAIN_METADATA = pd.read_csv(settings["train_metadata_path"])

    return _TRAIN_EMBEDDINGS, _TRAIN_METADATA


def load_retrieval_assets() -> tuple[np.ndarray, pd.DataFrame]:
    """유사도 검색에 필요한 임베딩과 메타데이터를 미리 준비한다."""
    return _load_retrieval_assets()


def _load_train_thumbnail_images() -> np.ndarray:
    """학습 CSV에서 유사 검색용 원본 이미지를 메모리에 준비한다."""
    global _TRAIN_THUMBNAIL_IMAGES

    if _TRAIN_THUMBNAIL_IMAGES is None:
        settings = load_settings()
        train_csv_path = settings["train_csv_path"]
        if train_csv_path is None:
            raise FileNotFoundError("TRAIN_CSV_NAME을 읽기 위한 DATA_DIR 설정이 필요합니다.")

        train_df = load_train_dataframe(train_csv_path)
        features, _ = split_features_and_labels(train_df)
        _TRAIN_THUMBNAIL_IMAGES = np.asarray(features, dtype=np.uint8).reshape(-1, *IMAGE_SIZE)

    return _TRAIN_THUMBNAIL_IMAGES


def get_thumbnail_image(index: int) -> np.ndarray:
    """학습 CSV에서 인덱스에 해당하는 28x28 이미지를 반환한다."""
    images = _load_train_thumbnail_images()

    if index < 0 or index >= len(images):
        raise IndexError(f"유효하지 않은 thumbnail index입니다: {index}")

    return images[index]


def encode_thumbnail_png(index: int) -> bytes:
    """학습 CSV 이미지를 PNG 바이트로 인코딩한다."""
    image = get_thumbnail_image(index)
    success, encoded_image = cv2.imencode(".png", image)
    if not success:
        raise ValueError("썸네일 PNG 인코딩에 실패했습니다.")
    return encoded_image.tobytes()


def _project_query_image(image_path: str | Path) -> np.ndarray:
    """쿼리 이미지를 PCA 임베딩 공간으로 투영한다."""
    pca_model, _ = load_inference_models()

    image = prepare_inference_image(image_path)
    hog_feature = extract_hog_feature(image).reshape(1, -1).astype(np.float64)
    return pca_model.transform(hog_feature).astype(np.float32)


def _cosine_similarity(query_embedding: np.ndarray, train_embeddings: np.ndarray) -> np.ndarray:
    """쿼리 임베딩과 train 임베딩 사이의 cosine similarity를 계산한다."""
    query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    train_norm = np.linalg.norm(train_embeddings, axis=1, keepdims=True)

    if np.any(query_norm == 0) or np.any(train_norm == 0):
        raise ValueError("유사도 계산 중 0 벡터가 발견되었습니다.")

    normalized_query = query_embedding / query_norm
    normalized_train = train_embeddings / train_norm
    return normalized_train @ normalized_query.T


def search_similar_images(
    image_path: str | Path,
    top_k: int = 5,
) -> list[dict[str, int | float | str]]:
    """쿼리 이미지와 가장 유사한 train 샘플 top-k를 반환한다."""
    train_embeddings, metadata = _load_retrieval_assets()
    query_embedding = _project_query_image(image_path)

    scores = _cosine_similarity(query_embedding, train_embeddings).ravel()
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = metadata.iloc[idx]
        results.append(
            {
                "index": int(row["index"]),
                "label": int(row["label"]),
                "score": float(scores[idx]),
                "thumbnail_path": f"/thumbnail/{int(row['index'])}",
            }
        )

    return results
