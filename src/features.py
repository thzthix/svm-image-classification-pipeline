from __future__ import annotations

import numpy as np
from skimage.feature import hog

HOG_ORIENTATIONS = 6
HOG_PIXELS_PER_CELL = (3, 3)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2"


def extract_hog_feature(image: np.ndarray) -> np.ndarray:
    """그레이스케일 이미지 하나에서 HOG 특징 벡터를 추출한다."""
    feature_vector = hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
    )
    return np.asarray(feature_vector, dtype=np.float32)


def extract_hog_features(images: np.ndarray) -> np.ndarray:
    """여러 장의 그레이스케일 이미지에서 HOG 특징 벡터를 추출한다."""
    feature_vectors = [extract_hog_feature(image) for image in images]
    return np.asarray(feature_vectors, dtype=np.float32)
