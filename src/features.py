from __future__ import annotations

import numpy as np
from skimage.feature import hog


def extract_hog_feature(image: np.ndarray) -> np.ndarray:
    """그레이스케일 이미지 하나에서 HOG 특징 벡터를 추출한다."""
    feature_vector = hog(
        image,
        orientations=6,
        pixels_per_cell=(3, 3),
        cells_per_block=(2, 2),
        block_norm="L2",
    )
    return np.asarray(feature_vector, dtype=np.float32)
