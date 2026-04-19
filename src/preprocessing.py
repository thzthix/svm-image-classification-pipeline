from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_grayscale_image(image_path: str | Path) -> np.ndarray:
    """단일 이미지를 그레이스케일 배열로 불러온다."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def normalize_pixels(image: np.ndarray) -> np.ndarray:
    """픽셀 값을 float32로 바꾸고 [0, 1] 범위로 정규화한다."""
    return image.astype(np.float32) / 255.0


def ensure_image_shape(
    image: np.ndarray,
    image_size: tuple[int, int] = (28, 28),
) -> np.ndarray:
    """입력 이미지가 기대하는 크기와 다를 때만 리사이즈한다."""
    if image.shape == image_size:
        return image
    return cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)


def prepare_inference_image(
    image_path: str | Path,
    image_size: tuple[int, int] = (28, 28),
) -> np.ndarray:
    """추론용 단일 이미지를 불러와 리사이즈하고 정규화한다."""
    image = load_grayscale_image(image_path)
    image = ensure_image_shape(image, image_size=image_size)
    return normalize_pixels(image)
