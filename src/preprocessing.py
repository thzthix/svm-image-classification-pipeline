from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMAGE_SIZE = (28, 28)
PIXEL_SCALE = 255.0


def load_grayscale_image(image_path: str | Path) -> np.ndarray:
    """단일 이미지를 그레이스케일 배열로 불러온다."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def normalize_pixels(image: np.ndarray) -> np.ndarray:
    """픽셀 값을 float32로 바꾸고 [0, 1] 범위로 정규화한다."""
    return image.astype(np.float32) / PIXEL_SCALE


def ensure_image_shape(
    image: np.ndarray,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """입력 이미지가 기대하는 크기와 다를 때만 리사이즈한다."""
    if image.shape == image_size:
        return image
    return cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)


def prepare_inference_image(
    image_path: str | Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """추론용 단일 이미지를 불러와 리사이즈하고 정규화한다."""
    image = load_grayscale_image(image_path)
    image = ensure_image_shape(image, image_size=image_size)
    return normalize_pixels(image)


def load_train_dataframe(csv_path: str | Path) -> pd.DataFrame:
    """학습용 CSV 파일을 데이터프레임으로 불러온다."""
    dataframe = pd.read_csv(csv_path)
    if "label" in dataframe.columns:
        return dataframe
    return pd.read_csv(csv_path, header=None)


def split_features_and_labels(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """학습 데이터프레임에서 픽셀 값과 레이블을 분리한다."""
    if "label" in df.columns:
        labels = df["label"]
        features = df.drop(columns=["label"])
        return features, labels

    labels = df.iloc[:, 0]
    features = df.iloc[:, 1:]
    return features, labels


def prepare_training_images(features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """학습용 픽셀 데이터를 정규화하고 28x28 이미지로 변환한다."""
    images = np.asarray(features, dtype=np.float32)
    images = normalize_pixels(images)
    return images.reshape(-1, *IMAGE_SIZE)
