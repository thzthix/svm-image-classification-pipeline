from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from .config import load_settings
from .features import extract_hog_features
from .model_io import save_pickle_artifact
from .preprocessing import (
    load_train_dataframe,
    prepare_training_images,
    split_features_and_labels,
)


def train_and_save_models() -> tuple[Path, Path]:
    """Fashion-MNIST 학습 데이터로 PCA와 SVM을 학습하고 저장한다."""
    settings = load_settings()
    train_csv_path = settings["data_dir"] / "mnist_train_small.csv"

    train_df = load_train_dataframe(train_csv_path)
    features, labels = split_features_and_labels(train_df)
    images = prepare_training_images(features)
    hog_features = extract_hog_features(images)
    hog_features = hog_features.astype(np.float64)

    if not np.isfinite(hog_features).all():
        raise ValueError("PCA 입력 HOG feature에 NaN 또는 inf가 포함되어 있습니다.")

    pca_model = PCA(n_components=466, random_state=45, svd_solver="full")
    projected_features = pca_model.fit_transform(hog_features)

    svm_model = SVC(gamma="scale", kernel="rbf", C=8, random_state=45)
    svm_model.fit(projected_features, labels)

    pca_path = save_pickle_artifact(pca_model, settings["pca_path"])
    svm_path = save_pickle_artifact(svm_model, settings["svm_path"])

    return pca_path, svm_path
