from __future__ import annotations

from typing import Callable

import numpy as np
try:
    import optuna
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "optuna가 설치되어 있지 않습니다. `python3.11 -m pip install optuna`로 먼저 설치해 주세요."
    ) from exc
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC

from src.config import load_settings
from src.features import extract_hog_features
from src.preprocessing import (
    load_train_dataframe,
    prepare_training_images,
    split_features_and_labels,
)

RANDOM_STATE = 45
N_SPLITS = 3
DEFAULT_N_TRIALS = 20


def load_hog_features_and_labels() -> tuple[np.ndarray, np.ndarray]:
    """학습 CSV를 읽고 고정 HOG 특징과 레이블을 준비한다."""
    settings = load_settings()
    train_df = load_train_dataframe(settings["train_csv_path"])
    pixel_features, labels = split_features_and_labels(train_df)
    images = prepare_training_images(pixel_features)
    hog_features = extract_hog_features(images).astype(np.float64)

    if not np.isfinite(hog_features).all():
        raise ValueError("Optuna 입력 HOG feature에 NaN 또는 inf가 포함되어 있습니다.")

    return hog_features, labels.to_numpy()


def sample_stratified_subset(
    hog_features: np.ndarray,
    labels: np.ndarray,
    sample_size: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """필요할 때만 label 분포를 유지하며 일부 데이터만 샘플링한다."""
    if sample_size is None or sample_size >= len(labels):
        return hog_features, labels

    unique_labels = np.unique(labels)
    if sample_size < len(unique_labels):
        raise ValueError("sample_size는 클래스 개수 이상이어야 합니다.")

    indices = np.arange(len(labels))
    sampled_indices, _ = train_test_split(
        indices,
        train_size=sample_size,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    return hog_features[sampled_indices], labels[sampled_indices]


def build_objective(
    hog_features: np.ndarray,
    labels: np.ndarray,
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        pca_n_components = trial.suggest_categorical(
            "pca_n_components",
            [128, 256, 384, 466, 512],
        )
        svm_c = trial.suggest_float("svm_c", 0.1, 20.0, log=True)
        svm_gamma = trial.suggest_float("svm_gamma", 1e-4, 1e-1, log=True)

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        fold_scores: list[float] = []

        for train_idx, valid_idx in cv.split(hog_features, labels):
            x_train = hog_features[train_idx]
            x_valid = hog_features[valid_idx]
            y_train = labels[train_idx]
            y_valid = labels[valid_idx]

            pca_model = PCA(
                n_components=pca_n_components,
                random_state=RANDOM_STATE,
                svd_solver="full",
            )
            x_train_pca = pca_model.fit_transform(x_train)
            x_valid_pca = pca_model.transform(x_valid)

            svm_model = SVC(
                C=svm_c,
                gamma=svm_gamma,
                kernel="rbf",
                random_state=RANDOM_STATE,
            )
            svm_model.fit(x_train_pca, y_train)
            predictions = svm_model.predict(x_valid_pca)
            fold_scores.append(f1_score(y_valid, predictions, average="macro"))

        return float(np.mean(fold_scores))

    return objective


def run_optuna_study(
    n_trials: int = DEFAULT_N_TRIALS,
    sample_size: int | None = None,
) -> optuna.study.Study:
    """PCA와 SVM 하이퍼파라미터를 Optuna와 StratifiedKFold로 탐색한다."""
    hog_features, labels = load_hog_features_and_labels()
    hog_features, labels = sample_stratified_subset(
        hog_features,
        labels,
        sample_size=sample_size,
    )
    print(f"HOG feature shape: {hog_features.shape}, labels shape: {labels.shape}")
    objective = build_objective(hog_features, labels)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best params: {study.best_params}")
    print(f"Best macro_f1: {study.best_value:.6f}")
    return study


def main() -> None:
    """기본 trial 수로 1차 Optuna 실험을 실행한다."""
    run_optuna_study()


if __name__ == "__main__":
    main()
