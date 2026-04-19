from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import load_settings
from .features import extract_hog_feature
from .model_io import load_pickle_artifact
from .preprocessing import prepare_inference_image


def predict_image(image_path: str | Path) -> dict[str, int | float]:
    """전처리, HOG, PCA, SVM 순서로 단일 이미지 클래스를 예측한다."""
    settings = load_settings()
    pca_model = load_pickle_artifact(settings["pca_path"])
    svm_model = load_pickle_artifact(settings["svm_path"])

    image = prepare_inference_image(image_path)
    hog_feature = extract_hog_feature(image).reshape(1, -1)
    projected_feature = pca_model.transform(hog_feature)
    predicted_class = int(svm_model.predict(projected_feature)[0])

    decision_score = 0.0
    if hasattr(svm_model, "decision_function"):
        decision = np.asarray(svm_model.decision_function(projected_feature))
        if decision.ndim == 1:
            decision_score = float(decision[0])
        else:
            decision_score = float(np.max(decision[0]))

    return {
        "predicted_class": predicted_class,
        "decision_score": decision_score,
    }
