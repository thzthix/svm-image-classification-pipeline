from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import load_settings
from .features import extract_hog_feature
from .model_io import load_pickle_artifact
from .preprocessing import prepare_inference_image

_PCA_MODEL = None
_SVM_MODEL = None


def load_inference_models() -> tuple[object, object]:
    """추론에 필요한 PCA와 SVM 모델을 한 번만 불러와 재사용한다."""
    global _PCA_MODEL, _SVM_MODEL

    if _PCA_MODEL is None or _SVM_MODEL is None:
        settings = load_settings()
        _PCA_MODEL = load_pickle_artifact(settings["pca_path"])
        _SVM_MODEL = load_pickle_artifact(settings["svm_path"])

    return _PCA_MODEL, _SVM_MODEL


def predict_image(image_path: str | Path) -> dict[str, int | float]:
    """전처리, HOG, PCA, SVM 순서로 단일 이미지 클래스를 예측한다."""
    pca_model, svm_model = load_inference_models()

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
