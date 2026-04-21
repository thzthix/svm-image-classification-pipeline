---
title: HOG PCA SVM API
emoji: 🧠
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

# HOG + PCA + SVM Image Classification Pipeline

## 프로젝트 개요

이 프로젝트는 Fashion-MNIST 스타일 이미지 분류 문제를 대상으로,  
HOG feature extraction, PCA dimensionality reduction, SVM classification을 사용해  
기존 노트북 실험 코드를 재구성한 classical ML 기반 이미지 분류 파이프라인입니다.

목표는 실험용 노트북 코드를 그대로 두는 대신,  
학습과 추론 흐름을 분리하고 재사용 가능한 형태로 정리하는 것이었습니다.
---
title: Svm Image Classification Api
emoji: 🐠
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

이 프로젝트는 Fashion-MNIST 이미지 분류 경진대회를 기반으로,  
HOG feature engineering, PCA dimensionality reduction, SVM classification을 활용하여 구현되었습니다.

기존 노트북 기반 코드를 다음 구조로 재구성했습니다:
- inference pipeline
- HOG/PCA 기반 feature space
- similarity search 시스템

2. `HOG`
- 이미지의 윤곽과 방향성 정보를 HOG feature vector로 변환

3. `PCA`
- HOG feature를 저차원 공간으로 투영해 차원을 축소

4. `SVM`
- PCA 결과를 입력으로 받아 최종 클래스를 예측

## 현재 구현된 기능

현재 저장소에는 아래 기능이 구현되어 있습니다.

- 학습용 CSV 로드
- label / pixel 분리
- 학습용 이미지 전처리
- HOG feature extraction
- PCA 학습 및 변환
- SVM 학습
- 학습된 `pca.pkl`, `svm.pkl` 저장
- 단일 이미지 추론 함수 `predict_image(image_path)` 구현
- train embedding 저장
- cosine similarity 기반 similarity search

현재 `src` 구조는 다음과 같습니다.

```text
src/
├─ config.py
├─ preprocessing.py
├─ features.py
├─ model_io.py
├─ train.py
├─ inference.py
└─ retrieval.py
```

## 실행 환경
## Environment

- Python 3.11
- macOS 기준 `venv` 사용
- 의존성은 `requirements.txt`로 관리

## Python 3.11 가상환경 실행 방법

프로젝트 루트에서 아래 순서로 실행합니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

버전 확인:

```bash
python --version
```

## 데이터 경로 설정

`.env` 파일에 데이터 경로를 설정합니다.

```env
DATA_DIR=/path/to/team_project_data
TRAIN_CSV_NAME=120000_augmented.csv
```

학습용 CSV는 다음 형식을 지원합니다.

- `label` 컬럼을 포함한 헤더형 CSV
- 또는 첫 번째 열이 label인 header 없는 CSV

## 학습 방법

학습을 실행하면 PCA와 SVM 모델을 학습한 뒤, `artifacts/` 폴더에 저장합니다.

```bash
python -c "from src.train import train_and_save_models; print(train_and_save_models())"
```

생성 파일:

- `artifacts/pca.pkl`
- `artifacts/svm.pkl`
- `artifacts/train_embeddings.npy`
- `artifacts/train_metadata.csv`

## 단일 이미지 추론 방법

학습이 끝난 뒤에는 단일 이미지 추론을 실행할 수 있습니다.

```bash
python -c "from src.inference import predict_image; print(predict_image('/path/to/team_project_data/public_test_dataset/data/00001.png'))"
```

반환 예시:

```python
{
    "predicted_class": 4,
    "decision_score": 9.291095597887397
}
```

설명:
- `predicted_class`: 예측 클래스
- `decision_score`: SVM decision function 기반 점수
- `decision_score`는 확률값이 아니라 분류 결정 함수의 출력값입니다

## Similarity Search 방법

저장된 train embedding을 재사용해 query 이미지와 가장 유사한 샘플 top-k를 찾을 수 있습니다.

```bash
python -c "from src.retrieval import search_similar_images; print(search_similar_images('/path/to/team_project_data/public_test_dataset/data/00001.png', top_k=5))"
```

반환 예시:

```python
[
    {"index": 1234, "label": 6, "score": 0.9821},
    {"index": 5521, "label": 6, "score": 0.9710},
    {"index": 902, "label": 2, "score": 0.9644},
]
```

## API 실행

FastAPI 서버는 아래 명령으로 실행할 수 있습니다.

```bash
python3.11 -m uvicorn src.api:app --reload
```

제공 endpoint:

- `POST /predict`
- `POST /similar`

서버 시작 시 startup 단계에서 추론에 필요한 artifact를 먼저 확인합니다.  
로컬 `artifacts/`에 파일이 없으면 Hugging Face Hub에서 아래 4개 파일을 내려받습니다.

- `pca.pkl`
- `svm.pkl`
- `train_embeddings.npy`
- `train_metadata.csv`

첫 기동에서는 다운로드 때문에 시작이 조금 느릴 수 있으며, 내려받은 파일은 로컬 `artifacts/`에 캐시됩니다.

## Optuna 실험 방법

Optuna + StratifiedKFold 기반 1차 튜닝 실험은 아래 명령으로 실행할 수 있습니다.

```bash
python3.11 -m experiments.optuna_cv
```

trial 수를 바꾸고 싶다면 아래처럼 실행합니다.

```bash
python3.11 -c "from experiments.optuna_cv import run_optuna_study; run_optuna_study(n_trials=10)"
```

주의:
- Optuna 패키지가 필요합니다.
- 데이터 크기에 따라 실행 시간이 오래 걸릴 수 있습니다.

## Render 배포 방법

Render에 배포할 때는 루트의 `Procfile`을 사용하면 됩니다.

```text
web: uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

설정 방법:

1. Render에서 새 `Web Service`를 생성하고 이 저장소를 연결합니다.
2. Runtime은 `Python`을 선택합니다.
3. Build Command는 `pip install -r requirements.txt`로 설정합니다.
4. Start Command는 비워 두거나 `Procfile`을 사용하도록 둡니다.
5. 배포 환경에서 Hugging Face artifact를 사용할 경우 `HF_REPO_ID`를 환경변수로 설정합니다.
6. 배포가 끝나면 `https://<your-render-url>/docs`에서 API를 바로 테스트할 수 있습니다.

참고:

- `/predict`, `/similar`는 startup 단계에서 artifact를 준비한 뒤 `artifacts/` 안의 모델 파일과 embedding 파일을 사용합니다.
- `DATA_DIR`는 학습 CSV 로드나 dataset 기반 스크립트 실행에 필요합니다.
- 배포 환경에서 학습이나 평가 스크립트를 실행할 경우에는 별도로 `DATA_DIR`를 설정해야 합니다.

배포 전 수동 체크리스트:

- Hugging Face Hub repo 이름이 `HF_REPO_ID`와 일치하는지 확인
- 아래 4개 파일이 HF Hub repo에 업로드되어 있는지 확인
  - `pca.pkl`
  - `svm.pkl`
  - `train_embeddings.npy`
  - `train_metadata.csv`
- HF repo가 private이면 Render 환경변수에 `HF_TOKEN`을 추가했는지 확인
- HF repo가 public이면 `HF_TOKEN` 없이도 다운로드 가능한지 확인
- Render 배포 후 `/docs`에 접속해 startup 이후 `/predict`, `/similar`가 정상 동작하는지 확인

## Hugging Face Docker Space 배포 방법

Hugging Face Space에서는 Docker SDK를 사용해 FastAPI 서버를 그대로 실행할 수 있습니다.

설정 방법:

1. Hugging Face에서 새 Space를 만들고 SDK를 `Docker`로 선택합니다.
2. 이 저장소를 Space repo에 push하거나 파일을 업로드합니다.
3. Space Settings에서 serving용 환경변수를 설정합니다.
4. 빌드가 끝나면 `https://<your-space-name>.hf.space/docs`에서 API를 테스트합니다.

권장 환경변수:

- `HF_REPO_ID`
- `HF_TOKEN` (artifact repo가 private일 때 필수)

참고:

- Docker Space는 루트 `README.md`의 YAML 설정과 `Dockerfile`을 기준으로 빌드됩니다.
- API serving만 사용할 경우 `DATA_DIR`는 꼭 필요하지 않습니다.
- 첫 기동에서는 artifact 다운로드 때문에 시작이 다소 느릴 수 있으며, 내려받은 파일은 컨테이너 내부 `artifacts/` 경로에 캐시됩니다.

## 현재 한계

현재 구현은 최소 동작 파이프라인을 우선 목표로 정리한 상태입니다.

- 데이터 증강(augmentation) 로직은 아직 파이프라인에 포함하지 않음
- 배치 추론은 아직 구현하지 않음
- similarity search는 최소 구현 수준이며 별도 인덱싱 최적화는 없음
- 학습 데이터 입력은 현재 `120000_augmented.csv` 기준으로 동작함
- 성능 검증 자동화 및 하이퍼파라미터 탐색은 아직 포함하지 않음

## 다음 확장 방향

현재 구조를 유지한 상태에서 아래 방향으로 확장할 수 있습니다.

- augmentation 결과 CSV를 학습 입력으로 연결
- public/private dataset 기반 추론 스크립트 추가
- similarity search 기능 추가
- Optuna + StratifiedKFold 기반 검증 보강
- FastAPI 기반 간단한 `/predict` API 추가

## 정리

이 저장소는 딥러닝 모델을 추가하지 않고,  
HOG + PCA + SVM 기반 classical ML 파이프라인을  
학습/추론 관점에서 정리한 포트폴리오용 프로젝트입니다.

핵심은 다음 두 가지입니다.

- feature engineering과 차원 축소를 포함한 전통적인 이미지 분류 흐름을 직접 구성했다는 점
- 노트북 실험 코드를 재사용 가능한 학습/추론 구조로 분리했다는 점
- `pip install -r requirements.txt`
