---
title: HOG PCA SVM API
emoji: 🧠
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

# HOG + PCA + SVM Image Classification Pipeline

Fashion-MNIST 스타일 이미지를 대상으로, HOG feature extraction, PCA dimensionality reduction, SVM classification을 사용해 분류와 유사 이미지 검색까지 연결한 classical ML 기반 포트폴리오 프로젝트입니다. 실험용 노트북 코드를 학습, 추론, 검색, 서빙 흐름으로 분리해 실제로 배포 가능한 형태까지 정리하는 데 초점을 맞췄습니다.

## 배포 링크

- Frontend Demo (Vercel): [https://svm-image-classification-pipeline.vercel.app/](https://svm-image-classification-pipeline.vercel.app/)
- Hugging Face Space Page: [https://huggingface.co/spaces/bookbo/svm-image-classification-api](https://huggingface.co/spaces/bookbo/svm-image-classification-api)
- API Base URL (Hugging Face Space): [https://bookbo-svm-image-classification-api.hf.space](https://bookbo-svm-image-classification-api.hf.space)
- API Docs: [https://bookbo-svm-image-classification-api.hf.space/docs](https://bookbo-svm-image-classification-api.hf.space/docs)

## 프로젝트 한눈에 보기

- 이미지를 업로드하면 `HOG -> PCA -> SVM` 파이프라인으로 클래스를 예측합니다.
- 학습 임베딩과 cosine similarity를 이용해 유사한 train 샘플 top-k를 찾습니다.
- 유사 결과 썸네일은 별도 이미지 파일을 만들지 않고, 학습 CSV의 픽셀 값을 재사용해 제공합니다.
- Frontend는 Vercel, API는 Hugging Face Docker Space로 분리 배포했습니다.

## 주요 기능

- 업로드 이미지 단일 분류
- top-k 유사 이미지 검색
- `GET /thumbnail/{index}` 기반 썸네일 조회
- React + Vite 프론트엔드와 FastAPI 백엔드 연동
- Hugging Face dataset/artifact 기반 배포 구성
- Optuna + StratifiedKFold 기반 하이퍼파라미터 탐색 실험 코드

## 배포 구성

- Frontend: React + Vite, Vercel 배포
- API: FastAPI, Hugging Face Docker Space 배포
- Dataset: Hugging Face dataset repo mount
- Artifacts: Hugging Face Hub에서 모델/임베딩 파일 다운로드
- Frontend는 `VITE_API_BASE_URL`을 통해 Hugging Face Space API에 요청합니다.

## 아키텍처

<img src="./docs/architecture.svg?v=3" alt="Architecture Diagram" width="100%" />

## 핵심 파이프라인

전체 흐름은 아래와 같습니다.

`preprocessing -> HOG -> PCA -> SVM`

1. `preprocessing`
- 입력 이미지를 그레이스케일로 로드
- `28x28` 크기로 맞춤
- pixel 값을 `0~1` 범위로 정규화

2. `HOG`
- 이미지의 윤곽과 방향성 정보를 HOG feature vector로 변환

3. `PCA`
- HOG feature를 저차원 공간으로 투영해 차원을 축소

4. `SVM`
- PCA 결과를 입력으로 받아 최종 클래스를 예측

## 구성

- 딥러닝 모델을 추가하기보다 classical ML 파이프라인을 끝까지 제품 형태로 연결하는 데 집중했습니다.
- 유사 이미지 검색은 PCA embedding + cosine similarity로 단순하게 구성했습니다.
- 썸네일은 학습 CSV의 픽셀 값을 재사용해 생성하고, 최초 로드 후 메모리에 캐시해 재사용합니다.
- dataset, artifacts, serving API를 분리해 배포 환경에서 역할이 섞이지 않도록 정리했습니다.

## 프로젝트 구조

```text
src/
├─ api.py
├─ config.py
├─ features.py
├─ inference.py
├─ labels.py
├─ model_io.py
├─ preprocessing.py
├─ retrieval.py
└─ train.py

experiments/
└─ optuna_cv.py

frontend/
├─ src/
├─ package.json
└─ vite.config.js
```

## 실행 환경

- Python 3.11
- macOS 기준 `venv` 사용
- Frontend는 Node.js + Vite 사용
- 의존성은 `requirements.txt`, `frontend/package.json`으로 관리

## 데이터 경로 설정

루트 `.env` 파일에 데이터 경로를 설정합니다.

```env
DATA_DIR=/path/to/team_project_data
TRAIN_CSV_NAME=120000_augmented.csv
```

학습용 CSV는 다음 형식을 지원합니다.

- `label` 컬럼을 포함한 헤더형 CSV
- 또는 첫 번째 열이 label인 header 없는 CSV

## 로컬 실행 방법

### 1. 백엔드 실행

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python3.11 -m uvicorn src.api:app --reload
```

### 2. 프론트엔드 실행

```bash
cd frontend
npm install
npm run dev
```

Frontend 환경변수 예시:

```env
VITE_API_BASE_URL=http://localhost:8000
```

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

- `predicted_class`: 예측 클래스
- `decision_score`: SVM decision function 기반 점수

## Similarity Search 방법

```bash
python -c "from src.retrieval import search_similar_images; print(search_similar_images('/path/to/team_project_data/public_test_dataset/data/00001.png', top_k=5))"
```

반환 예시:

```python
[
    {"index": 1234, "label": 6, "score": 0.9821, "thumbnail_path": "/thumbnail/1234"},
    {"index": 5521, "label": 6, "score": 0.9710, "thumbnail_path": "/thumbnail/5521"},
    {"index": 902, "label": 2, "score": 0.9644, "thumbnail_path": "/thumbnail/902"},
]
```

- `/similar`은 `index`, `label`, `score`, `thumbnail_path`를 반환합니다.
- `/thumbnail/{index}`는 학습 CSV의 row를 `28x28` PNG 이미지로 복원해 반환합니다.

## API 요약

제공 endpoint:

- `POST /predict`
- `POST /similar`
- `GET /thumbnail/{index}`

서버 시작 시 startup 단계에서 추론에 필요한 artifact를 먼저 확인합니다.  
로컬 `artifacts/`에 파일이 없으면 Hugging Face Hub에서 아래 4개 파일을 내려받습니다.

- `pca.pkl`
- `svm.pkl`
- `train_embeddings.npy`
- `train_metadata.csv`

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

## 배포 방법

### Frontend: Vercel

- Root Directory: `frontend`
- Build Command: `npm run build`
- Output Directory: `dist`
- Environment Variable:

```env
VITE_API_BASE_URL=https://bookbo-svm-image-classification-api.hf.space
```

### API: Hugging Face Docker Space

Hugging Face Space에서는 Docker SDK를 사용해 FastAPI 서버를 그대로 실행합니다.

권장 환경변수:

```env
HF_REPO_ID=bookbo/svm_image_classification_artifacts
DATA_DIR=/data/fashion-mnist-120k-augmented
TRAIN_CSV_NAME=120000_augmented.csv
```

artifact repo가 private이면 아래 secret도 필요합니다.

```env
HF_TOKEN=<your-hf-token>
```

추가 메모:

- Docker Space는 루트 `README.md`의 YAML 설정과 `Dockerfile`을 기준으로 빌드됩니다.
- dataset repo를 Space에 mount해서 사용할 경우 mount 경로를 `DATA_DIR`로 지정합니다.
- OpenCV는 서버 환경에 맞게 `opencv-python-headless`를 사용합니다.

## 현재 한계

- similarity search는 brute-force cosine similarity 기반입니다.
- 썸네일 이미지는 학습 CSV를 기반으로 생성하며, 최초 로드 시 메모리에 캐시합니다.
- 배치 추론은 아직 구현하지 않았습니다.
- 대규모 데이터셋 확장 시 retrieval 최적화가 더 필요합니다.
- 프론트엔드와 백엔드가 분리 배포되어 환경변수 설정이 필요합니다.

## 다음 개선 방향

- augmentation 결과 CSV를 학습 입력으로 더 명확하게 연결
- retrieval 속도 개선을 위한 인덱싱 최적화 검토
- public/private dataset 기반 추론 스크립트 보강
- Optuna 실험 결과 정리 및 비교 리포트 추가
- API/프론트 에러 상태와 배포 문서 더 보완

## 정리

이 프로젝트는 딥러닝 모델 없이도 classical ML 기반 이미지 분류와 유사 이미지 검색을 제품 형태로 연결할 수 있음을 보여주는 포트폴리오 프로젝트입니다.

핵심은 다음 두 가지입니다.

- 노트북 실험 코드를 재사용 가능한 학습/추론/서빙 구조로 분리했다는 점
- HOG + PCA + SVM 파이프라인을 프론트엔드 데모와 배포까지 연결했다는 점
