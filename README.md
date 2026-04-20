# HOG + PCA + SVM Image Classification Pipeline

## 프로젝트 개요

이 프로젝트는 Fashion-MNIST 스타일 이미지 분류 문제를 대상으로,  
HOG feature extraction, PCA dimensionality reduction, SVM classification을 사용해  
기존 노트북 실험 코드를 재구성한 classical ML 기반 이미지 분류 파이프라인입니다.

목표는 실험용 노트북 코드를 그대로 두는 대신,  
학습과 추론 흐름을 분리하고 재사용 가능한 형태로 정리하는 것이었습니다.

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

## 현재 한계

현재 구현은 최소 동작 파이프라인을 우선 목표로 정리한 상태입니다.

- 데이터 증강(augmentation) 로직은 아직 파이프라인에 포함하지 않음
- 배치 추론은 아직 구현하지 않음
- API 서빙은 아직 구현하지 않음
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
