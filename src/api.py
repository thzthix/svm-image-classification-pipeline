from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .inference import predict_image
from .model_io import ensure_all_artifacts_available
from .retrieval import search_similar_images

app = FastAPI(title="HOG PCA SVM API")


async def save_upload_to_temp(upload_file: UploadFile) -> Path:
    """업로드 이미지를 임시 파일로 저장하고 경로를 반환한다."""
    suffix = Path(upload_file.filename or "upload.png").suffix or ".png"

    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        content = await upload_file.read()
        if not content:
            raise HTTPException(status_code=400, detail="빈 파일은 업로드할 수 없습니다.")
        temp_file.write(content)
        return Path(temp_file.name)


def remove_temp_file(file_path: Path) -> None:
    """처리가 끝난 임시 파일을 삭제한다."""
    if file_path.exists():
        file_path.unlink()


@app.on_event("startup")
def prepare_artifacts() -> None:
    """서버 시작 시 추론에 필요한 모델 아티팩트를 준비한다."""
    ensure_all_artifacts_available()


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, int | float]:
    """업로드된 이미지 1장에 대해 클래스를 예측한다."""
    temp_path = await save_upload_to_temp(file)

    try:
        return predict_image(temp_path)
    except FileNotFoundError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        remove_temp_file(temp_path)
        await file.close()


@app.post("/similar")
async def similar(
    file: UploadFile = File(...),
    top_k: int = Form(5),
) -> list[dict[str, int | float]]:
    """업로드된 이미지와 유사한 train 샘플 top-k를 반환한다."""
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k는 1 이상이어야 합니다.")

    temp_path = await save_upload_to_temp(file)

    try:
        return search_similar_images(temp_path, top_k=top_k)
    except FileNotFoundError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        remove_temp_file(temp_path)
        await file.close()
