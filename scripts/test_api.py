from __future__ import annotations

import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings

BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 30


def get_sample_image_path() -> Path:
    """public_test_dataset에서 테스트용 이미지 1장을 찾는다."""
    settings = load_settings()
    data_dir = settings["data_dir"]
    image_dir = data_dir / "public_test_dataset" / "data"
    image_paths = sorted(image_dir.glob("*.png"))

    if not image_paths:
        raise FileNotFoundError(f"테스트 이미지를 찾을 수 없습니다: {image_dir}")

    return image_paths[0]


def print_error(response: requests.Response) -> None:
    """요청 실패 시 상태코드와 응답 메시지를 출력한다."""
    print(f"요청 실패: {response.status_code}")
    print(response.text)


def test_predict(image_path: Path) -> None:
    """predict endpoint를 호출하고 결과를 출력한다."""
    with image_path.open("rb") as image_file:
        response = requests.post(
            f"{BASE_URL}/predict",
            files={"file": (image_path.name, image_file, "image/png")},
            timeout=REQUEST_TIMEOUT,
        )

    print("[/predict]")
    if not response.ok:
        print_error(response)
        return

    result = response.json()
    print(f"predicted_class: {result['predicted_class']}")
    print(f"decision_score: {result['decision_score']}")


def test_similar(image_path: Path, top_k: int = 5) -> None:
    """similar endpoint를 호출하고 상위 결과를 출력한다."""
    with image_path.open("rb") as image_file:
        response = requests.post(
            f"{BASE_URL}/similar",
            files={"file": (image_path.name, image_file, "image/png")},
            data={"top_k": str(top_k)},
            timeout=REQUEST_TIMEOUT,
        )

    print("[/similar]")
    if not response.ok:
        print_error(response)
        return

    results = response.json()
    for item in results[:top_k]:
        print(f"label: {item['label']}, score: {item['score']}")


def main() -> None:
    """로컬 FastAPI 서버의 predict, similar endpoint를 테스트한다."""
    try:
        image_path = get_sample_image_path()
        print(f"test_image: {image_path}")
        test_predict(image_path)
        test_similar(image_path, top_k=5)
    except requests.RequestException as error:
        print("요청 중 예외가 발생했습니다.")
        print(str(error))
    except Exception as error:
        print("테스트 실행 중 예외가 발생했습니다.")
        print(str(error))


if __name__ == "__main__":
    main()
