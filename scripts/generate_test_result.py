from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_settings
from src.inference import predict_image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def generate_test_result(image_dir: str | Path, output_path: str | Path) -> Path:
    """디렉터리 내 모든 이미지를 예측해 testResult.txt 형식으로 저장한다."""
    image_dir = Path(image_dir)
    output_path = Path(output_path)

    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=lambda path: path.stem,
    )

    if not image_paths:
        raise ValueError(f"예측할 이미지가 없습니다: {image_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for image_path in image_paths:
            prediction = predict_image(image_path)
            predicted_class = int(prediction["predicted_class"])
            file.write(f"{image_path.stem} {predicted_class}\n")

    return output_path


def main() -> None:
    """명령줄에서 testResult.txt 생성 스크립트를 실행한다."""
    settings = load_settings()
    default_image_dir = settings["data_dir"] / "public_test_dataset" / "data"
    default_output_path = settings["data_dir"] / "public_test_dataset" / "testResult.txt"

    parser = argparse.ArgumentParser(description="디렉터리 이미지 예측 결과를 저장합니다.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=default_image_dir,
        help="예측할 이미지 디렉터리 경로",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=default_output_path,
        help="생성할 testResult.txt 경로",
    )
    args = parser.parse_args()

    result_path = generate_test_result(args.image_dir, args.output_path)
    print(result_path)


if __name__ == "__main__":
    main()
