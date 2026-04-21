from __future__ import annotations

from enum import IntEnum


class FashionClass(IntEnum):
    """Fashion-MNIST 클래스 인덱스를 표현한다."""

    T_SHIRT_TOP = 0
    TROUSER = 1
    PULLOVER = 2
    DRESS = 3
    COAT = 4
    SANDAL = 5
    SHIRT = 6
    SNEAKER = 7
    BAG = 8
    ANKLE_BOOT = 9


CLASS_DISPLAY_NAMES = {
    FashionClass.T_SHIRT_TOP: "T-shirt/top",
    FashionClass.TROUSER: "Trouser",
    FashionClass.PULLOVER: "Pullover",
    FashionClass.DRESS: "Dress",
    FashionClass.COAT: "Coat",
    FashionClass.SANDAL: "Sandal",
    FashionClass.SHIRT: "Shirt",
    FashionClass.SNEAKER: "Sneaker",
    FashionClass.BAG: "Bag",
    FashionClass.ANKLE_BOOT: "Ankle boot",
}


def get_label_name(label: int) -> str:
    """숫자 label을 사용자 표시용 클래스 이름으로 변환한다."""
    try:
        return CLASS_DISPLAY_NAMES[FashionClass(label)]
    except ValueError:
        return str(label)
