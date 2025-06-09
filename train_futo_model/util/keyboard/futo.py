import numpy as np


_KEY_WIDTH = 1.0 / 10.0
_KEY_HEIGHT = 1.0 / 3.0

_FIRST_ROW = ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"]
_FIRST_ROW_X_OFFSET = (10 - len(_FIRST_ROW)) * _KEY_WIDTH / 2
_SECOND_ROW = ["A", "S", "D", "F", "G", "H", "J", "K", "L"]
_SECOND_ROW_X_OFFSET = (10 - len(_SECOND_ROW)) * _KEY_WIDTH / 2
_THIRD_ROW = ["Z", "X", "C", "V", "B", "N", "M"]
_THIRD_ROW_X_OFFSET = (10 - len(_THIRD_ROW)) * _KEY_WIDTH / 2

_KEY_TO_CENTER = {}
for i, key in enumerate(_FIRST_ROW):
    _KEY_TO_CENTER[key] = np.array(
        (_KEY_WIDTH * (i + 0.5 + _FIRST_ROW_X_OFFSET), _KEY_HEIGHT * 0.5),
        dtype=np.float32,
    )
for i, key in enumerate(_SECOND_ROW):
    _KEY_TO_CENTER[key] = np.array(
        (_KEY_WIDTH * (i + 0.5 + _SECOND_ROW_X_OFFSET), _KEY_HEIGHT * 1.5),
        dtype=np.float32,
    )
for i, key in enumerate(_THIRD_ROW):
    _KEY_TO_CENTER[key] = np.array(
        (_KEY_WIDTH * (i + 0.5 + _THIRD_ROW_X_OFFSET), _KEY_HEIGHT * 2.5),
        dtype=np.float32,
    )

CENTER_TO_KEY = {tuple(v): k for k, v in _KEY_TO_CENTER.items()}


def get_key_by_position(position: np.ndarray) -> str:
    closest_center = min(
        _KEY_TO_CENTER.values(),
        key=lambda center: np.linalg.norm(center - position),
    )
    return CENTER_TO_KEY[tuple(closest_center)]
