import json
from pathlib import Path
from typing import Iterable, List, Tuple, Union
from sklearn.model_selection import train_test_split
import librosa
import re
import numpy as np


def provision_data(
    source: Path, val_part: float = 0.1, test_part: float = 0.2
) -> Tuple[np.ndarray, List[str], np.ndarray, List[str], np.ndarray, List[str]]:
    validate_x, validate_y, test_x, test_y = np.empty(1), [], np.empty(1), []
    with open(source / "labels.json") as f:
        train_y = json.load(f)["labels"]
        file_names = sorted([audio for audio in source.resolve().glob(r"audio_*.ogg")])
        train_x = list(map(lambda x: librosa.load(x)[0], file_names))
    if test_part:
        train_x, train_y, test_x, test_y = train_test_split(
            train_x, train_y, test_size=test_part
        )
    if val_part:
        train_x, train_y, validate_x, validate_y = train_test_split(
            train_x, train_y, test_size=val_part / (1 - test_part)
        )
    return train_x, train_y, validate_x, validate_y, test_x, test_y
