from pathlib import Path
from typing import Iterable, List, Tuple
import tensorflow as tf

import data


def train_models():
    for model in models:
        model.fit()


def save_models():
    pass


def main():
    data.provision_waveform()
    models: List[tf.keras.Model] = []
    train_models(models)
    save_models(models)


if __name__ == "__main__":
    main()
