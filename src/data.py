import logging
import os
from pprint import pprint
import time
from glob import glob
from pathlib import Path
from typing import Iterable
from torch.utils.data.dataset import IterableDataset
import tarfile
import imageio
import pandas as pd

import requests

CHUNK_SIZE = 8 * (1024**2)

# URLs for the zip files
METADATA_FILENAME = "Data_Entry_2017_v2020.csv"
IMAGE_URLS = [
    "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
    # "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
    # "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
    # "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
    # "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
    # "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
    # "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
    # "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
    # "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
    # "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
    # "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
    # "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
]


class LungDataset(IterableDataset):
    def __init__(self, directory: Path):
        super(LungDataset).__init__()
        self._directory = directory
        df = pd.read_csv(directory / METADATA_FILENAME)
        self._labels = set()
        for labels in df["Finding Labels"]:
            self._labels |= set(labels.split("|"))
        self._metadata = {
            row["Image Index"]: row["Finding Labels"] for _, row in df.iterrows()
        }

    def __iter__(self) -> Iterable:
        image_dir = self._directory / "images"
        for im_name in os.listdir(image_dir):
            im = imageio.imread(image_dir / im_name)
            for label in self._metadata[im_name].split("|"):
                yield im, label


def download(url: str, dest: Path, *, verbose=False):
    with requests.get(url, stream=True) as r:
        size = int(r.headers["Content-length"])
        r.raise_for_status()
        with open(dest, "wb") as f:
            counter = 1
            T = time.time()
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                dT = time.time() - T
                T = time.time()
                f.write(chunk)
                logging.info(
                    f"{dest.name}: {counter * CHUNK_SIZE/size:7.2%}   {8/dT:.2}MB/s"
                )
                counter += 1


def download_if_needed(dest: Path, *, clean: bool = False):
    tar_dest = dest / "tars"
    os.makedirs(dest, exist_ok=True)
    os.makedirs(tar_dest, exist_ok=True)
    if clean:
        for filename in glob(str(dest) + "/*.tar.gz"):
            os.unlink(filename)
    for idx, url in enumerate(IMAGE_URLS):
        tar_path = tar_dest / f"images_{idx + 1:02d}.tar.gz"
        if tar_path.name not in os.listdir(tar_dest):
            download(url, tar_path)
        with tarfile.open(tar_path) as tar:
            tar.extractall(dest)


def get(dest: Path) -> LungDataset:
    download_if_needed(dest)
    return LungDataset(dest)


if __name__ == "__main__":
    download_if_needed(Path("data"))
    for im, label in LungDataset(Path("data")):
        pprint(im)
        pprint(label)
        break
