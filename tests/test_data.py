from src.data import provision_data
from pathlib import Path


def test_provision_data():
    c, l, _, _, _, _ = provision_data(Path("tests/mock_data"), 0, 0)
