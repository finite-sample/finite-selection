import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

R_DIR = Path(__file__).parent / "fixtures" / "r"


def load(name: str):
    with (R_DIR / f"{name}.json").open() as fh:
        return json.load(fh)


def r_matrix(rows) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in row] for row in rows], float)


def r_data(name: str) -> pd.DataFrame:
    d = load(f"data_{name}")
    return pd.DataFrame(d["rows"], columns=d["columns"]).drop(columns="Index")


def fixture_names(prefix: str) -> list[str]:
    return sorted(p.stem for p in R_DIR.glob(f"{prefix}*.json"))


@pytest.fixture
def rng():
    return np.random.default_rng(20261013)
