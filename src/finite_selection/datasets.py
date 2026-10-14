"""Bundled example data."""

from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

_LALONDE_COLUMNS = (
    "Age",
    "Education",
    "Black",
    "Hispanic",
    "Married",
    "Nodegree",
    "Re74",
    "Re75",
)


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as err:  # pragma: no cover - exercised without the extra
        raise ImportError(
            'the bundled data needs pyarrow: pip install "finite-selection[data]"'
        ) from err
    return pa, pq


def lalonde_schema() -> pa.Schema:
    """Return the declared Arrow schema of the bundled Lalonde table."""
    pa, _ = _require_pyarrow()
    return pa.schema([(name, pa.float64()) for name in _LALONDE_COLUMNS])


def lalonde() -> pd.DataFrame:
    """Dehejia and Wahba's subset of the National Supported Work data.

    445 men (185 trained, 260 controls) with eight baseline covariates: age,
    years of education, indicators for Black, Hispanic, married and no high
    school degree, and earnings in 1974 and 1975. Rebuilt from
    https://users.nber.org/~rdehejia/data/nsw_dw.dta by
    ``scripts/build_lalonde.py``; rows follow the order of the R package FSM.
    Needs the ``data`` extra.

    Returns:
        One row per unit, one float column per covariate.

    Raises:
        ValueError: If the packaged file does not match its declared schema.
    """
    _, pq = _require_pyarrow()
    with files("finite_selection").joinpath("data/lalonde.parquet").open("rb") as fh:
        table = pq.read_table(fh)
    if not table.schema.equals(lalonde_schema()):
        raise ValueError(f"lalonde.parquet has schema {table.schema}")
    return table.to_pandas()
