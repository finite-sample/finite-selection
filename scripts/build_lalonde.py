"""Rebuild the bundled Lalonde data from the public NBER source file.

Source: Dehejia and Wahba's subset of the National Supported Work
demonstration, https://users.nber.org/~rdehejia/data/nsw_dw.dta (445 units:
185 treated, 260 controls). Rows are stably sorted by age with treated units
first within each age, which reproduces the row order of the R package FSM.

Usage: uv run python scripts/build_lalonde.py path/to/nsw_dw.dta
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from evenhand.datasets import lalonde_schema

COLUMNS = {
    "age": "Age",
    "education": "Education",
    "black": "Black",
    "hispanic": "Hispanic",
    "married": "Married",
    "nodegree": "Nodegree",
    "re74": "Re74",
    "re75": "Re75",
}

raw = pd.read_stata(sys.argv[1])
out = (
    raw.sort_values("age", kind="stable")
    .rename(columns=COLUMNS)[list(COLUMNS.values())]
    .astype("float64")
    .reset_index(drop=True)
)
table = pa.Table.from_pandas(out, schema=lalonde_schema(), preserve_index=False)
dest = Path(__file__).parents[1] / "src" / "evenhand" / "data" / "lalonde.parquet"
pq.write_table(table, dest)
sys.stderr.write(f"wrote {table.num_rows} rows to {dest}\n")
