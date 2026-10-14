"""The code in the user guide runs and prints what the guide says it prints."""

import contextlib
import io
import re
from pathlib import Path

GUIDE = Path(__file__).parents[1] / "docs" / "guide.md"


def test_guide_outputs_match():
    text = GUIDE.read_text()
    blocks = re.findall(r"```python\n(.*?)```\n\n```text\n(.*?)```", text, re.DOTALL)
    assert len(blocks) == 6
    namespace: dict = {}
    for code, expected in blocks:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(code, namespace)  # noqa: S102 - running our own documentation
        assert out.getvalue().strip() == expected.strip()
