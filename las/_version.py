from __future__ import annotations

import subprocess
from pathlib import Path


def _get_version() -> str:
    result = subprocess.run(
        ["python", "-m", "setuptools_scm", "--root", str(Path(__file__).parent.parent)],
        check=True,
        capture_output=True,
        text=True,
    )
    version = result.stdout.strip()
    version = version.replace(".dev", "-dev")
    return version


__version__ = _get_version()
