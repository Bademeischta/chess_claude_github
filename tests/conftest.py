"""Common pytest setup — ensures the project root is on sys.path so that
`import engine`, `import config`, etc. work when pytest is invoked from
anywhere (e.g. `pytest tests/`)."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
