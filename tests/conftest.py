# tests/conftest.py

import sys
from pathlib import Path

# 1) Identify the project root (the folder that contains `src/` and `tests/`)
project_root = Path(__file__).parent.parent

# 2) Insert it at the front of sys.path
sys.path.insert(0, str(project_root))
