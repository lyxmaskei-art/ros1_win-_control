from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def find_workspace_code_root(start_dir):
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "core").is_dir() and (candidate / "sim").is_dir():
            return candidate
        nested = candidate / "code"
        if (nested / "core").is_dir() and (nested / "sim").is_dir():
            return nested
    raise RuntimeError(f"Cannot locate workspace code root from {start_dir}")


WORKSPACE_CODE_ROOT = find_workspace_code_root(SCRIPT_DIR)
CORE_DIR = WORKSPACE_CODE_ROOT / "core"
METHODS_DIR = WORKSPACE_CODE_ROOT / "methods"
SIM_DIR = WORKSPACE_CODE_ROOT / "sim"
DOCS_DIR = PROJECT_ROOT / "docs"
RESULTS_DIR = PROJECT_ROOT / "results"


for candidate in (CORE_DIR, METHODS_DIR, SIM_DIR):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
