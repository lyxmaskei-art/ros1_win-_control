from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CORE_DIR = PROJECT_ROOT / 'core'
SIM_DIR = PROJECT_ROOT / 'sim'
DOCS_DIR = PROJECT_ROOT / 'docs'
RESULTS_DIR = PROJECT_ROOT / 'results'


for candidate in (CORE_DIR, SIM_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
