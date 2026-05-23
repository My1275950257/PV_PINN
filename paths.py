from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


def figure_path(filename):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURE_DIR / filename


def table_path(filename):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    return TABLE_DIR / filename
