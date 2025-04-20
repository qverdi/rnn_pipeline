from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


SEARCH_SPACE_FILE = PROJECT_ROOT / "input" / "params" / "model_params.json"
EXPERIMENT_PARAMS = PROJECT_ROOT / "input" / "params" / "experiment_params.json"
SEARCH_OPTIMIZER_PARAMS = PROJECT_ROOT / "input" / "params" / "search_params.json"
MODEL_DESIGN = PROJECT_ROOT / "input" / "params" / "model_design.json"
DATA_DIR = PROJECT_ROOT / "input" / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_OPTIMIZER_PARAMS = PROJECT_ROOT / "input" / "params" / "model_optimizer_params.json"
README_FILE = PROJECT_ROOT / "README.md"