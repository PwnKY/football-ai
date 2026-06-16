from pathlib import Path
import json
import pickle


# Project root is the folder above src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_directories():
    """Create important folders if they do not already exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_pickle(obj, path):
    """Save a Python object, such as a trained model, to disk."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """Load a Python object saved with save_pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path):
    """Save a small JSON file with readable formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
