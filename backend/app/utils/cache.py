import pickle
from pathlib import Path


def load_cache(cache_path: str | Path, enabled: bool = True):
    path = Path(cache_path)
    if not enabled or not path.exists():
        return None

    with path.open("rb") as file:
        return pickle.load(file)


def save_cache(cache_path: str | Path, value):
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as file:
        pickle.dump(value, file)

