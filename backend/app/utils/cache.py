import os
import pickle
import tempfile
from pathlib import Path


def load_cache(cache_path: str | Path, enabled: bool = True):
    path = Path(cache_path)
    if not enabled:
        print(f"Cache read disabled: {path.resolve()}")
        return None
    if not path.exists():
        print(f"Cache miss: {path.resolve()}")
        return None

    print(f"Cache hit: {path.resolve()}")
    with path.open("rb") as file:
        return pickle.load(file)


def save_cache(cache_path: str | Path, value):
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            pickle.dump(value, file)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        print(f"Cache save failed; existing cache preserved: {path.resolve()}")
        raise

    print(f"Cache saved: {path.resolve()}")
