"""Deterministic fingerprints for loaded data and local files."""

from hashlib import sha256
from pathlib import Path

import pandas as pd


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Hash schema, index, and values without serializing secrets."""
    digest = sha256()
    digest.update(repr(tuple(frame.columns)).encode("utf-8"))
    digest.update(repr(tuple(str(dtype) for dtype in frame.dtypes)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def file_fingerprint(path: Path) -> str:
    """Hash file bytes in bounded chunks."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
