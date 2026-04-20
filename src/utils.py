"""
Utilities: logging, checkpointing, retry logic, robust I/O
"""

import json
import logging
import pickle
import time
import traceback
from functools import wraps
from pathlib import Path

import pandas as pd

from config import LOGS_DIR, CHECKPOINTS_DIR


# ─── Logger ───────────────────────────────────────────────────────────────────
def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """Set up a logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    if log_file:
        fh = logging.FileHandler(LOGS_DIR / log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    else:
        fh = logging.FileHandler(LOGS_DIR / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─── Checkpoint ───────────────────────────────────────────────────────────────
class Checkpoint:
    """Persistent checkpoint manager for pipeline stages."""

    def __init__(self, name: str):
        self.name = name
        self.path = CHECKPOINTS_DIR / f"{name}.pkl"
        self.meta_path = CHECKPOINTS_DIR / f"{name}_meta.json"
        self.logger = setup_logger(f"checkpoint.{name}")

    def save(self, data, meta: dict = None):
        """Save data and optional metadata."""
        with open(self.path, "wb") as f:
            pickle.dump(data, f)
        meta = meta or {}
        meta["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        meta["name"] = self.name
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        self.logger.info(f"Checkpoint saved: {self.path}")

    def load(self):
        """Load data from checkpoint."""
        if not self.path.exists():
            return None
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        self.logger.info(f"Checkpoint loaded: {self.path}")
        return data

    def exists(self) -> bool:
        return self.path.exists()

    def meta(self) -> dict:
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                return json.load(f)
        return {}

    def delete(self):
        self.path.unlink(missing_ok=True)
        self.meta_path.unlink(missing_ok=True)


# ─── Retry Decorator ──────────────────────────────────────────────────────────
def retry(max_attempts: int = 5, delay: float = 3.0, backoff: float = 2.0,
          exceptions=(Exception,), logger=None):
    """Exponential-backoff retry decorator."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        msg = f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        if logger:
                            logger.error(msg)
                        raise
                    msg = f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. Retrying in {_delay:.1f}s..."
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    time.sleep(_delay)
                    _delay *= backoff
        return wrapper
    return decorator


# ─── Safe CSV I/O ─────────────────────────────────────────────────────────────
def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with error handling."""
    try:
        return pd.read_csv(path, **kwargs)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")


def safe_save_csv(df: pd.DataFrame, path: Path, **kwargs):
    """Save DataFrame to CSV with error handling."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)


# ─── Stage Runner ─────────────────────────────────────────────────────────────
def run_stage(stage_name: str, func, *args, force_rerun: bool = False, **kwargs):
    """
    Run a pipeline stage with checkpoint support.
    If checkpoint exists and force_rerun=False, loads from checkpoint.
    """
    logger = setup_logger(f"stage.{stage_name}")
    ckpt = Checkpoint(stage_name)

    if ckpt.exists() and not force_rerun:
        logger.info(f"[{stage_name}] Loading from checkpoint (saved: {ckpt.meta().get('timestamp', 'unknown')})")
        return ckpt.load()

    logger.info(f"[{stage_name}] Running stage...")
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        ckpt.save(result, meta={"elapsed_seconds": elapsed})
        logger.info(f"[{stage_name}] Completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        logger.error(f"[{stage_name}] FAILED: {e}\n{traceback.format_exc()}")
        raise


# ─── Progress Helper ──────────────────────────────────────────────────────────
def chunked(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
