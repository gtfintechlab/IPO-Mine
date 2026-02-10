# src/utils/config.py
"""
Package-friendly, notebook-safe path configuration.

Priority:
  1) Explicit override set via set_data_root(path) (persisted to a user config file)
  2) Previously saved config in user config dir
  3) Repo-style root near this module (prefers dir that has BOTH 'src/' and 'data/' siblings)
  4) Per-user data dir (cross-platform; uses platformdirs if available, else fallback)

Exposes:
  BASE_DIR, DATA_DIR, RAW_DIR, OUTPUT_DIR, PARSED_DIR, MAPPINGS_FILE
  set_data_root(path), resolve_from_base(*parts), print_config()
"""

from __future__ import annotations
from pathlib import Path
import json
from typing import Optional, Iterable

PROJECT_NAME = "S1"  # used for per-user dirs and default fallback naming

# ---- cross-platform user dirs (no hard dependency on platformdirs) ----
try:
    from platformdirs import user_config_dir, user_data_dir  # type: ignore

    def _user_config_dir() -> Path: return Path(user_config_dir(PROJECT_NAME))
    def _user_data_dir()  -> Path: return Path(user_data_dir(PROJECT_NAME))
except Exception:
    # Minimal fallbacks if platformdirs isn't installed
    def _user_config_dir() -> Path: return Path.home() / ".config" / PROJECT_NAME
    def _user_data_dir()  -> Path: return Path.home() / f".local/share/{PROJECT_NAME}"

# Where we persist the chosen data root (NOT SURE THIS EXISTS ANYMORE)
CONFIG_FILE = _user_config_dir() / "config.json"

# ---- helpers to read/write small user config ----
def _read_saved_root() -> Optional[Path]:
    try:
        if CONFIG_FILE.exists():
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            p = Path(data.get("data_root", "")).expanduser().resolve()
            return p if p.exists() else None
    except Exception:
        pass
    return None

def _write_saved_root(path: Path) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"data_root": str(path)}
    CONFIG_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

# ---- heuristics to detect a repo-style base near this module or cwd ----
# We intentionally DO NOT anchor on a 'data' folder that sits *inside* src/.
# Instead, prefer a directory that has BOTH 'src/' and 'data/' as children.
ANCHORS: tuple[str, ...] = ("src", "data", "pyproject.toml", "README.md")

def _score_repo_root(candidate: Path) -> int:
    """
    Higher score == more likely to be the real project root.
    - Prefer dirs that have both 'src' and 'data' (siblings) (+3)
    - Bonus if they also have README or pyproject (+1 each)
    - Avoid picking a nested src/ as root (return 0 if name == 'src')
    """
    if candidate.name == "src":
        return 0
    score = 0
    has_src = (candidate / "src").exists()
    has_data = (candidate / "data").exists()
    if has_src and has_data:
        score += 3
    elif has_data:
        score += 1  # weak signal: may be a subdir with stray data/
    if (candidate / "README.md").exists():
        score += 1
    if (candidate / "pyproject.toml").exists():
        score += 1
    return score

def _best_ancestor(start: Path) -> Optional[Path]:
    """
    Walk up from `start`, compute a score for each ancestor, and return the best.
    Prefer higher (closer to FS root) on ties so we escape nested src/ trees.
    """
    cur = start if start.is_dir() else start.parent
    cur = cur.resolve()
    best: tuple[int, Optional[Path]] = (-1, None)
    for anc in [cur, *cur.parents]:
        sc = _score_repo_root(anc)
        if sc > best[0]:
            best = (sc, anc)
    return best[1] if best[0] > 0 else None

def _repo_like_root() -> Optional[Path]:
    # Prefer a root near this module
    here = Path(__file__).resolve()
    chosen = _best_ancestor(here)
    if chosen:
        return chosen
    # Fallback: try from CWD (useful when running ad-hoc scripts)
    return _best_ancestor(Path.cwd())

# ---- decide BASE_DIR once, with a simple singleton Config ----
class _Config:
    def __init__(self) -> None:
        self._base_dir = self._decide_base_dir()
        self._ensure_core_dirs()

    def _decide_base_dir(self) -> Path:
        # 1) Saved user choice
        saved = _read_saved_root()
        if saved:
            return saved
        # 2) Repo-like root near code (or cwd), with strong preference for sibling src/data
        repo = _repo_like_root()
        if repo:
            return repo
        # 3) Per-user data root
        user_root = _user_data_dir()
        user_root.mkdir(parents=True, exist_ok=True)
        return user_root

    @property
    def BASE_DIR(self) -> Path: return self._base_dir
    @property
    def DATA_DIR(self) -> Path: return (self.BASE_DIR / "data").resolve()
    @property
    def RAW_DIR(self) -> Path: return (self.DATA_DIR / "raw").resolve()
    @property
    def OUTPUT_DIR(self) -> Path: return (self.DATA_DIR / "output").resolve()
    @property
    def PARSED_DIR(self) -> Path: return (self.OUTPUT_DIR / "parsed").resolve()

    def _ensure_core_dirs(self) -> None:
        for d in (self.DATA_DIR, self.RAW_DIR, self.OUTPUT_DIR, self.PARSED_DIR):
            d.mkdir(parents=True, exist_ok=True)

    # Public utilities
    def set_data_root(self, path: Path | str) -> None:
        """
        Persistently change the data root (i.e., BASE_DIR).
        Creates necessary subdirectories under <path>/data/{raw,output/parsed}.
        """
        p = Path(path).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        self._base_dir = p
        self._ensure_core_dirs()
        _write_saved_root(p)

    def resolve_from_base(self, *parts: str | Path) -> Path:
        return (self.BASE_DIR.joinpath(*parts)).resolve()

    def print_config(self) -> None:
        print("S1 path configuration")
        print(f"  BASE_DIR   = {self.BASE_DIR}")
        print(f"  DATA_DIR   = {self.DATA_DIR}")
        print(f"  RAW_DIR    = {self.RAW_DIR}")
        print(f"  OUTPUT_DIR = {self.OUTPUT_DIR}")
        print(f"  PARSED_DIR = {self.PARSED_DIR}")
        print(f"  CONFIG_FILE= {CONFIG_FILE}")

# Create a single shared config instance
_cfg = _Config()

# Module-level exports (easy to import in notebooks/code)
BASE_DIR   = _cfg.BASE_DIR
DATA_DIR   = _cfg.DATA_DIR
RAW_DIR    = _cfg.RAW_DIR
OUTPUT_DIR = _cfg.OUTPUT_DIR
PARSED_DIR = _cfg.PARSED_DIR

# Added: single canonical location for the mappings JSON inside data/
MAPPINGS_FILE = DATA_DIR / "global_section_names.json"

def set_data_root(path: Path | str) -> None: _cfg.set_data_root(path)
def resolve_from_base(*parts: str | Path) -> Path: return _cfg.resolve_from_base(*parts)
def print_config() -> None: _cfg.print_config()

__all__ = [
    "PROJECT_NAME",
    "BASE_DIR",
    "DATA_DIR",
    "RAW_DIR",
    "OUTPUT_DIR",
    "PARSED_DIR",
    "MAPPINGS_FILE",
    "set_data_root",
    "resolve_from_base",
    "print_config",
]

