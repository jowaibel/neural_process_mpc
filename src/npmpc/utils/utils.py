import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def get_project_root() -> Path:
    # this file lives at <root>/src/npmpc/utils/utils.py
    return Path(__file__).resolve().parents[3]


_FOLDER = None
_LOG_ENV = 'NP_MPC_LOG_FILE'
_LEVEL_ENV = 'NP_MPC_LOG_LEVEL'


def set_folder(folder):
    global _FOLDER
    _FOLDER = Path(folder) if folder else None


def get_folder():
    return _FOLDER


def resolve_path(value):
    if isinstance(value, str) and value and _FOLDER is not None and '/' not in value:
        return str(_FOLDER / value)
    return value


def setup_logging(debug=False):
    base = _FOLDER if _FOLDER is not None else get_project_root()
    log_dir = base / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.environ[_LOG_ENV] = str(log_file)
    os.environ[_LEVEL_ENV] = 'DEBUG' if debug else 'INFO'
    _configure_root(str(log_file), debug)
    return str(log_file)


def attach_logging():
    path = os.environ.get(_LOG_ENV)
    if path:
        _configure_root(path, os.environ.get(_LEVEL_ENV, 'INFO') == 'DEBUG')


def _configure_root(path, debug):
    root = logging.getLogger()
    if any(getattr(h, '_np_mpc_marker', False) for h in root.handlers):
        return
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sh = logging.StreamHandler(sys.stderr); sh.setFormatter(fmt); sh._np_mpc_marker = True
    fh = logging.FileHandler(path, mode='a'); fh.setFormatter(fmt); fh._np_mpc_marker = True
    root.handlers.clear()
    root.addHandler(sh)
    root.addHandler(fh)
    root.setLevel(logging.DEBUG if debug else logging.INFO)
