"""D-line pipeline: always-on monitor, smoke preflight, feature registry.

Import contract: diacritizer/scripts/train.py does sys.path-insert on
diacritizer/src, then "from pipeline import ...". Every module here is
import-safe standalone and must never raise into the training loop.
"""
from .monitor import TrainMonitor  # noqa: F401
