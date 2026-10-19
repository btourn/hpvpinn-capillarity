"""Locations of the inputs and outputs of the scripts of the package.

Layout of a clone (the package is installed in editable mode by ``uv sync``)::

    <root>/data/                      input data files (data_<case>.dat)
    <root>/results/hp_vpinn/          results_<study>/, final_runs/, Figures/,
                                      sensitivity_*/, quadrature_check_*/
    <root>/results/multistart/        multi-start summaries
    <root>/results/conventional/      conventional-inversion outputs

``<root>`` is the clone; it can be overridden with the environment variable
``HPVPINN_ROOT``. The results directory can be moved with ``HPVPINN_RESULTS``
and the data directory with ``HPVPINN_DATA``.
"""

import importlib.util
import os
from pathlib import Path

MODEL = "hp_vpinn"      # name of the model directory under the results root


def repo_root() -> Path:
    env = os.environ.get("HPVPINN_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    env = os.environ.get("HPVPINN_DATA")
    return Path(env).resolve() if env else repo_root() / "data"


def data_file(name: str) -> Path:
    return data_dir() / name


def results_root() -> Path:
    env = os.environ.get("HPVPINN_RESULTS")
    return Path(env).resolve() if env else repo_root() / "results"


def model_dir(model: str = MODEL) -> Path:
    """Directory of the results of one model (``<results>/<model>``)."""
    return results_root() / model


def multistart_dir() -> Path:
    return results_root() / "multistart"


def conventional_dir() -> Path:
    return results_root() / "conventional"


def module_source(module: str) -> Path:
    """Path of the source file of a module, without importing it.

    Used to read constants from a module with ``ast`` when JAX must not be
    imported (aggregation of saved runs).
    """
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise ImportError(f"Module {module!r} not found.")
    return Path(spec.origin)
