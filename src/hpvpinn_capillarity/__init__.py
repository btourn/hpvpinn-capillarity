"""hp-VPINN framework for inverse hydraulic parameter recovery in unsaturated porous media.

Importing this package enables 64-bit precision in JAX, the regime of every run
reported in the manuscript. The flag is set through the environment variable
that JAX reads when it is imported, so that modules that do not need JAX (the
aggregation of saved runs, for example) stay free of it; if JAX was imported
before, the flag is set directly. It must be set before any array is created.
"""

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
if "jax" in sys.modules:
    sys.modules["jax"].config.update("jax_enable_x64", True)

__version__ = "2.0.0"
