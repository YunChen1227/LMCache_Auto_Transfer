# SPDX-License-Identifier: Apache-2.0
"""NAS KV cache journal + startup warmup plugin for the LMCache MP server.

Load with::

    lmcache server ... --server-module '{"module_path": "nas_kv_warmup",
                                         "config": {...}}'
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.ext_server_module import (
        ExtServerModuleBuildContext,
        ExtServerModuleComponents,
    )


def build_server_modules(
    ctx: "ExtServerModuleBuildContext",
) -> "ExtServerModuleComponents":
    """Server-module factory; imports LMCache lazily so the pure submodules
    (journal, recorder, replayer) stay importable without it."""
    # First Party
    from nas_kv_warmup.module import build_server_modules as _build

    return _build(ctx)
