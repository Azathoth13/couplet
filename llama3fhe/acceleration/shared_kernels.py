"""Install the packaged EasyFHE key-switch, arithmetic and EvalMod extensions.

Extensions replace operations in ``easyfhe.fhe.ops`` and update any corresponding
functions re-exported by ``easyfhe.fhe``. The context, ciphertexts and execution
schedule are unchanged. Batched BSGS prepares source variants for up to five
key-switch digits before loading the CUDA library.

``COUPLET_KERNEL_ROOT`` can select an alternative extension source directory;
Python modules are always loaded from the application package. Host caches are
selected by ``ODIN_SYLPH_HOSTCACHE`` or the explicit ``hostcache`` argument.
"""
from __future__ import annotations

import os
from pathlib import Path


def install(*, hostcache: bool | None = None) -> dict:
    root = os.environ.get("COUPLET_KERNEL_ROOT", str(Path(__file__).with_name("fhe_ext")))
    from easyfhe import fhe
    from easyfhe.fhe.ops import alignment, arithmetic, kernels, rotation
    from llama3fhe.acceleration.fhe_ext import evalmod, keyswitch

    kernel_dir = None
    if os.environ.get("ODIN_BATCH_BS", "0") == "1" or os.environ.get("ODIN_WIDE_BSGS", "0") == "1":
        # Raised-modulus batches can need five key-switch digits.
        from . import kernel_sources
        kernel_dir = kernel_sources.prepare(root)
    elif os.environ.get("COUPLET_KERNEL_ROOT"):
        from .fhe_ext import cudalib
        kernel_dir = str(Path(root) / "kernels")
        cudalib._DIR = kernel_dir

    before = {m: dict(vars(m)) for m in (arithmetic, rotation, alignment, kernels)}
    if hostcache is None:
        hostcache = os.environ.get("ODIN_SYLPH_HOSTCACHE", "0") == "1"
    if hostcache:
        from llama3fhe.acceleration.fhe_ext import hostcache as hc
        hc.install()
    keyswitch.install()
    evalmod.install()

    changed = {}
    for m, old in before.items():
        for name, cur in vars(m).items():
            if callable(cur) and old.get(name) is not cur and name in old:
                changed[f"{m.__name__.rsplit('.', 1)[-1]}.{name}"] = True
                # the fhe namespace re-exports the ops functions by value; follow the patch
                if getattr(fhe, name, None) is old[name]:
                    setattr(fhe, name, cur)
                    changed[f"fhe.{name}"] = True
    return {"keyswitch": keyswitch.enabled(), "evalmod": evalmod.enabled(), "hostcache": bool(hostcache),
            "kernel_dir": kernel_dir, "patched": sorted(changed), "modules": [keyswitch, evalmod]}


if __name__ == "__main__":
    print({k: v for k, v in install().items() if k != "modules"})
