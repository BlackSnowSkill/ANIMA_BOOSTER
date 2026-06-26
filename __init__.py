"""
© 2026 blacksnowskill (BSS). All rights reserved.
Developed by: blacksnowskill (BSS)

ANIMA_BOOSTER — ComfyUI custom node package
Maximum performance optimization for Anima DiT 2B model.

Nodes:
  - AnimaBoosterLoader       : All-in-one optimized model loader (BSS)
  - AnimaLatentImage         : Latent generator with auto-valid dimensions for Anima (BSS)
  - AnimaTeaCache            : Adaptive timestep-aware caching (BSS)
"""

import logging

logger = logging.getLogger("BSS_ANIMA_BOOSTER")

try:
    from .nodes.node_loader import AnimaBoosterLoader, AnimaBoosterCheckpointLoader
    from .nodes.node_latent import AnimaLatentImage
    from .nodes.node_teacache import AnimaTeaCache
    _load_error = None
except Exception as e:
    _load_error = e
    logger.error(f"[ANIMA_BOOSTER] Failed to load nodes: {e}", exc_info=True)
    AnimaBoosterLoader = None
    AnimaBoosterCheckpointLoader = None
    AnimaLatentImage = None
    AnimaTeaCache = None

__version__ = "1.3.2"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Node category for ComfyUI
BSS_CATEGORY = "BSS/AnimaBooster"

if AnimaBoosterLoader is not None:
    NODE_CLASS_MAPPINGS["AnimaBoosterLoader"] = AnimaBoosterLoader
    NODE_DISPLAY_NAME_MAPPINGS["AnimaBoosterLoader"] = "Anima Booster Loader (BSS)"

if AnimaBoosterCheckpointLoader is not None:
    NODE_CLASS_MAPPINGS["AnimaBoosterCheckpointLoader"] = AnimaBoosterCheckpointLoader
    NODE_DISPLAY_NAME_MAPPINGS["AnimaBoosterCheckpointLoader"] = "Anima Checkpoint Loader (BSS)"

if AnimaLatentImage is not None:
    NODE_CLASS_MAPPINGS["AnimaLatentImage"] = AnimaLatentImage
    NODE_DISPLAY_NAME_MAPPINGS["AnimaLatentImage"] = "Anima Latent Image (BSS)"

if AnimaTeaCache is not None:
    NODE_CLASS_MAPPINGS["AnimaTeaCache"] = AnimaTeaCache
    NODE_DISPLAY_NAME_MAPPINGS["AnimaTeaCache"] = "Anima TeaCache (BSS)"





WEB_DIRECTORY = "./web"

if _load_error:
    logger.warning(f"[ANIMA_BOOSTER] Partial load due to error: {_load_error}")
else:
    loaded = list(NODE_CLASS_MAPPINGS.keys())
    logger.info(f"[ANIMA_BOOSTER] Loaded {len(loaded)} nodes: {loaded} | Version: v1.3.0 | Authorship: blacksnowskill (BSS)")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "__version__"]
