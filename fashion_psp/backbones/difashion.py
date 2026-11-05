"""DiFashion integration declaration.

Online pruning must be extracted from DiFashion.fashion_generation without
changing PNDMScheduler semantics.  In particular, prune every batch-aligned
condition and preserve terminal scheduler decoding for the final reward node.
"""

from fashion_psp.backbones.base import AdapterStatus, BackboneAdapterBase
from fashion_psp.config import BACKBONE_SPECS


DiFashionAdapter = BackboneAdapterBase(
    spec=BACKBONE_SPECS["difashion"],
    status=AdapterStatus.OFFLINE_READY,
    integration_notes=(
        "Existing collector and offline selector are verified. Online adapter must reindex "
        "latents, CFG tensors, category/mutual/history conditions, generators, and trace rows."
    ),
)
