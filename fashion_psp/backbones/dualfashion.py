"""DualFashion flow-matching integration declaration."""

from fashion_psp.backbones.base import AdapterStatus, BackboneAdapterBase
from fashion_psp.config import BACKBONE_SPECS


DualFashionAdapter = BackboneAdapterBase(
    spec=BACKBONE_SPECS["dualfashion"],
    status=AdapterStatus.OFFLINE_READY,
    integration_notes=(
        "The prototype online runner already estimates x0 from a flow-matching transition. "
        "Adapter extraction must retain scheduler step-index semantics when pruning."
    ),
)
