"""FashionDPO DDIM integration declaration."""

from fashion_psp.backbones.base import AdapterStatus, BackboneAdapterBase
from fashion_psp.config import BACKBONE_SPECS


FashionDPOAdapter = BackboneAdapterBase(
    spec=BACKBONE_SPECS["fashiondpo"],
    status=AdapterStatus.ONLINE_PENDING,
    integration_notes=(
        "FashionDPO sample.py and inf4eval.py expose a 50-step DDIM argument. "
        "Validate the actual loop count and decoded x0 semantics before enabling PSP."
    ),
)
