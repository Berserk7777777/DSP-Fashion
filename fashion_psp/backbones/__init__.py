"""Backbone integration declarations; implementations remain opt-in."""

from .base import AdapterStatus, BackboneAdapterBase
from .difashion import DiFashionAdapter
from .dualfashion import DualFashionAdapter
from .fashiondpo import FashionDPOAdapter

__all__ = ["AdapterStatus", "BackboneAdapterBase", "DiFashionAdapter", "DualFashionAdapter", "FashionDPOAdapter"]
