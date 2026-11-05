"""Shared non-executable adapter declaration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fashion_psp.config import BackboneSpec


class AdapterStatus(str, Enum):
    DECLARED = "declared"
    OFFLINE_READY = "offline_ready"
    ONLINE_PENDING = "online_pending"
    ONLINE_READY = "online_ready"


@dataclass(frozen=True)
class BackboneAdapterBase:
    spec: BackboneSpec
    status: AdapterStatus
    integration_notes: str
