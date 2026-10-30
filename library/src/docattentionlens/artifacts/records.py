from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class OperationalReceipt(BaseModel):
    """Operational evidence is explicitly separate from scientific result records."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["0.1.0"] = "0.1.0"
    record_id: str
    record_kind: Literal["operational_receipt"] = "operational_receipt"
    created_at: datetime
    producer_version: str
    dependencies: list[str]
    provenance: dict[str, str]
    output: dict[str, Any]
