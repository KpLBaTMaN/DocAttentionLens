"""A bounded CPU operation for verifying execution, not a research method."""

from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field


class IntegrityInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str = Field(default="DocAttentionLens CPU check", max_length=4096)


class IntegrityOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: str = "utf8_sha256_v1"
    byte_count: int
    sha256: str
    interpretation: str = "Infrastructure checksum receipt; no model or scientific measurement."


def check_integrity(value: IntegrityInput) -> IntegrityOutput:
    encoded = value.text.encode("utf-8")
    return IntegrityOutput(byte_count=len(encoded), sha256=sha256(encoded).hexdigest())
