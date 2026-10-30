import os
import re
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from docattentionlens.artifacts.records import OperationalReceipt


class FileReceiptStore:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, record_id: str) -> Path:
        if not re.fullmatch(r"receipt_[0-9a-f]{32}", record_id):
            raise ValueError("Invalid receipt ID")
        return self.root / "receipts" / f"{record_id}.json"

    def probe(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        probe = self.root / f".probe-{uuid4().hex}"
        try:
            with probe.open("xb") as stream:
                stream.write(b"ready")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            probe.unlink(missing_ok=True)

    def put(self, receipt: OperationalReceipt) -> str:
        body = receipt.model_dump_json().encode("utf-8")
        final = self._path(receipt.record_id)
        final.parent.mkdir(parents=True, exist_ok=True)
        scratch = self.root / "staging"
        scratch.mkdir(parents=True, exist_ok=True)
        staged = scratch / f"{uuid4().hex}.json"
        try:
            with staged.open("xb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            # Atomic, exclusive publication on the same volume; never overwrite.
            os.link(staged, final)
        finally:
            staged.unlink(missing_ok=True)
        return sha256(body).hexdigest()

    def read(self, record_id: str, expected_hash: str) -> OperationalReceipt:
        body = self._path(record_id).read_bytes()
        if sha256(body).hexdigest() != expected_hash:
            raise ValueError("Receipt checksum mismatch")
        record = OperationalReceipt.model_validate_json(body)
        if record.record_id != record_id:
            raise ValueError("Receipt identity mismatch")
        return record
