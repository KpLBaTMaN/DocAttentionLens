"""Checksummed immutable plans/results; worker result admission is fenced by JobRepository.finish."""
import json
import re
from hashlib import sha256
from docattentionlens.datasets.adapter import canonical
from docattentionlens_api.errors import fail
from .captures import CaptureStore


class FrozenRecordRepository:
    """One canonical JSON file per frozen record under artifact_root/<folder>; its digest is admitted to <table>.

    Subclasses name the folder, ID pattern and column (key), table, record model, the library's
    freeze function (as a staticmethod) and their messages."""
    folder = pattern = key = table = ""
    model = None
    freeze = None
    unknown = integrity_failed = prepared_only = ""

    def __init__(self,jobs):
        self.jobs=jobs; self.root=jobs.settings.artifact_root/self.folder

    def path(self,identifier):
        if not re.fullmatch(self.pattern,identifier): fail("not_found",self.unknown,404)
        return self.root/(identifier+".json")

    def write(self,record):
        record=self.freeze(record); CaptureStore.immutable(self.path(getattr(record,self.key)),canonical(record.model_dump(mode="json")))
        return record.content_sha256

    def admit(self,record):
        if record.kind!="prepared": raise ValueError(self.prepared_only)
        digest=self.write(record)
        with self.jobs.connect() as conn:
            conn.execute(f"INSERT INTO {self.table}({self.key},source_case_id,kind,method_id,content_hash) VALUES (%s,%s,'prepared',%s,%s)",
                         (getattr(record,self.key),record.source_case_id,record.method_id,digest))
        return self.read(getattr(record,self.key),digest)

    def read(self,identifier,expected):
        """Hash the stored JSON as written (the form freeze() hashed), then validate it.
        Re-dumping through the current model would add fields introduced after a record
        was frozen (e.g. StudyOutcome.half_rule_totals) and break legacy checksums."""
        try:
            content=json.loads(self.path(identifier).read_bytes())
            digest=sha256(canonical({k:v for k,v in content.items() if k!="content_sha256"})).hexdigest()
            if digest!=expected or content.get("content_sha256")!=expected or content.get(self.key)!=identifier:
                raise ValueError("Frozen record checksum differs")
            return self.model.model_validate(content)
        except (OSError,ValueError,AttributeError): fail("artifact_incomplete",self.integrity_failed,409)

    def get(self,identifier):
        self.path(identifier)
        with self.jobs.connect() as conn:
            row=conn.execute(f"SELECT content_hash FROM {self.table} WHERE {self.key}=%s",(identifier,)).fetchone()
        if not row: fail("not_found",self.unknown,404)
        return self.read(identifier,row["content_hash"])

    def list(self,case_id=None,limit=25):
        # One metadata connection for the page; a tampered record still fails closed (409).
        with self.jobs.connect() as conn:
            rows=conn.execute(f"SELECT {self.key},content_hash FROM {self.table} WHERE (%s::text IS NULL OR source_case_id=%s) ORDER BY created_at DESC,{self.key} DESC LIMIT %s",(case_id,case_id,limit)).fetchall()
        return [self.read(r[self.key],r["content_hash"]) for r in rows]
