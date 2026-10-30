"""Immutable content-addressed saved-data calculations and searchable metadata."""
from hashlib import sha256
import re
from .captures import CaptureStore
from docattentionlens_api.errors import fail
from docattentionlens.datasets.adapter import canonical
from docattentionlens.methods.contracts import AnalysisResult


class AnalysisRepository:
    def __init__(self, jobs):
        self.jobs = jobs
        self.root = jobs.settings.artifact_root / "analyses"

    def path(self, identifier):
        if not re.fullmatch(r"analysis_[a-f0-9]{64}", identifier):
            fail("not_found", "Unknown method analysis", 404)
        return self.root / (identifier + ".json")

    def find(self, identifier):
        path = self.path(identifier)
        with self.jobs.connect() as conn:
            row = conn.execute("SELECT content_hash FROM method_analyses WHERE analysis_id=%s", (identifier,)).fetchone()
        if row is None:
            return None
        try:
            result = AnalysisResult.model_validate_json(path.read_bytes())
            digest = sha256(canonical(result.model_dump(mode="json", exclude={"content_sha256"}))).hexdigest()
            if digest != row["content_hash"] or result.content_sha256 != digest or result.analysis_id != identifier:
                raise ValueError("Analysis checksum differs")
            return result
        except (OSError, ValueError):
            fail("artifact_incomplete", "Method analysis is missing or failed integrity verification", 409)

    def get(self, identifier):
        result = self.find(identifier)
        if result is None:
            fail("not_found", "Unknown method analysis", 404)
        return result

    def list(self,capture_id=None,limit=25):
        with self.jobs.connect() as conn:
            rows=conn.execute("SELECT analysis_id FROM method_analyses WHERE (%s::text IS NULL OR generation_capture_id=%s) ORDER BY created_at DESC,analysis_id DESC LIMIT %s",(capture_id,capture_id,limit)).fetchall()
        return [self.get(r["analysis_id"]) for r in rows]

    def put(self, result):
        content = result.model_dump(mode="json", exclude={"content_sha256"})
        digest = sha256(canonical(content)).hexdigest()
        content["content_sha256"] = digest
        raw = canonical(content)
        if len(raw) > 4 * 1024 * 1024:
            fail("analysis_budget_exceeded", "Reduce the compared layer/head population", 422)
        path = self.path(result.analysis_id)
        try:
            CaptureStore.immutable(path, raw)
        except FileExistsError:
            if path.read_bytes() != raw:
                fail("version_mismatch", "Analysis identity conflicts with its stored result", 409)
        with self.jobs.connect() as conn:
            conn.execute("INSERT INTO method_analyses(analysis_id,generation_capture_id,method_id,variant_id,content_hash) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                         (result.analysis_id,result.generation_capture_id,result.method_id,result.variant_id,digest))
        return self.get(result.analysis_id)
