"""Immutable numeric payloads; admission remains fenced by the job transaction."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import os
import re
from uuid import uuid4
import numpy as np

from docattentionlens.datasets.adapter import canonical
from docattentionlens.retrieval.contracts import ArrayRef, RetrievalCapture
from docattentionlens_api.errors import fail


class CaptureStore:
    # Layer shards: ColPali has 18 language layers, ColQwen2 28, ColQwen2.5 36.
    array_pattern = r"query|page_[0-7]|attn_(?:query|document|joint|joint_answer)_l(?:[0-9]|[12][0-9]|3[0-5])"
    manifest_type = RetrievalCapture

    def __init__(self, root: Path):
        self.root = root / "retrieval"

    def path(self, identifier):
        if not re.fullmatch(r"capture_[a-f0-9]{32}", identifier):
            raise ValueError("Invalid capture identifier")
        return self.root / identifier

    @staticmethod
    def immutable(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        stage = path.with_name(path.name+"."+uuid4().hex+".staging")
        try:
            with stage.open("xb") as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            os.link(stage,path)
        finally:
            stage.unlink(missing_ok=True)

    def put(self, record, arrays):
        folder = self.path(record.capture_id)
        refs = []
        for name, values in arrays.items():
            if not re.fullmatch(self.array_pattern, name):
                raise ValueError("Invalid array name")
            dtype = self.dtype(name)
            if dtype == np.uint16:
                # Raw bit patterns are stored exactly as given; a cast would change their meaning.
                values = np.asarray(values)
                if values.dtype != np.uint16:
                    raise ValueError("Bit-pattern array must already be uint16")
            elif np.asarray(values).dtype == np.uint16:
                raise ValueError("uint16 bit patterns are stored only for declared bit-pattern arrays")
            else:
                values = np.asarray(values, dtype=np.float32)
            stream = BytesIO(); np.save(stream,values,allow_pickle=False); raw = stream.getvalue()
            self.immutable(folder/(name+".npy"),raw)
            refs.append(ArrayRef(name=name,sha256=sha256(raw).hexdigest(),shape=list(values.shape),axes=self.axes(name),dtype=str(values.dtype),bytes=len(raw)))
        # The manifest is the commit point: a record refused here leaves no readable capture.
        self.admit(record, refs)
        content = record.model_copy(update={"arrays":refs}).model_dump(mode="json",exclude={"content_sha256"})
        digest = sha256(canonical(content)).hexdigest(); content["content_sha256"]=digest
        self.immutable(folder/"manifest.json",canonical(content))
        return digest

    def read(self, identifier, expected):
        content = json.loads((self.path(identifier)/"manifest.json").read_bytes())
        digest = sha256(canonical({k:v for k,v in content.items() if k!="content_sha256"})).hexdigest()
        if digest!=expected or content["content_sha256"]!=expected:
            raise ValueError("Capture checksum differs")
        return self.manifest_type.model_validate(content)

    def admit(self, record, refs):
        """Final manifest check against the stored array references; nothing extra for retrieval captures."""

    def dtype(self, name):
        """Stored dtype per array name; every array here is float32."""
        return np.float32

    def axes(self, name):
        if name.startswith("attn_"):
            return ["query_head", "source_position", "target_position"]
        return ["query_position" if name=="query" else "page_position", "embedding"]

    def array(self, record, name):
        ref = next((r for r in record.arrays if r.name==name),None)
        if ref is None or not re.fullmatch(self.array_pattern, name):
            raise ValueError("Unknown numeric array")
        raw = (self.path(record.capture_id)/(name+".npy")).read_bytes()
        if len(raw)!=ref.bytes or sha256(raw).hexdigest()!=ref.sha256:
            raise ValueError("Array checksum differs")
        values = np.load(BytesIO(raw),allow_pickle=False)
        if list(values.shape)!=ref.shape or values.dtype!=self.dtype(name) or ref.dtype!=str(values.dtype) or (values.dtype==np.float32 and not np.isfinite(values).all()):
            raise ValueError("Array contract differs")
        return values

    def verified_reader(self, record):
        """Array reader for library calculations: a missing or corrupt payload is a 409, never a calculation error."""
        def read(name):
            try:
                return self.array(record, name)
            except (OSError, ValueError):
                fail("artifact_incomplete", "Attention payload is missing or failed integrity verification", 409)
        return read


class CaptureRepository:
    def __init__(self, jobs):
        self.jobs = jobs
        self.store = CaptureStore(jobs.settings.artifact_root)

    def get(self, identifier):
        with self.jobs.connect() as conn:
            row = conn.execute("SELECT * FROM retrieval_captures WHERE capture_id=%s",(identifier,)).fetchone()
            if not row:
                row=conn.execute("SELECT content_hash FROM experiment_captures WHERE capture_id=%s AND kind='retrieval'",(identifier,)).fetchone()
            if not row:
                row=conn.execute("SELECT content_hash FROM attention_experiment_captures WHERE capture_id=%s AND kind='retrieval'",(identifier,)).fetchone()
        if not row:
            fail("not_found","Unknown retrieval capture",404)
        try:
            return self.store.read(identifier,row["content_hash"])
        except (OSError,ValueError,KeyError):
            fail("artifact_incomplete","Capture is missing or failed integrity verification",409)

    def list(self, case_id, limit, before=None):
        with self.jobs.connect() as conn:
            rows = conn.execute("SELECT * FROM retrieval_captures WHERE (%s::text IS NULL OR case_id=%s) AND (%s::text IS NULL OR (created_at,capture_id)<(SELECT created_at,capture_id FROM retrieval_captures WHERE capture_id=%s)) ORDER BY created_at DESC,capture_id DESC LIMIT %s",(case_id,case_id,before,before,limit)).fetchall()
        return [self.get(r["capture_id"]) for r in rows]
