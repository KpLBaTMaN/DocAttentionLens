"""Generation-specific immutable payload namespace and durable admission."""
import re
import numpy as np
from .captures import CaptureStore
from docattentionlens_api.errors import fail
from docattentionlens.generation.contracts import GenerationCapture
from hashlib import sha256
from docattentionlens.generation import full_attention as full_rows
from docattentionlens.generation.full_attention import triangle_length


class GenerationStore(CaptureStore):
    # full_l{layer}: opt-in lossless BF16 prefill triangles (generation/full_attention.py), stored as uint16 bit patterns.
    # step_0..step_7 on ordinary captures; full-attention captures keep every generated step, step_0..step_63.
    array_pattern = r"step_(?:[0-9]|[1-5][0-9]|6[0-3])|question_[0-9]{1,4}|pixels_[0-1]|full_l(?:0|[1-9][0-9]{0,2})"
    manifest_type = GenerationCapture

    def __init__(self, root):
        self.root = root / "generation"

    def path(self, identifier):
        if not re.fullmatch(r"generation_[a-f0-9]{32}",identifier):
            raise ValueError("Invalid generation identifier")
        return self.root / identifier

    @staticmethod
    def full(name):
        return re.fullmatch(r"full_l(?:0|[1-9][0-9]{0,2})", name) is not None

    def dtype(self, name):
        return np.uint16 if self.full(name) else np.float32

    def axes(self, name):
        if self.full(name):
            return ["query_head","causal_triangle"]
        return ["layer","query_head","causal_target_position"] if name.startswith(("step_","question_")) else ["grouped_visual_patch","channel_temporal_pixels"]

    def put(self, record, arrays):
        full = record.full_attention
        names = sorted(name for name in arrays if self.full(name))
        if full is None:
            if names:
                raise ValueError("Full attention arrays need a full attention declaration")
        else:
            length = triangle_length(full.position_count)
            if names != sorted(full.layer_arrays) or any(np.asarray(arrays[name]).dtype != np.uint16
                    or np.asarray(arrays[name]).shape != (full.query_head_count, length) for name in names):
                raise ValueError("Full attention arrays differ from their declaration")
            # Squares are rebuilt from the decode rows, so every declared step's row must be stored.
            if any(s.array not in arrays or np.asarray(arrays[s.array]).shape != (record.layer_count, record.query_head_count, s.target_count)
                   for s in record.steps):
                raise ValueError("Full attention prediction-step rows differ from their declaration")
        return super().put(record, arrays)

    def source_array(self, record, name):
        """A source row's (layers, heads, targets) values: a retained array, or a prefill row read from the triangles."""
        source = full_rows.prefill_row_index(name)
        if source is None:
            return self.array(record, name)
        full = record.full_attention
        if full is None or not 0 <= source < full.position_count:
            raise ValueError("This prompt row was not captured")
        return np.stack([full_rows.row(self.array(record, layer), full.position_count, source) for layer in full.layer_arrays])

    @staticmethod
    def source_sha256(record, name):
        """Integrity identity of a source row: its array hash, or for a triangle row the triangles' hashes and position."""
        ref = next((r for r in record.arrays if r.name == name), None)
        if ref is not None:
            return ref.sha256
        source = full_rows.prefill_row_index(name)
        if source is None or record.full_attention is None:
            return None
        hashes = {r.name: r.sha256 for r in record.arrays if r.name in record.full_attention.layer_arrays}
        return sha256(("prefill_row|%d|%s" % (source, "|".join(hashes[n] for n in record.full_attention.layer_arrays))).encode()).hexdigest()

    def admit(self, record, refs):
        full = record.full_attention
        if full is not None and sum(r.bytes for r in refs if self.full(r.name)) != full.bytes:
            raise ValueError("Full attention stored bytes differ from their declaration")


class GenerationRepository:
    def __init__(self, jobs):
        self.jobs=jobs
        self.store=GenerationStore(jobs.settings.artifact_root)

    def get(self, identifier):
        with self.jobs.connect() as conn:
            row=conn.execute("SELECT * FROM generation_captures WHERE capture_id=%s",(identifier,)).fetchone()
            if not row:
                row=conn.execute("SELECT content_hash FROM experiment_captures WHERE capture_id=%s AND kind='generation'",(identifier,)).fetchone()
            if not row:
                row=conn.execute("SELECT content_hash FROM document_study_captures WHERE capture_id=%s",(identifier,)).fetchone()
            if not row:
                row=conn.execute("SELECT content_hash FROM attention_experiment_captures WHERE capture_id=%s AND kind='generation'",(identifier,)).fetchone()
        if not row:
            fail("not_found","Unknown generator capture",404)
        try:
            return self.store.read(identifier,row["content_hash"])
        except (OSError,ValueError,KeyError):
            fail("artifact_incomplete","Generator capture is missing or failed integrity verification",409)

    def list(self, case_id, limit, before=None):
        with self.jobs.connect() as conn:
            rows=conn.execute("SELECT * FROM generation_captures WHERE (%s::text IS NULL OR case_id=%s) AND (%s::text IS NULL OR (created_at,capture_id)<(SELECT created_at,capture_id FROM generation_captures WHERE capture_id=%s)) ORDER BY created_at DESC,capture_id DESC LIMIT %s",(case_id,case_id,before,before,limit)).fetchall()
        return [self.get(r["capture_id"]) for r in rows]
