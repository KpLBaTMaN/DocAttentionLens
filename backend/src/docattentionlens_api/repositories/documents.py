"""Durable immutable snapshots and cases, optimistic investigation revisions."""
from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
from hashlib import sha256
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4

from psycopg.types.json import Jsonb

from docattentionlens_api.errors import fail, readable
from docattentionlens_api.repositories.jobs import JobRepository
from docattentionlens import __version__
from docattentionlens.datasets.adapter import canonical
from docattentionlens.datasets.contracts import (
    CaseRecord, InvestigationRecord, Snapshot,
)


class SnapshotStore:
    def __init__(self, root: Path):
        self.root = root / "datasets"

    def path(self, dataset_id):
        if not re.fullmatch(r"dataset_[a-f0-9]{64}", dataset_id):
            fail("not_found", "Unknown dataset", 404)
        return self.root / (dataset_id + ".json")

    def put(self, snapshot: Snapshot):
        self.root.mkdir(parents=True, exist_ok=True)
        raw = canonical(snapshot.model_dump(mode="json"))
        checksum = sha256(raw).hexdigest()
        destination = self.path(snapshot.summary.dataset_id)
        handle, temporary = tempfile.mkstemp(dir=self.root, prefix="staging-")
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(raw)
                file.flush()
                os.fsync(file.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if sha256(destination.read_bytes()).hexdigest() != checksum:
                    raise ValueError("Dataset snapshot conflict")
        finally:
            Path(temporary).unlink(missing_ok=True)
        return checksum

    @lru_cache(maxsize=2)
    def read(self, dataset_id, checksum):
        try:
            raw = self.path(dataset_id).read_bytes()
        except FileNotFoundError:
            fail("resource_unavailable", "Registered snapshot is missing", 503)
        if sha256(raw).hexdigest() != checksum:
            fail("version_mismatch", "Snapshot checksum verification failed", 409)
        value = Snapshot.model_validate_json(raw)
        if value.summary.dataset_id != dataset_id:
            fail("version_mismatch", "Snapshot identity verification failed", 409)
        return value


class DocumentRepository:
    def __init__(self, settings):
        self.jobs = JobRepository(settings)
        self.store = SnapshotStore(settings.artifact_root)

    def image_bytes(self, snapshot, page):
        settings = self.jobs.settings
        from docattentionlens_api.services.dataset_sources import source_root
        # Uploads and their marked versions share the rendered pages under uploaded-pages.
        root = (settings.artifact_root / "uploaded-pages" if snapshot.summary.adapter_version.startswith("uploaded_") else source_root(settings.data_root, snapshot.summary)).resolve()
        path = (root / snapshot.image_files[page.page_id]).resolve()
        if not path.is_relative_to(root):
            fail("invalid_reference", "Source path escapes registered dataset")
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            fail("source_missing", "Original page asset is missing", 404)
        if sha256(raw).hexdigest() != page.image_sha256:
            fail("version_mismatch", "Original page differs from registered content", 409)
        return raw

    def custom_case(self, document_id, query):
        snapshot, document = self.resource("documents", document_id)
        cid = "case_" + uuid4().hex
        query = query.strip()
        if not query:
            fail("invalid_input", "Enter an investigation question")
        record = CaseRecord(key=cid, case_id=cid, dataset_id=snapshot.summary.dataset_id,
            dataset_version=snapshot.summary.dataset_version, document_id=document_id,
            question_id="prompt_" + sha256(query.encode()).hexdigest(), query_variant="custom", task_scope="document",
            query=query, page_ids=document.page_ids, page_image_sha256={p.page_id: p.image_sha256 for p in snapshot.pages if p.page_id in document.page_ids},
            annotation_version=document.annotation_version, support_page_ids=[], support_element_ids=[], reference_answer="", producer_version=__version__,
            created_at=datetime.now(UTC).isoformat(), dependencies=[snapshot.summary.dataset_id, document_id, *document.page_ids],
            provenance={"adapter_version": snapshot.summary.adapter_version, "prompt_source": "user", "support_role": "evaluation_annotation"})
        return self.admit_case(record)

    def admit_case(self, record):
        """Seal a new case with its content checksum and store it; cases are immutable once admitted."""
        record = record.model_copy(update={"content_sha256": sha256(canonical(record.model_dump(mode="json", exclude={"content_sha256"}))).hexdigest()})
        with self.jobs.connect() as conn:
            conn.execute("INSERT INTO cases(case_id,dataset_id,document_id,content) VALUES (%s,%s,%s,%s)",
                (record.case_id, record.dataset_id, record.document_id, Jsonb(record.model_dump(mode="json"))))
        return record

    def datasets(self):
        with self.jobs.connect() as conn:
            return conn.execute("SELECT * FROM datasets ORDER BY created_at,dataset_id").fetchall()

    def snapshot(self, dataset_id):
        with self.jobs.connect() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE dataset_id=%s", (dataset_id,)).fetchone()
        if not row:
            fail("not_found", "Unknown dataset", 404)
        return self.store.read(dataset_id, row["content_hash"])

    def resource(self, kind, key):
        field = {"documents": "document_id", "pages": "page_id", "elements": "element_id", "questions": "question_id", "tables": "table_id"}[kind]
        for row in self.datasets():
            snapshot = self.store.read(row["dataset_id"], row["content_hash"])
            for record in getattr(snapshot, kind):
                if getattr(record, field) == key:
                    return snapshot, record
        fail("not_found", "Unknown resource", 404)

    def create_case(self, request):
        snapshot, question = self.resource("questions", request.question_id)
        document = next(doc for doc in snapshot.documents if doc.document_id == question.document_id)
        grounding = question.table_grounding
        if grounding is not None and (request.query_variant != "original" or request.task_scope != "table"):
            fail("invalid_scope", "Table questions have one wording and a table scope: use original wording and table scope")
        if grounding is None and request.task_scope == "table":
            fail("invalid_scope", "Only table questions have a table scope")
        if (request.query_variant == "document_scoped" and request.task_scope != "document") or (request.query_variant == "collection_scoped" and request.task_scope != "collection"):
            fail("invalid_scope", "Question wording and task scope disagree")
        # A table question takes its table's page; a pilot document holds only that page.
        default_pages = question.support_page_ids if grounding is not None else document.page_ids
        page_ids = default_pages if request.page_ids is None else request.page_ids
        if not page_ids or len(set(page_ids)) != len(page_ids) or not set(page_ids) <= set(document.page_ids):
            fail("invalid_reference", "Input pages must be unique pages of the question document")
        query = question.question if request.query_variant == "original" else getattr(question, request.query_variant + "_question")
        cid = "case_" + uuid4().hex
        record = CaseRecord(key=cid, case_id=cid, dataset_id=snapshot.summary.dataset_id,
            dataset_version=snapshot.summary.dataset_version, document_id=document.document_id,
            question_id=question.question_id, query_variant=request.query_variant, task_scope=request.task_scope,
            query=query, page_ids=page_ids, page_image_sha256={p.page_id: p.image_sha256 for p in snapshot.pages if p.page_id in page_ids},
            annotation_version=document.annotation_version, support_page_ids=question.support_page_ids,
            support_element_ids=question.support_element_ids, reference_answer=question.answer, producer_version=__version__,
            created_at=datetime.now(UTC).isoformat(), dependencies=[snapshot.summary.dataset_id, document.document_id, question.question_id, *page_ids, *([grounding.table_id] if grounding else [])],
            provenance={"adapter_version":snapshot.summary.adapter_version,"dataset_version":snapshot.summary.dataset_version,"annotation_version":document.annotation_version,"support_role":"evaluation_annotation",
                **({"structure_kind":"table_grounding","authored_input":grounding.authored_input,"supplied_input":"full_page",
                    "input_note":"Question authored against the table crop; supplied as the verified full page." if grounding.authored_input == "table_crop"
                        else "Question authored against the full page; supplied as that page."} if grounding else {})},
            table_grounding=grounding)
        return self.admit_case(record)

    def case(self, case_id):
        with self.jobs.connect() as conn:
            row = conn.execute("SELECT content FROM cases WHERE case_id=%s", (case_id,)).fetchone()
        if not row:
            fail("not_found", "Unknown case", 404)
        content = row["content"]
        if content.get("content_sha256") and sha256(canonical({k:v for k,v in content.items() if k!="content_sha256"})).hexdigest()!=content["content_sha256"]:
            fail("version_mismatch", "Case checksum verification failed", 409)
        return CaseRecord.model_validate(content)

    def cases(self, limit, before):
        with self.jobs.connect() as conn:
            return conn.execute("SELECT content FROM cases WHERE (%s::text IS NULL OR (created_at,case_id)<(SELECT created_at,case_id FROM cases WHERE case_id=%s)) ORDER BY created_at DESC,case_id DESC LIMIT %s", (before, before, limit)).fetchall()

    def validate_view(self, request):
        from docattentionlens.methods.registry import get_method, require_executable, prepare_parameters, get_definition
        case = self.case(request.case_id)
        snapshot = self.snapshot(case.dataset_id)
        state = request.view_state
        try:
            method = get_method(state.attention_method_id, state.attention_variant_id)
            require_executable(method.method_id, method.variant_id)
            if get_definition(method.method_id, method.variant_id).calculate is None:
                raise ValueError("Selected variant is not a saved prediction analysis")
        except (KeyError, ValueError, NotImplementedError):
            fail("invalid_reference", "Selected attention method variant is not supported")
        self.validate_perspective_methods(state.perspective_methods)
        if state.retrieval_capture_id:
            from docattentionlens_api.repositories.captures import CaptureRepository
            capture = CaptureRepository(self.jobs).get(state.retrieval_capture_id)
            if capture.case_id!=case.case_id or state.query_position>=len(capture.query_positions):
                fail("invalid_reference", "Retrieval selection does not belong to this case or captured query")
            if state.investigation_model_role == "retrieval" and state.retrieval_attention and capture.attention_encodings:
                from docattentionlens.retrieval.attention import prepare, methods
                from docattentionlens.retrieval.contracts import RetrievalAttentionRequest
                try:
                    retrieval_method = next(m for m in methods() if m.method_id == state.retrieval_attention_method)
                    prepare(capture, RetrievalAttentionRequest(encoding_id=state.retrieval_encoding_id,
                        source_index=state.retrieval_source_index, method_id=state.retrieval_attention_method,
                        parameters={**state.retrieval_attention_parameters,"layer":state.attention_layer,
                            **({"head":state.attention_head} if "head" in retrieval_method.parameters_schema.get("properties",{}) else {})}))
                except (ValueError, StopIteration) as exc:
                    fail("invalid_parameters", readable(exc))
        if state.generation_capture_id:
            from docattentionlens_api.repositories.generations import GenerationRepository
            generation=GenerationRepository(self.jobs).get(state.generation_capture_id)
            if generation.case_id!=case.case_id or state.retrieval_capture_id!=generation.retrieval_capture_id or state.attention_step not in {s.step for s in generation.steps} or state.attention_layer>=generation.layer_count or state.attention_head>=generation.query_head_count:
                fail("invalid_reference", "Generator selection does not match case, retrieval lineage or captured coverage")
            parameters = {**state.attention_parameters, "step":state.attention_step, "layer":state.attention_layer}
            schema = method.parameters_schema.get("properties", {})
            if "head" in schema:
                parameters["head"] = state.attention_head
            try:
                prepare_parameters(method.method_id, method.variant_id, parameters, generation)
                for key, draft in state.attention_parameter_drafts.items():
                    capture_id, method_id, variant_id = key.split("/")
                    draft_capture = generation if capture_id == generation.capture_id else GenerationRepository(self.jobs).get(capture_id)
                    if draft_capture.case_id != case.case_id:
                        raise ValueError("Method draft capture belongs to a different case")
                    prepare_parameters(method_id, variant_id, draft, draft_capture)
            except (KeyError, ValueError, NotImplementedError) as exc:
                fail("invalid_parameters", readable(exc))
            if state.attention_analysis_id:
                from docattentionlens_api.repositories.analyses import AnalysisRepository
                result = AnalysisRepository(self.jobs).get(state.attention_analysis_id)
                resolved = prepare_parameters(method.method_id, method.variant_id, parameters, generation).model_dump(mode="json")
                if result.generation_capture_id != generation.capture_id or result.method_id != method.method_id or result.variant_id != method.variant_id or result.parameters != resolved:
                    fail("invalid_reference", "Selected analysis does not match the method draft and capture")
        elif state.evidence_signal=="attention":
            fail("invalid_reference", "Attention display requires a generator capture")
        page_ids = set(case.page_ids)
        if state.page_id is not None and state.page_id not in page_ids:
            fail("invalid_reference", "Selected page does not belong to the case")
        elements = {e.element_id: e for e in snapshot.elements if e.document_id == case.document_id and e.page_id in page_ids}
        for key in (state.selected_element_id, state.source_element_id, state.target_element_id):
            if key is not None and key not in elements:
                fail("invalid_reference", "Selected element does not belong to case input pages")
        if state.selected_element_id and elements[state.selected_element_id].page_id != state.page_id:
            fail("invalid_reference", "Selected element is not on the current page")
        display = request.display_settings
        if (display.center_x is None) != (display.center_y is None):
            fail("invalid_request", "View centre needs both coordinates or neither")
        if display.center_x is not None:
            page = next((p for p in snapshot.pages if p.page_id == state.page_id), None)
            if page is None or display.center_x > page.width or display.center_y > page.height:
                fail("invalid_reference", "View centre is outside the selected page")

    @staticmethod
    def validate_perspective_methods(remembered):
        """Each remembered method exists for its model role and belongs to its perspective."""
        from docattentionlens.methods.registry import list_methods
        from docattentionlens.retrieval.attention import methods as retriever_methods
        catalogues = {"generation": list_methods(), "retrieval": retriever_methods()}
        for key, value in remembered.items():
            role, perspective = key.split(":")
            method = next((m for m in catalogues[role] if f"{m.method_id}/{m.variant_id}" == value), None)
            if method is None or method.perspective != perspective:
                fail("invalid_reference", "A remembered analysis method does not belong to its model role and perspective")

    def save(self, request, investigation_id=None):
        self.validate_view(request)
        iid = investigation_id or "investigation_" + uuid4().hex
        with self.jobs.connect() as conn:
            if investigation_id:
                row = conn.execute("SELECT * FROM investigations WHERE investigation_id=%s FOR UPDATE", (iid,)).fetchone()
                if not row:
                    fail("not_found", "Unknown investigation", 404)
                if row["latest_revision"] != request.expected_revision:
                    current = conn.execute("SELECT content FROM investigation_revisions WHERE investigation_id=%s AND revision=%s", (iid, row["latest_revision"])).fetchone()["content"]
                    fail("revision_conflict", "Investigation has a newer revision", 409,
                        current_revision=row["latest_revision"], current_revision_id=current["revision_id"])
                revision = row["latest_revision"] + 1
                conn.execute("UPDATE investigations SET latest_revision=%s WHERE investigation_id=%s", (revision, iid))
            else:
                revision = 1
                conn.execute("INSERT INTO investigations(investigation_id,latest_revision) VALUES (%s,1)", (iid,))
            content = request.model_dump(mode="json", exclude={"expected_revision"})
            rid = "revision_" + uuid4().hex
            content.update(key=iid, investigation_id=iid, revision_id=rid, revision=revision,
                created_at=datetime.now(UTC).isoformat(), producer_version=__version__, schema_version="0.4.0")
            capture_id = request.view_state.retrieval_capture_id
            generation_id = request.view_state.generation_capture_id
            analysis_id = request.view_state.attention_analysis_id
            content.update(record_kind="investigation", dependencies=[request.case_id,*([capture_id] if capture_id else []),*([generation_id] if generation_id else []),*([analysis_id] if analysis_id else [])], provenance={"previous_revision":revision-1 if revision>1 else None,"purpose":"document_inspection","model_evidence":"retrieval_similarity_and_generator_attention" if generation_id else "retrieval_similarity" if capture_id else "unavailable"})
            content["content_sha256"] = sha256(canonical(content)).hexdigest()
            record = InvestigationRecord.model_validate(content)
            conn.execute("INSERT INTO investigation_revisions(investigation_id,revision,case_id,content) VALUES (%s,%s,%s,%s)",
                (iid, revision, request.case_id, Jsonb(record.model_dump(mode="json"))))
        return record

    def investigation(self, iid, revision=None):
        with self.jobs.connect() as conn:
            row = conn.execute("SELECT r.content FROM investigation_revisions r JOIN investigations i USING(investigation_id) WHERE i.investigation_id=%s AND r.revision=COALESCE(%s,i.latest_revision)", (iid, revision)).fetchone()
        if not row:
            fail("not_found", "Unknown investigation revision", 404)
        record = InvestigationRecord.model_validate(row["content"])
        content = {k:v for k,v in row["content"].items() if k!="content_sha256"}
        if sha256(canonical(content)).hexdigest() != record.content_sha256:
            fail("version_mismatch", "Investigation checksum verification failed", 409)
        return record

    def investigations(self, limit, before, case_id):
        with self.jobs.connect() as conn:
            return conn.execute("SELECT r.content FROM investigations i JOIN investigation_revisions r ON r.investigation_id=i.investigation_id AND r.revision=i.latest_revision WHERE (%s::text IS NULL OR r.case_id=%s) AND (%s::text IS NULL OR (i.created_at,i.investigation_id)<(SELECT created_at,investigation_id FROM investigations WHERE investigation_id=%s)) ORDER BY i.created_at DESC,i.investigation_id DESC LIMIT %s", (case_id, case_id, before, before, limit)).fetchall()
