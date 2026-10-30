"""Resolve a table case's verified structure from its frozen table revision, never client capabilities."""
from hashlib import sha256
from docattentionlens.datasets.adapter import canonical
from docattentionlens.methods.contracts import TableInputs
from docattentionlens_api.errors import fail
from docattentionlens_api.repositories.documents import DocumentRepository


def table_inputs(jobs, record):
    """The table, page and grounding a capture's case reads, or None for a layout case.

    Experiment and document-study captures rerun layout cases only, so they have no table inputs."""
    if record.provenance.get("experiment_protocol_id") or record.provenance.get("document_study_id"):
        return None
    documents = DocumentRepository(jobs.settings)
    case = documents.case(record.case_id)
    ground = case.table_grounding
    if ground is None:
        return None
    snapshot = documents.snapshot(case.dataset_id)
    table = next((t for t in snapshot.tables if t.table_id == ground.table_id), None)
    if table is None or table.table_sha256 != ground.table_sha256:
        fail("version_mismatch", "The case's frozen table revision is absent from its dataset snapshot", 409)
    page = next((p for p in snapshot.pages if p.page_id == table.page_id), None)
    if page is None or page.page_id not in case.page_ids:
        fail("invalid_reference", "The table's page is not an input page of this case")
    identity = {"case_sha256": case.content_sha256, "table_sha256": table.table_sha256, "grounding_sha256": table.grounding_sha256,
                "table": table.model_dump(mode="json"), "grounding": ground.model_dump(mode="json"), "page": page.model_dump(mode="json")}
    return TableInputs(case, table, page, ground, sha256(canonical(identity)).hexdigest())
