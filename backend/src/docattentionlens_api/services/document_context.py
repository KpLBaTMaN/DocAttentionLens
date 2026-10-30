"""Resolve annotations from immutable dataset/experiment lineage, never client capabilities."""
from hashlib import sha256
from docattentionlens.datasets.adapter import canonical
from docattentionlens.methods.contracts import DocumentInputs
from docattentionlens_api.errors import fail
from docattentionlens_api.repositories.documents import DocumentRepository


def annotations(jobs, record):
    documents=DocumentRepository(jobs.settings)
    protocol_id=record.provenance.get("experiment_protocol_id")
    study_id=record.provenance.get("document_study_id")
    if not protocol_id and not study_id:
        with jobs.connect() as conn:
            owner=conn.execute("SELECT experiment_id FROM experiment_captures WHERE capture_id=%s",(record.capture_id,)).fetchone()
        if owner: protocol_id=owner["experiment_id"]
    if study_id:
        from docattentionlens_api.repositories.document_studies import DocumentStudyRepository
        study=DocumentStudyRepository(jobs).get(study_id)
        case=study.case
        snapshot=documents.snapshot(case.dataset_id)
        pages=tuple(p for p in snapshot.pages if p.page_id in case.page_ids)
        elements=tuple(e for e in snapshot.elements if e.document_id==case.document_id and e.page_id in case.page_ids)
        origin="document experiment; retained source element identity"
    elif protocol_id:
        from docattentionlens_api.repositories.experiments import ExperimentRepository
        protocol=ExperimentRepository(jobs).get(protocol_id)
        arm=next((a for a in protocol.arms if a.case.case_id==record.case_id),None)
        if arm is None: fail("invalid_reference","Capture case is absent from its experiment protocol")
        case=arm.case; snapshot=documents.snapshot(case.dataset_id)
        pages=tuple([arm.page]+[p for p in snapshot.pages if p.page_id in case.page_ids])
        elements=tuple(arm.elements+[e for e in snapshot.elements if e.document_id==case.document_id and e.page_id in case.page_ids])
        origin="reviewed layout experiment lineage"
    else:
        case=documents.case(record.case_id); snapshot=documents.snapshot(case.dataset_id)
        pages=tuple(p for p in snapshot.pages if p.page_id in case.page_ids)
        elements=tuple(e for e in snapshot.elements if e.document_id==case.document_id and e.page_id in case.page_ids)
        origin="imported" if snapshot.summary.adapter_version.startswith("uploaded_") else "dataset"
    support_categories={e.element_id:e.element_category for e in snapshot.elements if e.document_id==case.document_id and e.element_id in case.support_element_ids}
    # Layout-derived IDs retain source support categories through element lineage.
    support_categories.update({e.element_id:e.element_category for e in elements if e.element_id in case.support_element_ids})
    identity={"case_sha256":case.content_sha256,"annotation_version":case.annotation_version,"support_categories":support_categories,
              "elements":[e.model_dump(mode="json") for e in elements],"pages":[p.model_dump(mode="json") for p in pages]}
    question=next((q for q in snapshot.questions if q.question_id==case.question_id),None)
    return DocumentInputs(case,elements,pages,sha256(canonical(identity)).hexdigest(),origin,question.question_type if question else "unknown",support_categories)
