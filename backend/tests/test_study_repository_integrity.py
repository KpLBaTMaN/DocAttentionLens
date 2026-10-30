"""Stored-JSON checksum verification and single-query listing for study/experiment repositories."""
import json
from hashlib import sha256
from types import SimpleNamespace as NS
import pytest
from fastapi import HTTPException
from docattentionlens.datasets.adapter import canonical
from docattentionlens.datasets.contracts import CaseRecord
from docattentionlens.experiments.document_studies import DocumentStudy,StudyOutcome,freeze
from docattentionlens.experiments.attention_interventions import AttentionExperiment
from docattentionlens_api.repositories.document_studies import DocumentStudyRepository
from docattentionlens_api.repositories.attention_interventions import AttentionExperimentRepository

CASE=CaseRecord(key="case_a",case_id="case_a",dataset_id="dataset_a",dataset_version="v1",document_id="doc_a",question_id="question_a",
    query_variant="document",task_scope="document",query="Question",page_ids=["page_a"],page_image_sha256={"page_a":"a"*64},annotation_version="v1",
    support_page_ids=["page_a"],support_element_ids=["element_b"],reference_answer="fixture only",producer_version="fixture")


class Jobs:
    """Fake metadata store: counts connections and serves rows keyed by record id."""
    def __init__(self,root,key):
        self.settings=NS(artifact_root=root); self.key=key; self.rows=[]; self.connections=0
    def connect(self):
        self.connections+=1; jobs=self
        class Conn:
            def __enter__(self): return self
            def __exit__(self,*exc): return False
            def execute(self,query,params):
                rows=[r for r in jobs.rows if "LIMIT" in query or r[jobs.key]==params[0]]
                return NS(fetchone=lambda:rows[0] if rows else None,fetchall=lambda:rows)
        return Conn()


def store(repo,jobs,identifier,content):
    """Freeze content exactly as written (as an older model would have) and register its digest."""
    content={k:v for k,v in content.items() if k!="content_sha256"}
    content["content_sha256"]=sha256(canonical(content)).hexdigest()
    path=repo.path(identifier); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(canonical(content))
    jobs.rows.append({jobs.key:identifier,"content_hash":content["content_sha256"]})
    return path


def study(n):
    outcome=StudyOutcome(name="original",capture_id="generation_"+"c"*32,capture_sha256="d"*64,answer="42",output_token_ids=[1,2],input_positions=7)
    return DocumentStudy(study_id="document_study_"+f"{n:032x}",kind="result",created_at="2026-10-01T00:00:00+00:00",
        source_generation_capture_id="generation_"+"a"*32,source_capture_sha256="b"*64,source_case_id="case_a",case=CASE,
        method_id="document.element_masking",variant_id="v1",parameters={},model_registration_id="builtin",model_revision="fixture",
        arms=[{"name":"original"}],outcomes=[outcome],provenance={"purpose":"unit_fixture"})


def test_legacy_document_study_without_later_outcome_fields_verifies(tmp_path):
    jobs=Jobs(tmp_path,"study_id"); repo=DocumentStudyRepository(jobs)
    legacy=study(1).model_dump(mode="json")
    for outcome in legacy["outcomes"]: outcome.pop("half_rule_totals"); outcome.pop("attribution")
    store(repo,jobs,legacy["study_id"],legacy)
    record=repo.get(legacy["study_id"])
    assert record.outcomes[0].half_rule_totals is None
    # The old check re-dumped through the current model, which now adds the missing keys.
    assert freeze(record).content_sha256!=record.content_sha256


def test_tampered_document_study_is_rejected_and_list_uses_one_query(tmp_path):
    jobs=Jobs(tmp_path,"study_id"); repo=DocumentStudyRepository(jobs)
    paths=[store(repo,jobs,r.study_id,r.model_dump(mode="json")) for r in (study(2),study(3),study(4))]
    jobs.connections=0
    assert [r.study_id for r in repo.list("case_a")]==[r["study_id"] for r in jobs.rows]
    assert jobs.connections==1
    tampered=json.loads(paths[1].read_bytes()); tampered["outcomes"][0]["answer"]="43"
    paths[1].write_bytes(canonical(tampered))
    for call in (lambda:repo.get(tampered["study_id"]),lambda:repo.list("case_a")):
        with pytest.raises(HTTPException) as error: call()
        assert error.value.status_code==409
    jobs.rows[0]["content_hash"]="e"*64  # metadata digest differs from an untouched file
    with pytest.raises(HTTPException) as error: repo.get(jobs.rows[0]["study_id"])
    assert error.value.status_code==409


def test_attention_experiment_legacy_verifies_tamper_rejected_single_query(tmp_path):
    jobs=Jobs(tmp_path,"experiment_id"); repo=AttentionExperimentRepository(jobs)
    def experiment(n):
        return AttentionExperiment(experiment_id="attention_experiment_"+f"{n:032x}",kind="prepared",created_at="2026-10-01T00:00:00+00:00",
            source_capture_id="generation_"+"a"*32,source_capture_sha256="b"*64,source_case_id="case_a",case=CASE,role="generation",
            method_id="attention.head_ablation",variant_id="captured_attention_v1",parameters={},model_registration_id="builtin",
            model_revision="fixture",page_ids=["page_a"],intervention={},provenance={"purpose":"unit_fixture"}).model_dump(mode="json")
    legacy=experiment(1); legacy.pop("result")  # optional field absent from an older record
    paths=[store(repo,jobs,c["experiment_id"],c) for c in (legacy,experiment(2))]
    jobs.connections=0
    assert len(repo.list("case_a"))==2 and jobs.connections==1
    tampered=json.loads(paths[0].read_bytes()); tampered["parameters"]["head"]=1
    paths[0].write_bytes(canonical(tampered))
    with pytest.raises(HTTPException) as error: repo.get(legacy["experiment_id"])
    assert error.value.status_code==409
