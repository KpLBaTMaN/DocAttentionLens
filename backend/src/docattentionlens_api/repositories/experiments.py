"""Immutable experiment manifests/assets and metadata admission."""
from hashlib import sha256
import json
import re
from docattentionlens.datasets.adapter import canonical
from docattentionlens.experiments.contracts import ExperimentRecord
from docattentionlens_api.repositories.captures import CaptureStore
from docattentionlens_api.errors import fail


class ExperimentStore:
    def __init__(self, root): self.root=root/"experiments"

    def path(self, identifier):
        if not re.fullmatch(r"experiment_[a-f0-9]{32}", identifier): raise ValueError("Invalid experiment ID")
        return self.root/identifier

    def put(self, record, assets=None):
        folder=self.path(record.experiment_id)
        for name, raw in (assets or {}).items():
            arm=next(a for a in record.arms if a.name==name)
            if sha256(raw).hexdigest()!=arm.asset_sha256: raise ValueError("Asset checksum differs")
            CaptureStore.immutable(folder/(name+".png"),raw)
        content=record.model_dump(mode="json",exclude={"content_sha256"})
        digest=sha256(canonical(content)).hexdigest(); content["content_sha256"]=digest
        CaptureStore.immutable(folder/"manifest.json",canonical(content))
        return digest

    def read(self, identifier, expected):
        content=json.loads((self.path(identifier)/"manifest.json").read_bytes())
        digest=sha256(canonical({k:v for k,v in content.items() if k!="content_sha256"})).hexdigest()
        if digest!=expected or content["content_sha256"]!=expected: raise ValueError("Experiment checksum differs")
        record=ExperimentRecord.model_validate(content)
        if record.experiment_id!=identifier: raise ValueError("Experiment identity differs")
        return record

    def image(self, record, name):
        arm=next((a for a in record.arms if a.name==name),None)
        if not arm: raise ValueError("Unknown arm")
        # Final protocols/results depend on the immutable draft asset owner.
        owner=record.provenance.get("asset_owner",record.experiment_id)
        raw=(self.path(owner)/(name+".png")).read_bytes()
        if sha256(raw).hexdigest()!=arm.asset_sha256: raise ValueError("Experiment asset checksum differs")
        return raw


class ExperimentRepository:
    def __init__(self,jobs): self.jobs=jobs; self.store=ExperimentStore(jobs.settings.artifact_root)

    def get(self,identifier):
        with self.jobs.connect() as conn: row=conn.execute("SELECT * FROM experiments WHERE experiment_id=%s",(identifier,)).fetchone()
        if not row: fail("not_found","Unknown experiment",404)
        try: return self.store.read(identifier,row["content_hash"])
        except (OSError,ValueError,KeyError): fail("artifact_incomplete","Experiment failed integrity verification",409)

    def admit(self,record,digest):
        with self.jobs.connect() as conn:
            conn.execute("INSERT INTO experiments(experiment_id,source_case_id,record_kind,content_hash,name) VALUES (%s,%s,%s,%s,%s)",(record.experiment_id,record.source_case_id,record.record_kind,digest,record.name))

    def list(self,case_id,limit,before):
        with self.jobs.connect() as conn:
            rows=conn.execute("SELECT experiment_id FROM experiments WHERE (%s::text IS NULL OR source_case_id=%s) AND (%s::text IS NULL OR (created_at,experiment_id)<(SELECT created_at,experiment_id FROM experiments WHERE experiment_id=%s)) ORDER BY created_at DESC,experiment_id DESC LIMIT %s",(case_id,case_id,before,before,limit)).fetchall()
        return [self.get(r["experiment_id"]) for r in rows]

    def recover_capture_index(self):
        """Rebuild links for already-published results; scientific files stay immutable."""
        from docattentionlens_api.repositories.generations import GenerationStore
        with self.jobs.connect() as conn:
            rows=conn.execute("SELECT experiment_id,content_hash FROM experiments WHERE record_kind='experiment_result'").fetchall()
            for row in rows:
                record=self.store.read(row["experiment_id"],row["content_hash"])
                for outcome in record.outcomes:
                    for kind,store in (("retrieval",CaptureStore(self.jobs.settings.artifact_root)),("generation",GenerationStore(self.jobs.settings.artifact_root))):
                        if kind not in outcome: continue
                        ref=outcome[kind]; capture=store.read(ref["capture_id"],ref["sha256"])
                        if capture.case_id!=next(a.case.case_id for a in record.arms if a.name==outcome["arm"]): raise ValueError("Capture lineage differs")
                        conn.execute("INSERT INTO experiment_captures(capture_id,experiment_id,kind,arm,content_hash) VALUES (%s,%s,%s,%s,%s) ON CONFLICT(capture_id) DO NOTHING",(ref["capture_id"],record.experiment_id,kind,outcome["arm"],ref["sha256"]))


def source_inputs(documents,case,page_ids):
    from docattentionlens_api.services.dataset_sources import source_root
    snapshot=documents.snapshot(case.dataset_id)
    root=source_root(documents.jobs.settings.data_root,snapshot.summary).resolve()
    inputs=[]
    for pid in page_ids:
        page=next(p for p in snapshot.pages if p.page_id==pid)
        path=(root/snapshot.image_files[pid]).resolve()
        if not path.is_relative_to(root): raise ValueError("Asset escapes dataset root")
        raw=path.read_bytes()
        if sha256(raw).hexdigest()!=case.page_image_sha256[pid]: raise ValueError("Original asset differs from frozen case")
        inputs.append((page,raw,[e for e in snapshot.elements if e.page_id==pid]))
    return inputs
