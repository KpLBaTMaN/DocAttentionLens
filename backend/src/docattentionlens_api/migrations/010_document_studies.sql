ALTER TABLE jobs DROP CONSTRAINT jobs_kind_check;
ALTER TABLE jobs ADD CONSTRAINT jobs_kind_check CHECK (kind IN ('integrity_check','calculation','dataset_import','retrieval_capture','generation_capture','layout_experiment','document_experiment'));
CREATE TABLE document_studies (
    study_id TEXT PRIMARY KEY,
    source_case_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('prepared','result')),
    method_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    job_id TEXT REFERENCES jobs(job_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE document_study_captures (
    capture_id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL REFERENCES document_studies(study_id),
    arm TEXT NOT NULL,
    content_hash TEXT NOT NULL
);
CREATE INDEX document_study_case ON document_studies(source_case_id,created_at);
