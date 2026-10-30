ALTER TABLE jobs DROP CONSTRAINT jobs_kind_check;
ALTER TABLE jobs ADD CONSTRAINT jobs_kind_check CHECK (kind IN ('integrity_check','calculation','dataset_import','retrieval_capture','generation_capture','layout_experiment'));
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    source_case_id TEXT NOT NULL REFERENCES cases(case_id),
    record_kind TEXT NOT NULL CHECK (record_kind IN ('experiment_draft','experiment_protocol','experiment_result')),
    content_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    job_id TEXT REFERENCES jobs(job_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX experiments_recency ON experiments(created_at,experiment_id);
