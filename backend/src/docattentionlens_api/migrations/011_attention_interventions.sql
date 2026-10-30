ALTER TABLE jobs DROP CONSTRAINT jobs_kind_check;
ALTER TABLE jobs ADD CONSTRAINT jobs_kind_check CHECK (kind IN ('integrity_check','calculation','dataset_import','retrieval_capture','generation_capture','layout_experiment','document_experiment','attention_intervention'));
CREATE TABLE attention_experiments (
    experiment_id TEXT PRIMARY KEY,
    source_case_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('prepared','result')),
    method_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    job_id TEXT REFERENCES jobs(job_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE attention_experiment_captures (
    capture_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES attention_experiments(experiment_id),
    kind TEXT NOT NULL,
    arm TEXT NOT NULL,
    content_hash TEXT NOT NULL
);
CREATE INDEX attention_experiment_case ON attention_experiments(source_case_id,created_at);
