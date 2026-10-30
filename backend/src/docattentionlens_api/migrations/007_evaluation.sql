CREATE TABLE evaluation_studies (
    study_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE evaluation_runs (
    study_id TEXT NOT NULL REFERENCES evaluation_studies(study_id),
    protocol_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    repetition INTEGER NOT NULL CHECK (repetition BETWEEN 1 AND 5),
    job_id TEXT REFERENCES jobs(job_id),
    result_id TEXT REFERENCES experiments(experiment_id),
    retrospective BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY(study_id, protocol_id, repetition),
    UNIQUE(study_id, result_id),
    CHECK (job_id IS NOT NULL OR result_id IS NOT NULL)
);
CREATE TABLE evaluation_evidence (
    evidence_id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL REFERENCES evaluation_studies(study_id),
    record_kind TEXT NOT NULL CHECK (record_kind IN ('answer_review','utility_trial')),
    content_hash TEXT NOT NULL,
    result_id TEXT REFERENCES experiments(experiment_id),
    reviewer TEXT,
    protocol_id TEXT REFERENCES experiments(experiment_id),
    participant TEXT,
    task TEXT,
    condition TEXT,
    trial_order INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(study_id, result_id, reviewer),
    UNIQUE(study_id, protocol_id, participant, task, condition)
);
CREATE INDEX evaluation_studies_recency ON evaluation_studies(created_at, study_id);
CREATE INDEX evaluation_evidence_study ON evaluation_evidence(study_id, created_at);
