ALTER TABLE jobs DROP CONSTRAINT jobs_kind_check;
ALTER TABLE jobs ADD CONSTRAINT jobs_kind_check CHECK(kind IN ('integrity_check','dataset_import','retrieval_capture','generation_capture'));
CREATE TABLE generation_captures (
    capture_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    retrieval_capture_id TEXT NOT NULL REFERENCES retrieval_captures(capture_id),
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    content_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX generation_captures_case ON generation_captures(case_id,created_at DESC,capture_id DESC);
