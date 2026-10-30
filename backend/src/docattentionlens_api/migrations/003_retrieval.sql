ALTER TABLE workers DROP CONSTRAINT workers_resource_class_check;
ALTER TABLE workers ADD CONSTRAINT workers_resource_class_check CHECK(resource_class IN ('cpu','gpu'));
ALTER TABLE workers ADD COLUMN capabilities JSONB NOT NULL DEFAULT '{}';
ALTER TABLE jobs DROP CONSTRAINT jobs_resource_class_check;
ALTER TABLE jobs ADD CONSTRAINT jobs_resource_class_check CHECK(resource_class IN ('cpu','gpu'));
ALTER TABLE jobs DROP CONSTRAINT jobs_kind_check;
ALTER TABLE jobs ADD CONSTRAINT jobs_kind_check CHECK(kind IN ('integrity_check','dataset_import','retrieval_capture'));
CREATE TABLE retrieval_captures (
    capture_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    content_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX retrieval_captures_case ON retrieval_captures(case_id,created_at DESC,capture_id DESC);
