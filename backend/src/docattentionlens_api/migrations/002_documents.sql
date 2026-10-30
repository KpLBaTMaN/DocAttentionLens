ALTER TABLE jobs DROP CONSTRAINT jobs_kind_check;
ALTER TABLE jobs ADD CONSTRAINT jobs_kind_check CHECK (kind IN ('integrity_check','dataset_import'));
ALTER TABLE jobs ADD COLUMN output_resource_id TEXT;
ALTER TABLE jobs ADD COLUMN output_resource_kind TEXT;
CREATE TABLE datasets (
    dataset_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    summary JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE cases (
    case_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    document_id TEXT NOT NULL,
    content JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE investigations (
    investigation_id TEXT PRIMARY KEY,
    latest_revision INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE investigation_revisions (
    investigation_id TEXT NOT NULL REFERENCES investigations(investigation_id),
    revision INTEGER NOT NULL,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    content JSONB NOT NULL,
    PRIMARY KEY (investigation_id,revision)
);
