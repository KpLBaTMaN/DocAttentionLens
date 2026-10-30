CREATE TABLE workers (
    worker_id TEXT PRIMARY KEY,
    resource_class TEXT NOT NULL CHECK (resource_class = 'cpu'),
    heartbeat_at TIMESTAMPTZ NOT NULL,
    producer_version TEXT NOT NULL
);
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    input_hash TEXT NOT NULL,
    specification JSONB NOT NULL,
    kind TEXT NOT NULL CHECK (kind = 'integrity_check'),
    resource_class TEXT NOT NULL CHECK (resource_class = 'cpu'),
    state TEXT NOT NULL CHECK (state IN ('queued','running','cancellation_requested','completed','failed','cancelled')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    attempt_id TEXT,
    worker_id TEXT REFERENCES workers(worker_id),
    lease_until TIMESTAMPTZ,
    result_id TEXT,
    error_code TEXT,
    error_message TEXT
);
CREATE INDEX jobs_queue ON jobs(resource_class, created_at, job_id) WHERE state IN ('queued','running','cancellation_requested');
CREATE TABLE attempts (
    attempt_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    worker_id TEXT NOT NULL REFERENCES workers(worker_id),
    state TEXT NOT NULL CHECK (state IN ('running','completed','failed','expired','cancelled')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    finished_at TIMESTAMPTZ,
    error_code TEXT
);
CREATE TABLE job_events (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    attempt_id TEXT,
    state TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE receipts (
    record_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id),
    content_hash TEXT NOT NULL,
    manifest JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
