CREATE TABLE experiment_captures (
    capture_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    kind TEXT NOT NULL CHECK(kind IN ('retrieval','generation')),
    arm TEXT NOT NULL,
    content_hash TEXT NOT NULL
);
CREATE INDEX experiment_capture_owner ON experiment_captures(experiment_id);
