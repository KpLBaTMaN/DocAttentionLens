CREATE TABLE method_analyses (
    analysis_id TEXT PRIMARY KEY,
    generation_capture_id TEXT NOT NULL,
    method_id TEXT NOT NULL,
    variant_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX method_analyses_capture ON method_analyses(generation_capture_id, created_at);
