CREATE TABLE element_marking_drafts (
    document_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL,
    content JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
