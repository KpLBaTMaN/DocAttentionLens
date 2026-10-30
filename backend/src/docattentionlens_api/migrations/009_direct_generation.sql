-- A supplied-page investigation has no retrieval stage.
ALTER TABLE generation_captures ALTER COLUMN retrieval_capture_id DROP NOT NULL;
