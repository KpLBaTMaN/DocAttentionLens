from hashlib import sha256
from importlib.resources import files

import psycopg

from docattentionlens_api.config import Settings

SCHEMA_VERSION = "012_element_marking"


def migrate(settings: Settings) -> None:
    with psycopg.connect(settings.database_url, connect_timeout=5) as conn:
        # A transaction-scoped lock serializes competing migration commands.
        conn.execute("SELECT pg_advisory_xact_lock(729441001)")
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, checksum TEXT NOT NULL, applied_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp())")
        resources = files("docattentionlens_api.migrations")
        for item in sorted(resources.iterdir(), key=lambda item: item.name):
            if not item.name.endswith(".sql"):
                continue
            version = item.name.removesuffix(".sql")
            sql = item.read_text(encoding="utf-8")
            checksum = sha256(sql.encode()).hexdigest()
            row = conn.execute("SELECT checksum FROM schema_migrations WHERE version = %s", (version,)).fetchone()
            if row:
                if row[0] != checksum:
                    raise RuntimeError(f"Applied migration changed: {version}")
                continue
            conn.execute(sql)
            conn.execute("INSERT INTO schema_migrations(version, checksum) VALUES (%s, %s)", (version, checksum))
    settings.artifact_root.mkdir(parents=True, exist_ok=True)


def main() -> None:
    migrate(Settings.from_env())
    print(f"Metadata schema ready: {SCHEMA_VERSION}")


if __name__ == "__main__":
    main()
