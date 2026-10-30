"""Each test uses a disposable schema, never truncating the application database."""
from dataclasses import replace
import os
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.types.json import Jsonb

from docattentionlens_api.config import Settings
from docattentionlens_api.migrate import migrate
from docattentionlens_api.repositories.documents import SnapshotStore
from docattentionlens_api.repositories.jobs import JobRepository
from docattentionlens.datasets.adapter import read_dataset

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def database(tmp_path):
    url = os.getenv("DOCATTENTIONLENS_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set DOCATTENTIONLENS_TEST_DATABASE_URL to run PostgreSQL integration tests")
    schema = f"docattentionlens_test_{uuid4().hex}"
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    # URI keeps configuration transport identical in native processes and Docker.
    from urllib.parse import quote
    scoped_url = url + ("&" if "?" in url else "?") + "options=" + quote(f"-csearch_path={schema}", safe="")
    settings = Settings(scoped_url, tmp_path / "artifacts", tmp_path / "data")
    try:
        migrate(settings)
        yield settings
    finally:
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture(scope="session")
def snapshot():
    """The document layout dataset as its adapter reads it, read once and shared: tests must not mutate it."""
    return read_dataset(ROOT / "data/doclayout_mm_rag_bench")


@pytest.fixture
def registered(database, snapshot):
    """Settings whose disposable schema has the document layout snapshot registered.

    test_table_grounding_api.py overrides this fixture with the table export."""
    store = SnapshotStore(database.artifact_root)
    checksum = store.put(snapshot)
    with JobRepository(database).connect() as conn:
        conn.execute("INSERT INTO datasets(dataset_id,content_hash,summary) VALUES (%s,%s,%s)",
            (snapshot.summary.dataset_id, checksum, Jsonb(snapshot.summary.model_dump(mode="json"))))
    return replace(database, data_root=ROOT / "data")
