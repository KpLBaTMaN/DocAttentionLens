from datetime import UTC, datetime
import numpy as np
import pytest
from docattentionlens_api.repositories.captures import CaptureStore
from docattentionlens.retrieval.contracts import RetrievalCapture


def test_immutable_numeric_payloads_are_verified_without_pickle(tmp_path):
    store=CaptureStore(tmp_path)
    # Explicit storage fixture; not submitted to the scientific artifact catalogue.
    record=RetrievalCapture(capture_id='capture_'+'a'*32,case_id='storage_fixture',created_at=datetime.now(UTC).isoformat(),producer_version='test',dependencies=[],dataset_version='test',annotation_version='test',query='test',query_positions=[],query_array='query',pages=[],provenance={'purpose':'test_fixture'},scorer={},geometry={},validation={})
    digest=store.put(record,{'query':np.array([[1,-2]],dtype=np.float32)})
    reopened=store.read(record.capture_id,digest)
    assert store.array(reopened,'query').tolist()==[[1,-2]]
    with pytest.raises(FileExistsError):store.put(record,{'query':np.array([[1,-2]],dtype=np.float32)})
    with pytest.raises(ValueError):store.path('../../outside')
    array=store.path(record.capture_id)/'query.npy'
    array.write_bytes(b'corrupt')
    with pytest.raises(ValueError):store.array(reopened,'query')
    manifest=store.path(record.capture_id)/'manifest.json'
    manifest.write_bytes(b'{}')
    with pytest.raises(ValueError):store.read(record.capture_id,digest)


def test_generator_chunks_are_separate_immutable_and_axis_named(tmp_path):
    from docattentionlens_api.repositories.generations import GenerationStore
    from docattentionlens.generation.contracts import GenerationCapture,ContextManifest
    store=GenerationStore(tmp_path)
    record=GenerationCapture(capture_id='generation_'+'b'*32,case_id='storage_fixture',retrieval_capture_id='test',
        created_at=datetime.now(UTC).isoformat(),producer_version='test',dependencies=[],
        context=ContextManifest(retrieved_page_ids=[],selected_page_ids=[],omitted_page_ids=[],query='fixture',prompt='fixture',pages=[],input_token_ids=[],input_attention_mask=[]),
        positions=[],output_token_ids=[1],answer='fixture',stop_reason='eos',steps=[],layer_count=1,query_head_count=1,kv_head_count=1,
        provenance={'purpose':'unit_fixture_not_model_evidence'},coverage={},validation={})
    digest=store.put(record,{'step_0':np.ones((1,1,1),dtype=np.float32),'pixels_0':np.zeros((4,1176),dtype=np.float32)})
    reopened=store.read(record.capture_id,digest)
    assert reopened.arrays[0].axes==['layer','query_head','causal_target_position']
    assert store.array(reopened,'step_0').shape==(1,1,1)
    with pytest.raises(ValueError):store.path('capture_'+'b'*32)
    with pytest.raises(ValueError):store.array(reopened,'step_8')
    with pytest.raises(FileExistsError):store.put(record,{'step_0':np.ones((1,1,1),dtype=np.float32)})
    (store.path(record.capture_id)/'step_0.npy').write_bytes(b'corrupt')
    with pytest.raises(ValueError):store.array(reopened,'step_0')
