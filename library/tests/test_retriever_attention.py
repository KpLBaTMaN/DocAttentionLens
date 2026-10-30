"""Encoding attention must allow future keys and honor retriever padding masks."""
from types import SimpleNamespace
import numpy as np
import pytest
from docattentionlens.retrieval.contracts import Position, EncodingAttention, RetrievalAttentionRequest
from docattentionlens.retrieval.attention import calculate, methods


def fixture():
    encoding=EncodingAttention(encoding_id='query',pass_kind='query',layer_count=2,query_head_count=2,
        position_count=3,attention_mask=[1,1,0],layer_arrays=['a','b'])
    capture=SimpleNamespace(capture_id='numeric_fixture',attention_encodings=[encoding],pages=[],
        query_positions=[Position(index=i,token_id=i,piece=str(i),role='query_text',eligible=i<2) for i in range(3)])
    # Source zero attends entirely to a later key. A generator mask would reject it.
    matrix=np.array([[[0,1,0],[.25,.75,0],[0,1,0]],[[.5,.5,0],[1,0,0],[.5,.5,0]]],dtype=np.float32)
    return capture,{'a':matrix,'b':matrix.copy()}


def run(method='attention.row',source=0,parameters=None):
    capture,arrays=fixture()
    return calculate(capture,RetrievalAttentionRequest(encoding_id='query',source_index=source,
        method_id=method,parameters=parameters or {}),arrays.__getitem__)


def test_future_keys_are_valid_and_prediction_identity_is_absent():
    result=run()
    assert result.row.values==[0,1,0] and result.row.source_index==0
    assert result.row.predicted_index is None and result.row.image_mass==0
    assert 'causal' not in result.row.normalization
    with pytest.raises(ValueError,match='source selection'):run(source=2)


def test_entropy_uses_active_keys_including_valid_zero_weights():
    result=run('attention.entropy',parameters={'head':1})
    assert result.statistics[0].value==pytest.approx(1)
    assert result.statistics[1].value==pytest.approx(1)
    assert result.statistics[2].value==2
    # Bidirectional encodings have no causal support; the bound names the valid key count.
    with pytest.raises(ValueError,match="this source's 2 valid target positions") as caught:run('attention.topk_mass',parameters={'k':3})
    assert 'causal' not in str(caught.value)


def test_head_population_and_depth_comparisons_use_captured_counts():
    assert run('attention.mean_heads').row.values==pytest.approx([.25,.75,0])
    result=run('attention.across_layers')
    assert [r.label for r in result.comparisons]==['Layer 0','Layer 1']
    with pytest.raises(ValueError,match='Layer'):run(parameters={'layer':2})
    with pytest.raises(ValueError,match='Head'):run(parameters={'head':2})
    with pytest.raises(ValueError,match='prediction steps'):run(parameters={'step':0})


def test_masked_keys_and_corrupt_axes_are_rejected():
    capture,arrays=fixture()
    arrays['a'][0,0]=[0,.5,.5]
    with pytest.raises(ValueError,match='masked support'):
        calculate(capture,RetrievalAttentionRequest(encoding_id='query'),arrays.__getitem__)
    capture.attention_encodings[0].attention_mask=[1,0,0]
    with pytest.raises(ValueError,match='axes'):
        calculate(capture,RetrievalAttentionRequest(encoding_id='query'),arrays.__getitem__)


def test_all_supported_variants_have_encoding_semantics():
    for method in methods():
        assert method.variant_id=='retriever_encoding_v1'
        assert 'step' not in method.parameters_schema['properties']
        assert 'generator_prediction_row_capture' not in method.requires
        parameters={'k':1} if method.method_id=='attention.topk_mass' else {}
        if method.perspective=='table_aware':
            # Table structure comes from a table grounding case; a query encoding has none.
            with pytest.raises(ValueError,match='table grounding case'):
                run(method.method_id,parameters=parameters)
        elif method.perspective=='document_aware':
            # A query encoding has no document positions: element mass is missing, not zero.
            with pytest.raises(ValueError,match='document encoding with a valid element mapping'):
                run(method.method_id,parameters=parameters)
        elif method.method_id=='attention.spatial_distance':
            # Query tokens have no page position, so a distance is unavailable rather than zero.
            with pytest.raises(ValueError,match='image-patch source'):
                run(method.method_id,parameters=parameters)
        elif method.requires and ('image_mask' in method.requires or 'captured_image_sources' in method.requires):
            with pytest.raises(ValueError,match='image|Spatial|targets'):
                run(method.method_id,parameters=parameters)
        else:
            assert run(method.method_id,parameters=parameters) is not None


def document_fixture(ownership):
    encoding=EncodingAttention(encoding_id='document',pass_kind='document',page_id='page_a',layer_count=1,query_head_count=1,
        position_count=3,attention_mask=[1,1,1],layer_arrays=['a'])
    page=SimpleNamespace(page_id='page_a',ownership=ownership,
        positions=[Position(index=0,token_id=0,piece='<img>',role='visual_patch',eligible=True,bbox=[0,0,1,1]),
                   Position(index=1,token_id=1,piece='<img>',role='visual_patch',eligible=True,bbox=[1,0,2,1]),
                   Position(index=2,token_id=2,piece='Describe',role='visual_prompt',eligible=True)])
    capture=SimpleNamespace(capture_id='numeric_fixture',attention_encodings=[encoding],pages=[page],query_positions=[])
    matrix=np.array([[[.5,.25,.25],[.25,.5,.25],[.25,.25,.5]]],dtype=np.float32)
    return capture,{'a':matrix}


def test_document_aware_methods_need_mapped_elements_in_a_document_encoding():
    mapped=[{'element_a':.5,'background':.5},{'background':1.0},{'non_page':1.0}]
    capture,arrays=document_fixture(mapped)
    body=RetrievalAttentionRequest(encoding_id='document',source_index=0,method_id='attention.element_allocation')
    row=calculate(capture,body,arrays.__getitem__).row
    masses={a.bucket:a.mass for a in row.allocations}
    assert masses==pytest.approx({'element_a':.25,'background':.5,'non_page':.25})
    capture,arrays=document_fixture([{'background':1.0},{'background':1.0},{'non_page':1.0}])
    with pytest.raises(ValueError,match='annotated elements mapped'):
        calculate(capture,body,arrays.__getitem__)
    # General methods remain available on the same unannotated page.
    assert calculate(capture,body.model_copy(update={'method_id':'attention.row'}),arrays.__getitem__).row.values==[.5,.25,.25]
