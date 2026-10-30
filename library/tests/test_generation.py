"""Explicit numerical fixtures; never published as model evidence."""
from types import SimpleNamespace as NS
from io import BytesIO
import numpy as np
import pytest
from PIL import Image
from docattentionlens.generation.contracts import GenerationPosition
from docattentionlens.generation.inspection import attention_row,pixel_preview
from docattentionlens.generation.qwen import pack_pixels


def fixture():
    positions=[
        GenerationPosition(index=0,token_id=10,piece="image",role="image",page_id="page_a",bbox=[0,0,28,28],ownership={"element_a":.75,"background":.25}),
        GenerationPosition(index=1,token_id=10,piece="image",role="image",page_id="page_b",bbox=[0,0,28,28],ownership={"unreviewed_geometry":1}),
        GenerationPosition(index=2,token_id=11,piece="question",role="query",ownership={"non_image:query":1}),
        GenerationPosition(index=3,token_id=12,piece="previous output",role="generated",ownership={"non_image:generated":1}),
        GenerationPosition(index=4,token_id=13,piece="predicted",role="generated",ownership={"non_image:generated":1})]
    record=NS(capture_id="test_fixture",steps=[NS(step=1,target_count=4,source_index=3,predicted_index=4)],layer_count=2,query_head_count=3,positions=positions)
    values=np.tile(np.array([.1,.2,.3,.4],dtype=np.float32),(2,3,1))
    values[1,2]=[.4,.3,.2,.1]
    return record,values


def test_prediction_row_keeps_full_domain_and_page_residuals():
    record,values=fixture()
    row=attention_row(record,values,1,0,0)
    assert row.source_index==3 and row.predicted_index==4
    assert len(row.values)==4  # Newly predicted token is not a causal target.
    assert row.image_mass==pytest.approx(.3)
    assert row.total_mass==pytest.approx(1)
    assert row.allocation_error<1e-8
    allocations={(a.bucket,a.page_id):a.mass for a in row.allocations}
    assert allocations["element_a","page_a"]==pytest.approx(.075)
    assert allocations["background","page_a"]==pytest.approx(.025)
    assert allocations["unreviewed_geometry","page_b"]==pytest.approx(.2)
    assert allocations["non_image:query",None]==pytest.approx(.3)
    assert attention_row(record,values,1,1,2).values!=row.values


def test_uncaptured_masked_or_incompatible_rows_are_refused():
    record,values=fixture()
    for selection in [(0,0,0),(1,2,0),(1,0,3),(-1,0,0)]:
        with pytest.raises(ValueError):attention_row(record,values,*selection)
    with pytest.raises(ValueError):attention_row(record,values[:,:,:3],1,0,0)
    bad=values.copy();bad[0,0,0]=-1
    with pytest.raises(ValueError):attention_row(record,bad,1,0,0)
    record.steps[0].predicted_index=3
    with pytest.raises(ValueError,match="alignment"):attention_row(record,values,1,0,0)


def test_no_silent_renormalization_or_negative_ownership():
    record,values=fixture()
    with pytest.raises(ValueError,match="normalization"):attention_row(record,values*.5,1,0,0)
    record.positions[0].ownership={"element_a":-1,"background":2}
    with pytest.raises(ValueError,match="Ownership"):attention_row(record,values,1,0,0)


@pytest.mark.parametrize("patch",[14,16])  # Qwen2/2.5-VL and Qwen3-VL patch sizes
def test_consumed_patch_preview_inverts_grouping_and_orientation(patch):
    grid=[1,4,6]
    pixels=np.random.default_rng(7).integers(0,256,(4*patch,6*patch,3),dtype=np.uint8)
    ip=NS(resample=Image.Resampling.BICUBIC,rescale_factor=1/255,image_mean=[.48,.46,.41],image_std=[.27,.26,.28],patch_size=patch)
    packed=pack_pixels(Image.fromarray(pixels),grid,ip)
    rendered=pixel_preview(packed,grid,ip.image_mean,ip.image_std,ip.rescale_factor)
    actual=np.asarray(Image.open(BytesIO(rendered)))
    assert np.array_equal(actual,pixels)
    with pytest.raises(ValueError):pixel_preview(packed,[2,4,6],ip.image_mean,ip.image_std,ip.rescale_factor)
def test_registered_checkpoint_integrity_is_checked_before_loading(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import pytest
    from docattentionlens.generation import qwen
    monkeypatch.setattr(qwen, '_runtime', None)
    monkeypatch.setattr('docattentionlens.retrieval.colpali.release_runtime', lambda: None)
    monkeypatch.setattr('docattentionlens.retrieval.colqwen.release_runtime', lambda: None)
    cuda=SimpleNamespace(is_available=lambda:True,is_bf16_supported=lambda:True,mem_get_info=lambda:(16*1024**3,16*1024**3))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=cuda))
    def forbidden_loader(*args,**kwargs):
        pytest.fail('Changed checkpoint bytes must be rejected before a model or processor is loaded')
    loader=SimpleNamespace(from_pretrained=forbidden_loader)
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoProcessor=loader,Qwen2TokenizerFast=loader,Qwen2_5_VLForConditionalGeneration=loader))
    (tmp_path/'model.safetensors').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='Checkpoint bytes differ'):
        qwen.runtime(tmp_path,{'registration_id':'fixture','path':str(tmp_path),'files':{'model.safetensors':'incorrect'}})

