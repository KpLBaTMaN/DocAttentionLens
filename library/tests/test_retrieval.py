"""Small numerical fixtures test algorithms, never stand in for model captures."""
from types import SimpleNamespace
import numpy as np
import pytest
from docattentionlens.retrieval.scoring import maxsim, ownership, contributions


def element(identifier,box,flags=()):
    return SimpleNamespace(element_id=identifier,bbox=SimpleNamespace(x1=box[0],y1=box[1],x2=box[2],y2=box[3]),geometry_flags=flags)


def test_negative_maxima_and_padding_are_not_floored_to_zero():
    query=np.array([[1,0],[0,1]],dtype=np.float32)
    document=np.array([[-.2,-.3],[-.5,-.1],[0,0]],dtype=np.float32)
    score,winners,values,_=maxsim(query,document,document_mask=[True,True,False])
    assert score==pytest.approx(-.3) and winners==[0,1]
    assert values==pytest.approx([-.2,-.1])
    assert maxsim(query,document)[0]==0


def test_masked_query_and_tie_policy_and_frozen_removal():
    q=np.array([[1,0],[0,1],[999,999]],dtype=np.float32)
    d=np.array([[1,0],[1,0],[0,1]],dtype=np.float32)
    score,winners,values,_=maxsim(q,d,[True,True,False])
    assert score==2 and winners==[0,2]
    # Equal winner is retained after removal; removal is not assumed to cause a decrease.
    assert maxsim(q,d,[True,True,False],[False,True,True])[0]==2
    assert maxsim(q,d,[True,True,False],[False,False,True])[0]==1
    with pytest.raises(ValueError):maxsim(q,d,document_mask=[False]*3)


def test_ownership_partitions_overlap_and_keeps_signed_residuals():
    mapping=ownership([0,0,10,10],[element('a',[0,0,10,5]),element('b',[0,0,5,10]),element('flagged',[-5,5,5,10],['out_of_bounds'])])
    assert mapping==pytest.approx({'a':.375,'b':.25,'unreviewed_geometry':.125,'background':.25})
    assert sum(mapping.values())==pytest.approx(1)
    ledger=contributions([0,1],[-2,3],[mapping,{'non_page':1}])
    assert ledger['unreviewed_geometry']==-.25 and ledger['non_page']==3
    assert sum(ledger.values())==pytest.approx(1)


def test_empty_and_degenerate_source_geometry_has_explicit_residual():
    assert ownership([0,0,10,10],[element('bad',[5,5,5,8],['degenerate'])])=={'background':1}
    with pytest.raises(ValueError):ownership([0,0,0,10],[])
    with pytest.raises(ValueError):maxsim([[float('nan')]],[[1]])
    with pytest.raises(ValueError):maxsim([[1]],[[1]],document_mask=[1])


def test_saved_inspection_exposes_masked_positions_as_unavailable_and_validates_removal():
    from docattentionlens.retrieval.inspection import similarity_row,fixed_vector_removal
    record=SimpleNamespace(capture_id='numeric_fixture',query_positions=[SimpleNamespace(eligible=True),SimpleNamespace(eligible=False)])
    page=SimpleNamespace(page_id='numeric_fixture',score=-.2,positions=[SimpleNamespace(index=i,eligible=i<2) for i in range(3)])
    q=np.array([[1],[999]],dtype=np.float32);d=np.array([[-.2],[-.5],[0]],dtype=np.float32)
    row=similarity_row(record,page,q,d,0)
    assert row.similarities[:2]==pytest.approx([-.2,-.5]) and row.similarities[2] is None
    assert row.winner_index==0
    diagnostic=fixed_vector_removal(record,page,q,d,[0])
    assert diagnostic.score==-.5 and diagnostic.score_delta==pytest.approx(-.3)
    with pytest.raises(ValueError):similarity_row(record,page,q,d,1)
    with pytest.raises(ValueError):fixed_vector_removal(record,page,q,d,[0,0])
    with pytest.raises(ValueError):fixed_vector_removal(record,page,q,d,[0,1])
