"""Shared block presentation; each direction's source/target predicates live in its method."""
import numpy as np
from .contracts import Calculation
from ._shared import selected_rows, selected_step, statistic
from ._attention_only import table
from .element_allocation import allocate_row

def block(context,p,source_image,target_image):
    record=context.record; step=selected_step(record,p.step)
    if (record.positions[step.source_index].role=="image")!=source_image:
        raise ValueError("Select an image-patch source" if source_image else "Select a textual (non-image) source")
    rows,mask=selected_rows(context,p); original=rows[0,0]
    indices=[i for i,x in enumerate(record.positions[:len(mask)]) if mask[i] and (x.role=="image")==target_image]
    if not indices: raise ValueError("No valid targets occur in the selected modality block")
    values=np.zeros_like(original); values[indices]=original[indices]
    mass=float(values.sum()); total=float(original[mask].sum())
    row=allocate_row(record,values,step,p.step,p.layer,p.head,quantity="modality_block_attention")
    return Calculation(row=row,source_indices=[step.source_index],
        statistics=[statistic("Original-domain block mass",mass),statistic("Original-domain excluded mass",total-mass),
                    statistic("Valid block targets",len(indices),"positions")],
        table=table("Target weights and within-block shares",["Original weight","Conditional share"],
            [f"{i} · {record.positions[i].piece}" for i in indices],
            [[float(values[i]),float(values[i]/mass) if mass>0 else None] for i in indices],"attention mass; dimensionless share"),
        normalization="page/row display keeps original block weights; conditional shares divide by block mass only in the table",
        notices=["Textual means all non-image positions, including prompts, special tokens and generated text.",
                 *(['The block has zero measured mass; conditional shares are unavailable.'] if mass==0 else []),
                 "Separate retriever query/document encodings are not query-to-document cross-attention."])
