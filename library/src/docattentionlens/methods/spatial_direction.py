"""Image-conditioned displacement vectors and directional sector mass."""
from typing import Literal
import numpy as np
from .contracts import RowParameters
from ._attention_only import define
from ._spatial import inputs,result
from ._literature import YEH_2023_ATTENTIONVIZ

class Parameters(RowParameters):
    source_scope: Literal["selected","all_image"] = "all_image"

def calculate(context,p):
    data=inputs(context,p)
    indices,boxes,centres,grid,weights,uniform,measured,mass=data
    delta=centres[None,:,:]-centres[:,None,:]
    distance=np.linalg.norm(delta,axis=-1)
    unit=np.divide(delta,distance[:,:,None],out=np.zeros_like(delta),where=distance[:,:,None]>0)
    vector=(weights[:,:,None]*unit).sum(1); length=np.linalg.norm(vector,axis=-1)
    angle=(np.arctan2(delta[:,:,1],delta[:,:,0])+2*np.pi)%(2*np.pi)
    sector=(np.floor((angle+np.pi/8)/(np.pi/4)).astype(int))%8
    sectors=np.stack([(weights*((sector==k)&(distance>0))).sum(-1) for k in range(8)],axis=1)
    displacement=(weights[:,:,None]*delta).sum(1)
    values=np.column_stack([length,np.diag(weights),displacement,sectors])
    return result(context,p,data,length,"Directional concentration",
        ["Vector length","Self mass","Mean dx (pixels)","Mean dy (pixels)","Right","Down-right","Down","Down-left","Left","Up-left","Up","Up-right"],values,units="dimensionless; displacement in source pixels")

DEFINITION=define("attention.spatial_direction","Spatial Attention Direction","spatial_attention","directional_concentration",
    "Mean unit-displacement vector length, eight sector masses and pixel displacement for image-conditioned patch attention.",
    Parameters,calculate,requires=["attention_weights","full_square_attention_matrices","complete_declared_support","image_mask"],displays=["patch_grid","source_table"],
    references=[YEH_2023_ATTENTIONVIZ],
    literature=("adaptation","AttentionViz (Yeh et al. 2023) reports directional neighbour patterns in ViT heads; the unit-vector concentration, eight sectors and image conditioning quantify them and are not its method."))
