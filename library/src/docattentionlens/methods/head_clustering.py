"""Ward clustering of captured head distributions; exploratory sample labels only."""
from typing import Literal
from pydantic import Field
import numpy as np
from .contracts import RowParameters, AxisIndex, Calculation, ResearchPlot, ResearchPoint
from ._attention_only import define,table
from ._head_population import profiles,divergences
from ._shared import statistic
from ._literature import CLARK_2019, LIN_1991_JSD, WARD_1963, ROUSSEEUW_1987

class Parameters(RowParameters):
    layers: list[AxisIndex] = Field(default_factory=lambda:[0],max_length=128)
    heads: list[AxisIndex] = Field(default_factory=list,max_length=128)
    source_group: Literal["all","text","image","question"] = "all"
    max_sources: int = Field(default=64,ge=2,le=256,strict=True)
    max_clusters: int = Field(default=8,ge=2,le=8,strict=True)

def ward(distance):
    """Lance-Williams Ward update; ties follow ascending cluster identities."""
    n=len(distance); d=np.full((2*n-1,2*n-1),np.inf); d[:n,:n]=distance
    np.fill_diagonal(d,np.inf)
    active=list(range(n)); sizes={i:1 for i in active}; members={i:[i] for i in active}
    linkage=[]; cuts={n:list(range(n))}
    for node in range(n,2*n-1):
        pairs=[(d[a,b],a,b) for a in active for b in active if a<b]
        value,a,b=min(pairs)
        size=sizes[a]+sizes[b]
        linkage.append([a,b,float(value),size])
        for v in active:
            if v in (a,b): continue
            total=sizes[v]+size
            square=((sizes[v]+sizes[a])*d[v,a]**2+(sizes[v]+sizes[b])*d[v,b]**2-sizes[v]*value**2)/total
            d[node,v]=d[v,node]=np.sqrt(max(0,square))
        sizes[node]=size; members[node]=members[a]+members[b]
        active=sorted([v for v in active if v not in (a,b)]+[node])
        labels=[0]*n
        for cluster,v in enumerate(sorted(active,key=lambda v:min(members[v]))):
            for i in members[v]: labels[i]=cluster
        cuts[len(active)]=labels
    return linkage,cuts

def silhouette(distance,labels):
    labels=np.asarray(labels); scores=[]
    for i,c in enumerate(labels):
        same=np.flatnonzero(labels==c); same=same[same!=i]
        if not len(same) or len(np.unique(labels))<2: scores.append(0); continue
        a=float(distance[i,same].mean())
        b=min(float(distance[i,labels==other].mean()) for other in np.unique(labels) if other!=c)
        scores.append((b-a)/max(a,b) if max(a,b)>0 else 0)
    return float(np.mean(scores))

def projection(distance):
    n=len(distance); centre=np.eye(n)-np.ones((n,n))/n
    gram=-.5*centre@(distance**2)@centre
    values,vectors=np.linalg.eigh(gram); order=np.argsort(values)[::-1][:2]
    coordinates=vectors[:,order]*np.sqrt(np.maximum(values[order],0))
    for col in range(coordinates.shape[1]):
        anchor=np.argmax(np.abs(coordinates[:,col]))
        if coordinates[anchor,col]<0: coordinates[:,col]*=-1
    return np.pad(coordinates,((0,0),(0,2-coordinates.shape[1])))

def calculate(context,p):
    axes,rows,masks,sources=profiles(context,p)
    if len(axes)<2: raise ValueError("Choose at least two captured heads")
    distance=np.sqrt(divergences(rows))
    linkage,cuts=ward(distance)
    candidates=list(range(2,min(p.max_clusters,len(axes)-1)+1)) or [2]
    scores={k:silhouette(distance,cuts[k]) for k in candidates}
    best=min(candidates,key=lambda k:(-scores[k],k))
    labels=cuts[best]; xy=projection(distance)
    names=[f"L{l} · H{h}" for l,h in axes]
    return Calculation(source_indices=sources,
        statistics=[statistic("Selected clusters",best,"clusters"),statistic("Mean silhouette",scores[best]),
                    statistic("Sampled sources",len(sources),"sources"),statistic("Compared heads",len(axes),"heads")],
        table=table("Captured head clusters",["Cluster","MDS x","MDS y"],names,[[labels[i],*xy[i].tolist()] for i in range(len(axes))],"cluster ID; projected coordinates"),
        tables=[table("Silhouette by cluster count",["Silhouette"],[str(k) for k in candidates],[[scores[k]] for k in candidates]),
                table("Ward merge tree",["Left node","Right node","Ward distance","Members"],[str(len(axes)+i) for i in range(len(linkage))],linkage,"node IDs; distance; head count")],
        plots=[ResearchPlot(kind="projection",title="Captured attention heads · classical MDS",x_label="MDS coordinate 1",y_label="MDS coordinate 2",
            points=[ResearchPoint(label=names[i],x=float(xy[i,0]),y=float(xy[i,1]),group=f"Cluster {labels[i]}") for i in range(len(axes))])],
        normalization="each sampled row normalized by its captured row sum; distance is sqrt(mean source-wise base-2 Jensen–Shannon divergence)",
        notices=["Ward clusters and pattern names describe this capture and sampled source population; stable specialization is not established.",
                 "Sources are sampled deterministically, retaining non-image rows first; head population is bounded to 64."])

DEFINITION=define("heads.clustering","Head Clustering","head_and_representation","head_clusters",
    "Ward-cluster selected heads using source-averaged Jensen–Shannon distance; choose cluster count by silhouette and display a classical-MDS projection.",
    Parameters,calculate,requires=["attention_weights","multiple_source_rows","complete_declared_support"],displays=["head_grid","projection"],
    limitations=["At most 64 head profiles and 256 source rows; labels are exploratory, not a learned corpus taxonomy."],
    references=[CLARK_2019,LIN_1991_JSD,WARD_1963,ROUSSEEUW_1987],
    literature=("adaptation","Builds on Clark et al. (2019), who embed heads with MDS from Jensen-Shannon divergences summed over data. Adds bounded source sampling, square-root mean JSD, Ward (1963) clustering, silhouette (Rousseeuw 1987) selection and classical MDS."))
