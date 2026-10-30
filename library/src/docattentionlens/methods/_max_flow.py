"""Float-capacity Dinic solver for explicitly bounded layered graphs."""
from collections import deque

class FlowNetwork:
    """Vertex index and arc lists built once; every solve starts from fresh residual capacities.

    Arc 2k is edge k and arc 2k+1 its reverse. Arcs join each vertex's list in edge order,
    exactly as a freshly built graph would, so repeated solves give identical flows.
    """
    def __init__(self,edges):
        vertices=set()
        for a,b,_ in edges: vertices.update((a,b))
        self.index={v:i for i,v in enumerate(sorted(vertices))}
        self.arcs=[[] for _ in self.index]
        self.capacity=[]
        for a,b,value in edges:
            u,v=self.index[a],self.index[b]
            self.arcs[u].append((len(self.capacity),v)); self.capacity.append(float(value))
            self.arcs[v].append((len(self.capacity),u)); self.capacity.append(0.0)

    def maximum_flow(self,source,target):
        if source not in self.index or target not in self.index: return 0.0
        arcs=self.arcs; capacity=self.capacity.copy(); count=len(arcs)
        s,t=self.index[source],self.index[target]; total=0.0
        while True:
            levels=[-1]*count; levels[s]=0; queue=deque([s])
            while queue:
                u=queue.popleft()
                for arc,v in arcs[u]:
                    if levels[v]<0 and capacity[arc]>1e-12: levels[v]=levels[u]+1; queue.append(v)
            if levels[t]<0: return total
            pointers=[0]*count
            def push(u,value):
                if u==t: return value
                out=arcs[u]
                while pointers[u]<len(out):
                    arc,v=out[pointers[u]]
                    if capacity[arc]>1e-12 and levels[v]==levels[u]+1:
                        sent=push(v,min(value,capacity[arc]))
                        if sent>1e-12:
                            capacity[arc]-=sent; capacity[arc^1]+=sent
                            return sent
                    pointers[u]+=1
                return 0.0
            while (sent:=push(s,float("inf")))>1e-12: total+=sent

def maximum_flow(edges,source,target):
    return FlowNetwork(edges).maximum_flow(source,target)
