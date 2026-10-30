"""Explicitly inferred page-local heading scope; no parentage inferred from depth alone."""
from .contracts import RowParameters, Calculation, ResearchEdge
from ._document import masses, document, table, define


class Parameters(RowParameters):
    pass


def compute(context, parameters):
    row, measured, known, _ = masses(context, parameters)
    sections={}; parents={}; unassigned=[]
    measured_pages={known[i].page_id for i in measured}
    for page in sorted((p for p in document(context).pages if p.page_id in measured_pages),key=lambda p:p.page_number):
        stack=[]
        for e in sorted((e for e in known.values() if e.page_id==page.page_id and e.reading_order is not None),key=lambda e:(e.reading_order,e.element_id)):
            if e.element_type.casefold() in ("heading","title"):
                level=e.level if e.level is not None else 0
                while stack and stack[-1][1]>=level: stack.pop()
                parents[e.element_id]=stack[-1][0] if stack else None
                sections[e.element_id]=[]; stack.append((e.element_id,level))
            if e.element_id in measured:
                if stack: sections[stack[-1][0]].append(e.element_id)
                else: unassigned.append(e.element_id)
    # A scope with no measured element is unavailable, not a measured zero.
    direct={i:sum(measured[j] for j in owned) if owned else None for i,owned in sections.items()}
    rolled=dict(direct)
    for i in reversed(list(sections)):
        if parents[i] and rolled[i] is not None: rolled[parents[i]]=(rolled[parents[i]] or 0.0)+rolled[i]
    ids=list(sections)
    assigned={j for owned in sections.values() for j in owned}
    outside=[i for i in measured if i not in assigned]
    notices=["Heuristically inferred page-local scope: reading order plus heading level; not dataset-provided or researcher-reviewed relationships.",
             "Direct masses are disjoint. Inclusive masses overlap across ancestor rows and must not be summed."]
    if not ids:
        notices.insert(0,"No heading or title with an annotated reading order was found on the measured pages, so no section "
                         "scope can be inferred and every measured element is reported under Outside inferred sections.")
    return Calculation(row=row,table=table("Inferred section roll-up",["Direct mass","Including descendants"],
        [f"{known[i].element_type} · {known[i].native_id}" for i in ids],[[direct[i],rolled[i]] for i in ids],"mass",ids,[known[i].page_id for i in ids]),
        tables=[table("Outside inferred sections",["Mass"],["Unassigned or unordered elements","Residuals"],
            [[sum(measured[i] for i in outside) if outside else None],[row.total_mass-sum(measured.values())]])],
        edges=[ResearchEdge(source=p,target=i,value=rolled[i],signal="section",units="descendant mass") for i,p in parents.items() if p and rolled[i] is not None],
        notices=notices)


DEFINITION=define("attention.section_rollup","Section roll-up · inferred scope","Aggregate elements under page-local heading scopes inferred by annotated order and heading level.",Parameters,compute,
    requires=None,limitations=["Page-local inference only; cross-page scope and reviewed section relations are not implied."])
