"""Deterministic OCR/structure serialization and token-span ownership without evaluation labels."""
import json


def serialize(page,elements,representation):
    if representation not in ("ocr","structured"): raise ValueError("Unknown textual document representation")
    text=f"Document page {page.page_number}:\n"; spans=[]
    for e in sorted(elements,key=lambda e:(e.reading_order is None,e.reading_order or 0,e.element_id)):
        if not e.ocr.text: continue
        value=e.ocr.text
        if representation=="structured":
            value=json.dumps({"element":e.element_id,"type":e.element_type,"order":e.reading_order,"text":value},ensure_ascii=False,sort_keys=True)
        start=len(text); text+=value; spans.append((start,len(text),e.element_id,page.page_id)); text+="\n"
    return text,spans


def text_ownership(offset,spans):
    start,end=offset
    if end<=start: return {},None
    owned={}; pages=set()
    for a,b,element,page in spans:
        overlap=max(0,min(end,b)-max(start,a))
        if overlap:
            owned[element]=owned.get(element,0)+overlap/(end-start); pages.add(page)
    if len(pages)>1: raise ValueError("A textual model position straddles document-page ownership")
    total=sum(owned.values())
    if total>1+1e-8: raise ValueError("Serialized element spans overlap")
    if owned and total<1: owned["non_image:document_text"]=1-total
    return owned,next(iter(pages)) if pages else None
