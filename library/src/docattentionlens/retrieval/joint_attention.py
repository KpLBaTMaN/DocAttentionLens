"""Diagnostic page-plus-question encodings; never used for actual retrieval scores.

The question replaces the page prompt's "Describe the image." caption. The answer variant
appends "\\nAnswer: {reference answer}" to that text: the reference answer is a case annotation,
so the encoding records its source and the exact text.
"""
from .contracts import Position
from .encoding_capture import capture_encoding


ANSWER_SEPARATOR = "\nAnswer: "


def capture_joint(torch,model,processor,image,query,positions,ownership,arrays,page_id,causal,float32_attention=False,*,answer=None,encoding_id="joint"):
    original=processor.visual_prompt_prefix
    phrase="Describe the image."
    marker="___DOCATTENTIONLENS_DIAGNOSTIC_QUERY___"
    answer=None if answer is None else answer.strip()
    if answer == "":
        raise ValueError("The case has no reference answer to add to the diagnostic prompt")
    text=query if answer is None else query+ANSWER_SEPARATOR+answer
    if original.count(phrase)!=1 or marker in original or marker in text:
        raise ValueError("The pinned processor's diagnostic prompt template differs")
    template=original.replace(phrase,marker)
    try:
        processor.visual_prompt_prefix=original.replace(phrase,text)
        batch=processor(images=[image],return_tensors="pt")
    finally:
        processor.visual_prompt_prefix=original
    visual=[p for p in positions if p.role=="visual_patch"]
    if causal:
        formatted=template.replace(processor.image_token,processor.image_token*len(visual))
    else:
        helper=type(processor).__call__.__globals__["build_string_from_input"]
        formatted=helper(prompt=template,bos_token=processor.tokenizer.bos_token,image_seq_len=len(visual),image_token="<image>",num_images=1)
    start=formatted.index(marker); formatted=formatted.replace(marker,text); end=start+len(query)
    # The answer span, when present, follows the question and the "\nAnswer: " separator.
    answer_start=end+len(ANSWER_SEPARATOR); answer_end=start+len(text)
    offsets=processor.tokenizer(formatted,return_offsets_mapping=True,add_special_tokens=False)
    tokens=batch["input_ids"][0].tolist(); mask=batch["attention_mask"][0].tolist()
    if offsets["input_ids"]!=tokens:
        raise ValueError("Diagnostic joint query offsets differ from the actual input IDs")
    image_token=model.config.vlm_config.image_token_id
    mapped=[]; new_ownership=[]; cursor=0
    for i,(token,active,span) in enumerate(zip(tokens,mask,offsets["offset_mapping"],strict=True)):
        bbox=None; bucket={"non_page":1.0}
        if token==image_token:
            if cursor>=len(visual): raise ValueError("Diagnostic image positions differ")
            p=visual[cursor]; cursor+=1; bbox=p.bbox; bucket=ownership[p.index]; role="visual_patch"
        else:
            role=("query_text" if span[1]>start and span[0]<end
                  else "answer_text" if answer is not None and span[1]>answer_start and span[0]<answer_end else "visual_prompt")
        mapped.append(Position(index=i,token_id=token,piece=processor.tokenizer.convert_ids_to_tokens(token),role=role,eligible=bool(active),bbox=bbox))
        new_ownership.append(bucket)
    if cursor!=len(visual) or not any(p.role=="query_text" for p in mapped):
        raise ValueError("Diagnostic joint image/query alignment differs")
    if answer is not None and not any(p.role=="answer_text" for p in mapped):
        raise ValueError("Diagnostic joint answer tokens are missing from the actual input IDs")
    _embeddings,encoding,error=capture_encoding(torch,model,batch.to("cuda"),encoding_id,"joint",arrays,page_id,causal=causal,float32_attention=float32_attention)
    return encoding.model_copy(update={"positions":mapped,"ownership":new_ownership,"prompt_text":original.replace(phrase,text),
        "answer_source":None if answer is None else "reference","answer_text":answer}),error
