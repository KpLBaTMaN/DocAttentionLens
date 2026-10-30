"""Pinned Qwen-VL generator adapters (Qwen2-VL, Qwen2.5-VL, Qwen3-VL).

One capture procedure serves every profile in `contracts.GENERATORS`; the profile
supplies the checkpoint, patch geometry and model-specific runtime checks. Only
optional worker installations import Torch.
"""
from datetime import UTC, datetime
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from uuid import uuid4
import gc
import platform
import numpy as np
from PIL import Image
from docattentionlens import __version__
from docattentionlens.retrieval.scoring import ownership
from . import full_attention as triangle
from contextlib import contextmanager
from .contracts import (QWEN25_VL_3B, MAX_INPUT_POSITIONS, MAX_NEW_TOKENS, CAPTURE_STEPS, SYSTEM_PROMPT, ContextManifest,
    ContextPage, FullAttention, GenerationPosition, GenerationCapture, PredictionStep, SourceRow, QUESTION_ROWS_LIMIT,
    STANDARD_IMAGE_POSITIONS)

_runtime = None
_checkpoint_files = {}
_runtime_key = None


def release_runtime():
    global _runtime, _runtime_key
    _runtime = None
    _runtime_key = None
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def checkpoint_hashes(folder):
    files = {}
    for path in sorted(folder.iterdir()):
        if path.is_file():
            digest = sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024), b""):
                    digest.update(chunk)
            files[path.name] = digest.hexdigest()
    return files


def runtime(cache, checkpoint=None, profile=QWEN25_VL_3B):
    global _runtime, _checkpoint_files, _runtime_key
    key = checkpoint["registration_id"] if checkpoint else profile.registration_id
    if _runtime is not None and _runtime_key != key:
        release_runtime()
    if _runtime is None:
        from docattentionlens.retrieval.colpali import release_runtime as release_colpali
        release_colpali()
        from docattentionlens.retrieval.colqwen import release_runtime as release_colqwen
        release_colqwen()
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen2TokenizerFast
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info()[0] < profile.required_free_gib*1024**3:
            raise RuntimeError(f"Generator requires a BF16-capable CUDA device with {profile.required_free_gib} GiB free")
        registered_files = checkpoint_hashes(Path(checkpoint["path"])) if checkpoint else None
        if checkpoint and registered_files != checkpoint["files"]:
            raise RuntimeError("Checkpoint bytes differ from the registered manifest")
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        options = dict(revision=profile.revision, cache_dir=str(cache / "hub")) if not checkpoint else dict(local_files_only=True)
        source = profile.model_id if not checkpoint else checkpoint["path"]
        # The slow image processor is the validated PIL resize for every profile, including
        # Qwen3-VL whose checkpoint names the fast variant; pack_pixels checks it below.
        processor = AutoProcessor.from_pretrained(source, use_fast=False, **options)
        processor.tokenizer = Qwen2TokenizerFast.from_pretrained(source, **options)
        ip = processor.image_processor
        ip.size = {"shortest_edge": profile.min_pixels, "longest_edge": profile.max_pixels}
        ip.min_pixels, ip.max_pixels = profile.min_pixels, profile.max_pixels
        if type(ip).__name__ != "Qwen2VLImageProcessor" or (ip.patch_size, ip.merge_size, ip.temporal_patch_size) != (profile.patch_size, profile.merge_size, profile.temporal_patch_size) or not ip.do_resize or not ip.do_rescale or not ip.do_normalize:
            raise RuntimeError("Processor differs from the validated still-image mapping")
        model, loading = getattr(transformers, profile.model_class).from_pretrained(source,
            dtype=torch.bfloat16, attn_implementation={"text_config":"eager", "vision_config":"sdpa"}, output_loading_info=True, **options)
        if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError("Checkpoint weight loading is incomplete or incompatible")
        model = model.to("cuda").eval()
        text = model.config.text_config
        if (text._attn_implementation != "eager" or model.config.vision_config._attn_implementation != "sdpa" or getattr(text, "use_sliding_window", False)
                or (text.num_hidden_layers, text.num_attention_heads, text.num_key_value_heads) != (profile.layers, profile.query_heads, profile.kv_heads)):
            raise RuntimeError("Expected causal language eager attention and vision SDPA")
        # Check actual preprocessing against an independent PIL resize and patch
        # grouping. Arbitrary colours across patch boundaries catch transposition.
        fixture = np.random.default_rng(17).integers(0,256,(112,168,3),dtype=np.uint8)
        actual = ip(images=Image.fromarray(fixture), return_tensors="np")
        expected = pack_pixels(Image.fromarray(fixture), actual["image_grid_thw"][0], ip)
        if not np.allclose(actual["pixel_values"],expected,atol=1e-6,rtol=0):
            raise RuntimeError("Processor resize/temporal/grouping fixture failed")
        from huggingface_hub import snapshot_download
        folder = Path(checkpoint["path"]) if checkpoint else Path(snapshot_download(profile.model_id, local_files_only=True, **options))
        _checkpoint_files = registered_files if checkpoint else checkpoint_hashes(folder)
        _runtime = (torch, processor, model)
        _runtime_key = key
    return _runtime


@contextmanager
def image_budget(ip, profile, image_positions):
    """Resize pages for at most `image_positions` merged positions while processing, then restore the plan.

    The runtime keeps one processor; the standard 512-position ceiling is its resting state.
    """
    resting = (dict(ip.size), ip.min_pixels, ip.max_pixels)
    ip.size = {"shortest_edge": profile.min_pixels, "longest_edge": profile.pixels_for(image_positions)}
    ip.min_pixels, ip.max_pixels = profile.min_pixels, profile.pixels_for(image_positions)
    try:
        yield dict(ip.size)
    finally:
        ip.size, ip.min_pixels, ip.max_pixels = resting


def pack_pixels(image, grid, ip):
    """Independent row-major grouping of resized RGB pixels into merged cells."""
    t,h,w = (int(x) for x in grid)
    p = int(ip.patch_size)
    if t != 1 or h % 2 or w % 2:
        raise ValueError("Only still images and 2x2 spatial merging are supported")
    rgb = np.asarray(image.resize((w*p,h*p),resample=ip.resample),dtype=np.float32)
    rgb = (rgb*ip.rescale_factor-np.asarray(ip.image_mean,dtype=np.float32))/np.asarray(ip.image_std,dtype=np.float32)
    patches=[]
    for row in range(0,h,2):
        for col in range(0,w,2):
            for dr in range(2):
                for dc in range(2):
                    patch = rgb[(row+dr)*p:(row+dr+1)*p,(col+dc)*p:(col+dc+1)*p].transpose(2,0,1)
                    patches.append(np.stack([patch,patch],axis=1).reshape(-1))
    return np.asarray(patches,dtype=np.float32)


def npy_bytes(values) -> int:
    """Exact size of `values` saved with np.save (the store's on-disk bytes)."""
    header = BytesIO()
    np.lib.format.write_array_header_1_0(header, np.lib.format.header_data_from_array_1_0(values))
    return header.tell() + values.nbytes


def pack_prefill(attentions, bfloat16, heads, n):
    """Pack each layer's dense (1, heads, n, n) BF16 prefill attention as its lossless causal triangle.

    Layers cross to the host one at a time, so host memory holds one dense square. Returns the
    packed arrays and the largest row-mass deviation over every prefill row.
    """
    packed = {}; mass_error = 0.0
    for layer, matrix in enumerate(attentions):
        if matrix.dtype != bfloat16 or tuple(matrix.shape) != (1, heads, n, n):
            raise ValueError("Full prefill attention is not a BF16 (1, heads, n, n) matrix")
        square = matrix[0].cpu().float().numpy()
        if not np.isfinite(square).all() or np.any(square < 0):
            raise ValueError("Full prefill attention has non-finite or negative weights")
        mass_error = max(mass_error, float(np.abs(square.sum(axis=-1, dtype=np.float64) - 1).max()))
        packed[triangle.array_name(layer)] = triangle.pack(square)
        del square
    return packed, mass_error


def verify_full_attention(arrays, n, layers, question):
    """Every layer's unpacked triangle must reproduce the separately copied step-0 and question rows bit for bit."""
    names = [triangle.array_name(layer) for layer in range(layers)]
    if sorted(name for name in arrays if name.startswith("full_l")) != sorted(names):
        raise ValueError("Full attention layers differ from the decoder layers")
    def bits(values):
        return np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    for layer, name in enumerate(names):
        square = triangle.unpack(arrays[name], n)
        checks = [(square[:, n-1, :], arrays["step_0"][layer])]
        checks += [(square[:, i, :i+1], arrays[f"question_{i}"][layer]) for i in question]
        for unpacked, saved in checks:
            if unpacked.shape != saved.shape or not np.array_equal(bits(unpacked), bits(saved)):
                raise ValueError("Full attention triangle differs from the retained prefill rows")
        del square


def retained_step_limit(full_attention):
    """Prediction steps whose logits and rows a capture keeps: the first CAPTURE_STEPS, or with full
    attention every generated step (generation stops at MAX_NEW_TOKENS)."""
    return MAX_NEW_TOKENS if full_attention else CAPTURE_STEPS


def check_retained_steps(full_attention, output_count, plain_logits, sampled_logits, copied):
    """Both generate passes sampled logits for the same retained steps, and the capture copied each step's row.

    Returns the largest absolute logit difference over every retained step; above 1e-5 the capture fails.
    """
    expected = min(retained_step_limit(full_attention), output_count)
    if len(copied) != expected or len(plain_logits) != expected or len(sampled_logits) != expected:
        raise ValueError("Capture-on/off token or coverage equivalence failed")
    error = max(float(np.abs(a - b).max()) for a, b in zip(plain_logits, sampled_logits, strict=True))
    if error > 1e-5:
        raise ValueError("Capture-on/off logits equivalence failed")
    return error


def verify_decode_rows(arrays, steps, n, layers, heads, output_count):
    """A full-attention capture keeps every prediction step, so its squares reach the last consumed position.

    Step k is the row of position n+k-1 over its n+k causal targets (KV-cached decoding has no future
    keys); each row must be exact BF16 as the eager decoder computed it, finite, non-negative and
    normalized within 0.02. Step 0 is the prefill's last row, checked bit for bit by verify_full_attention.
    """
    if [s.step for s in steps] != list(range(output_count)):
        raise ValueError("Full attention capture is missing prediction steps")
    for k, s in enumerate(steps):
        values = arrays.get(s.array)
        if (s.array != f"step_{k}" or s.source_index != n + k - 1 or s.target_count != n + k or values is None
                or values.dtype != np.float32 or values.shape != (layers, heads, n + k)):
            raise ValueError("Retained decode row differs from the full-attention axes")
        if (not np.isfinite(values).all() or np.any(values < 0)
                or float(np.abs(values.sum(axis=-1, dtype=np.float64) - 1).max()) > .02):
            raise ValueError("Retained decode row differs from normalized causal attention")
        triangle.bf16_bits(values)


def full_attention_record(arrays, n, layers, heads, step_count):
    names = [triangle.array_name(layer) for layer in range(layers)]
    return FullAttention(layout=triangle.LAYOUT, position_count=n, square_limit=n+max(0, step_count-1), layer_arrays=names,
        query_head_count=heads, bytes=sum(npy_bytes(arrays[name]) for name in names))


def capture(case, retrieval, inputs, cache, interrupted=lambda:False, *, checkpoint=None, system_prompt=SYSTEM_PROMPT, profile=QWEN25_VL_3B, representation="image", steering=None, intervention=None, full_attention=False, image_token_limit=STANDARD_IMAGE_POSITIONS):
    """Free-running greedy generation, with a plain run on identical inputs.

    QA labels and OCR never enter prompt construction. Hooks copy the last
    consumed row and discard dense matrices before generate can retain them.
    Both runs use the same eager language kernel; equivalence is not a claim
    about other kernels or output correctness. `profile` selects the pinned
    checkpoint; an uploaded `checkpoint` must match that profile's architecture.
    With `full_attention`, the capture run also keeps every layer's complete
    prefill matrix as its lossless BF16 causal triangle (`full_l{layer}`), and
    both runs retain every generated step (logits and, in the capture run, the
    decode row) instead of the first CAPTURE_STEPS.
    `image_token_limit` above 512 resizes a single supplied page for more merged positions.
    """
    selected_ids=[page.page_id for page,_raw,_elements in inputs]
    profile.pixels_for(image_token_limit)
    if image_token_limit!=STANDARD_IMAGE_POSITIONS and (len(inputs)!=1 or representation!="image"):
        raise ValueError("The 1,024-position plan needs exactly one supplied page image")
    candidates = {p.page_id for p in retrieval.pages} if retrieval else set(case.page_ids)
    if (retrieval and (retrieval.case_id != case.case_id or retrieval.query != case.query)) or not 1 <= len(inputs) <= 2 or len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= candidates:
        raise ValueError("Generator context differs from frozen case, retrieval lineage or page selection")
    for page, _raw, elements in inputs:
        candidate = next((p for p in retrieval.pages if p.page_id == page.page_id), None) if retrieval else None
        if (candidate and candidate.image_sha256 != page.image_sha256) or any(e.page_id != page.page_id for e in elements):
            raise ValueError("Candidate hash or element page identity differs")
    torch, processor, model = runtime(cache, checkpoint, profile)
    merged = profile.merge_size**2
    if interrupted():
        raise RuntimeError("Acquisition interrupted")
    identifier = "generation_"+uuid4().hex
    images=[]
    for page, raw, _elements in inputs:
        if sha256(raw).hexdigest()!=case.page_image_sha256[page.page_id] or page.image_sha256!=case.page_image_sha256[page.page_id]:
            raise ValueError("Page differs from frozen case input")
        image=Image.open(BytesIO(raw)); image.load()
        if image.size!=(page.width,page.height) or image.getexif().get(274,1)!=1:
            raise ValueError("Unsupported source image dimensions or orientation")
        images.append(image.convert("RGB"))
    if representation not in ("image","ocr","structured"): raise ValueError("Unsupported document representation")
    content=[]; serialized=[]
    for page,_,elements in inputs:
        if representation=="image":
            content += [{"type":"text","text":f"Document page {page.page_number}:"},{"type":"image"}]
        else:
            from .document_inputs import serialize
            text,spans=serialize(page,elements,representation); serialized.append((text,spans))
            content.append({"type":"text","text":text})
    content += [{"type":"text","text":case.query}]
    messages=[{"role":"system","content":system_prompt},{"role":"user","content":content}]
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    with image_budget(processor.image_processor,profile,image_token_limit) as processor_size:
        processed=processor(text=[prompt], **({"images":images} if representation=="image" else {}), return_tensors="pt")
    tokens=processed.input_ids[0].tolist()
    mask=processed.attention_mask[0].tolist()
    n=len(tokens)
    if n>MAX_INPUT_POSITIONS or not all(mask):
        raise ValueError("Processed context exceeds plan or contains unsupported padding; no truncation applied")
    grid=processed.image_grid_thw.tolist() if representation=="image" else []
    if representation=="image" and len(grid)!=len(inputs):
        raise ValueError("Image count differs from selected context")
    expanded=prompt
    for g in grid:
        expanded=expanded.replace(processor.image_token,"<placeholder>"*(int(np.prod(g))//merged),1)
    expanded=expanded.replace("<placeholder>",processor.image_token)
    tokenized=processor.tokenizer(expanded,add_special_tokens=False,return_offsets_mapping=True)
    if tokenized.input_ids!=tokens:
        raise ValueError("Expanded prompt offsets do not match actual model input")
    query_start=expanded.rfind(case.query)
    if query_start<0:
        raise ValueError("Frozen query is absent from actual prompt")
    special=set(processor.tokenizer.all_special_ids)
    text_spans=[]; text_cursor=0
    for text,spans in serialized:
        start=expanded.find(text,text_cursor)
        if start<0: raise ValueError("Serialized document text is absent from actual input")
        text_spans.extend((start+a,start+b,e,p) for a,b,e,p in spans); text_cursor=start+len(text)
    positions=[]
    for i,(token,offset) in enumerate(zip(tokens,tokenized.offset_mapping,strict=True)):
        role="special" if token in special else "query" if offset[1]>query_start and offset[0]<query_start+len(case.query) else "prompt"
        from .document_inputs import text_ownership
        owned,pid=text_ownership(offset,text_spans)
        positions.append(GenerationPosition(index=i,token_id=token,piece=processor.tokenizer.convert_ids_to_tokens(token),role="document_text" if owned else role,page_id=pid,ownership=owned or {"non_image:"+role:1.0}))
    image_indices=[i for i,token in enumerate(tokens) if token==model.config.image_token_id]
    if len(image_indices)!=sum(int(np.prod(g))//merged for g in grid):
        raise ValueError("Visual token count differs from processor grid")
    arrays={}; pages=[]; cursor=0; pixel_cursor=0; processor_error=0.0
    # Save the exact BF16 values consumed by the vision encoder, cast losslessly
    # to FP32 NPY. Inputs contain no crops, padding or tiles.
    # Check geometry on the processor's float32 output: casting both sides to BF16 first lets a
    # one-ulp float32 difference straddle a BF16 rounding boundary (seen with Qwen3-VL's 0.5
    # mean/std), which is a rounding artefact rather than a mapping error.
    pixels32=processed.pixel_values.float().numpy() if representation=="image" else None
    processed=processed.to("cuda")
    if representation=="image": processed["pixel_values"]=processed.pixel_values.to(torch.bfloat16)
    for index,((page,_raw,elements),image,g) in enumerate(zip(inputs if representation=="image" else [],images if representation=="image" else [],grid,strict=True)):
        count=int(np.prod(g))//merged
        indices=image_indices[cursor:cursor+count]; cursor+=count
        if indices!=list(range(indices[0],indices[0]+count)):
            raise ValueError("Image token span is not contiguous")
        h,w=g[1]//profile.merge_size,g[2]//profile.merge_size
        for k,i in enumerate(indices):
            r,c=divmod(k,w)
            box=[c*page.width/w,r*page.height/h,(c+1)*page.width/w,(r+1)*page.height/h]
            positions[i]=GenerationPosition(index=i,token_id=tokens[i],piece=positions[i].piece,role="image",page_id=page.page_id,bbox=box,ownership=ownership(box,elements))
        pixel_count=int(np.prod(g))
        actual=processed.pixel_values[pixel_cursor:pixel_cursor+pixel_count].float().cpu().numpy()
        expected=pack_pixels(image,g,processor.image_processor)
        processor_error=max(processor_error,float(np.abs(pixels32[pixel_cursor:pixel_cursor+pixel_count]-expected).max())); pixel_cursor+=pixel_count
        if processor_error>1e-5:
            raise ValueError("Actual page pixel mapping differs from independent reconstruction")
        name=f"pixels_{index}"; arrays[name]=actual
        pages.append(ContextPage(page_id=page.page_id,page_number=page.page_number,image_sha256=page.image_sha256,
            source_width=page.width,source_height=page.height,resized_width=g[2]*profile.patch_size,resized_height=g[1]*profile.patch_size,
            grid_thw=g,position_indices=indices,pixel_array=name,preview_url=f"/api/v1/generation/captures/{identifier}/pages/{page.page_id}/input-image"))
    if representation!="image":
        for page,_raw,_elements in inputs:
            indices=[p.index for p in positions if p.page_id==page.page_id]
            pages.append(ContextPage(page_id=page.page_id,page_number=page.page_number,image_sha256=page.image_sha256,
                source_width=page.width,source_height=page.height,resized_width=0,resized_height=0,grid_thw=[],position_indices=indices,pixel_array=None,preview_url=page.image_url))
    context=ContextManifest(construction_version="selected_page_images_v1" if representation=="image" else "selected_document_representation_v2",representation=representation,ocr_consumed=representation!="image",
        retrieved_page_ids=[p.page_id for p in retrieval.pages] if retrieval else [],selected_page_ids=[p.page_id for p in pages],
        omitted_page_ids=[p.page_id for p in retrieval.pages if p.page_id not in {p.page_id for p in pages}] if retrieval else [],query=case.query,prompt=expanded,
        pages=pages,input_token_ids=tokens,input_attention_mask=mask,system_prompt=system_prompt,input_source="retrieved_pages" if retrieval else "supplied_pages",
        image_token_limit=image_token_limit)
    # Question tokens follow every page image, so their prefill rows attend to the patches.
    question=[p.index for p in positions if p.role=="query"]
    if not question:
        raise ValueError("Frozen query positions are absent from the actual prompt")
    question_truncated=len(question)>QUESTION_ROWS_LIMIT
    question=question[:QUESTION_ROWS_LIMIT]
    from transformers import StoppingCriteria, StoppingCriteriaList
    class Interrupted(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):
            return interrupted()
    options=dict(max_new_tokens=MAX_NEW_TOKENS,do_sample=False,num_beams=1,use_cache=True,
        return_dict_in_generate=True,stopping_criteria=StoppingCriteriaList([Interrupted()]))
    question_rows=[]; sampled_logits=[]; copied=[]; observed_sources=[]; observed_positions=[]; prefill_positions=[]; phase=["plain"]; step=[0]
    full_arrays={}; full_mass_error=[0.0]
    # Both runs sample logits for the same steps; the capture run also copies each of those rows.
    limit=retained_step_limit(full_attention)
    merger_output=[None]; vision_order=[representation!="image"]
    def merger_hook(_module,_args,output):
        if not vision_order[0]:
            merger_output[0]=output
    def vision_hook(module,args,kwargs,output):
        if not vision_order[0]:
            # Qwen3-VL also returns its DeepStack features; the merged tokens come first.
            merged_tokens=output[0] if isinstance(output,tuple) else output
            if profile.window_permutation:
                order,_=module.get_window_index(kwargs["grid_thw"])
                if not torch.equal(torch.sort(order).values,torch.arange(len(order),device=order.device)) or not torch.equal(merged_tokens,merger_output[0][torch.argsort(order)]):
                    raise RuntimeError("Runtime vision merge/window inversion differs from row-major mapping")
            elif not torch.equal(merged_tokens,merger_output[0]):
                # Without window permutation the merger output is inserted unchanged, in row-major order.
                raise RuntimeError("Runtime vision merge/window inversion differs from row-major mapping")
            vision_order[0]=True; merger_output[0]=None
    def hook(_module,args,kwargs,output):
        k=step[0]; step[0]+=1
        if k<limit:
            sampled_logits.append(output.logits[0,-1].float().cpu().numpy().copy())
            if phase[0]=="capture":
                observed_sources.append(int(kwargs["input_ids"][0,-1]))
                # Qwen2/2.5-VL receive text + multimodal RoPE rows; Qwen3-VL derives its RoPE positions
                # inside the model, so the cache position alone aligns the source.
                position_ids=kwargs.get("position_ids")
                positional=position_ids[:,0,:].cpu().tolist() if position_ids is not None else None
                cache_positions=kwargs["cache_position"].cpu().tolist()
                if cache_positions[-1]!=n+k-1 or (positional is not None and (len(positional)!=4 or positional[0][-1]!=n+k-1)):
                    raise ValueError("Actual text/cache position differs from prediction source")
                if k==0:
                    if cache_positions!=list(range(n)) or (positional is not None and positional[0]!=list(range(n))):
                        raise ValueError("Unsupported positional padding or text ordering")
                    if positional is not None:
                        prefill_positions.extend(positional)
                observed_positions.append({"step":k,"text_temporal_height_width":[axis[-1] for axis in positional] if positional is not None else None,"cache_position":cache_positions[-1]})
                if output.attentions is None or len(output.attentions)!=model.config.text_config.num_hidden_layers:
                    raise RuntimeError("Language attention weights unavailable")
                row=torch.stack([a[0,:,-1,:] for a in output.attentions]).float().cpu().numpy()
                if row.shape!=(model.config.text_config.num_hidden_layers,model.config.text_config.num_attention_heads,n+k) or not np.isfinite(row).all() or np.any(row<0):
                    raise ValueError("Captured causal row has incompatible axes")
                copied.append(row)
                if k==0:
                    # Copy only question source rows from the dense prefill matrices.
                    rows=torch.stack([a[0,:,question,:] for a in output.attentions]).float().cpu().numpy()
                    if rows.shape!=(model.config.text_config.num_hidden_layers,model.config.text_config.num_attention_heads,len(question),n) or not np.isfinite(rows).all() or np.any(rows<0):
                        raise ValueError("Captured question rows have incompatible axes")
                    question_rows.append(rows)
                    if full_attention:
                        # Keep the complete dense prefill matrices, packed losslessly one layer at a time.
                        packed,mass=pack_prefill(output.attentions,torch.bfloat16,model.config.text_config.num_attention_heads,n)
                        full_arrays.update(packed); full_mass_error[0]=mass
        # Generation needs logits and KV cache, not retained dense attention.
        output.attentions=None
    hooks=[model.register_forward_hook(hook,with_kwargs=True),
        model.model.visual.merger.register_forward_hook(merger_hook),
        model.model.visual.register_forward_hook(vision_hook,with_kwargs=True)]
    torch.cuda.reset_peak_memory_stats()
    try:
        from .steering import apply_steering
        from ..attention_interventions import apply as apply_attention_intervention
        with torch.inference_mode(), apply_steering(model,torch,steering,positions), apply_attention_intervention(model,torch,intervention,positions):
            model.model.rope_deltas=None
            plain=model.generate(**processed,output_attentions=False,**options)
            plain_ids=plain.sequences[0,n:].tolist(); del plain
            plain_logits=sampled_logits.copy(); sampled_logits.clear()
            if interrupted():
                raise RuntimeError("Acquisition interrupted")
            phase[0]="capture"; step[0]=0; model.model.rope_deltas=None
            captured=model.generate(**processed,output_attentions=True,**options)
            output_ids=captured.sequences[0,n:].tolist(); del captured
            if not prefill_positions:
                # Qwen3-VL: record the multimodal RoPE positions its forward derives internally.
                rope,_=model.model.get_rope_index(processed.input_ids,processed.get("image_grid_thw"),None,processed.attention_mask)
                prefill_positions.extend(rope[:,0,:].cpu().tolist())
        if interrupted():
            raise RuntimeError("Acquisition interrupted")
    finally:
        for handle in hooks:
            handle.remove()
    if plain_ids!=output_ids or not vision_order[0] or len(copied)!=min(limit,len(output_ids)) or len(question_rows)!=1:
        raise ValueError("Capture-on/off token or coverage equivalence failed")
    if full_attention and (len(full_arrays)!=model.config.text_config.num_hidden_layers or full_mass_error[0]>.02):
        raise ValueError("Full prefill attention coverage or normalization failed")
    # Every retained step's logits (the first eight, or every generated step with full attention).
    logits_error=check_retained_steps(full_attention,len(output_ids),plain_logits,sampled_logits,copied)
    steps=[]; row_mass_error=0.0
    for k,token in enumerate(output_ids):
        piece=processor.tokenizer.convert_ids_to_tokens(token)
        positions.append(GenerationPosition(index=n+k,token_id=token,piece=piece,role="generated",ownership={"non_image:generated":1.0}))
        if k<len(copied):
            source=positions[n+k-1]
            if observed_sources[k]!=source.token_id:
                raise ValueError("Actual consumed token differs from prediction alignment")
            name=f"step_{k}"; arrays[name]=copied[k]
            row_mass_error=max(row_mass_error,float(np.abs(copied[k].sum(axis=-1,dtype=np.float64)-1).max()))
            steps.append(PredictionStep(step=k,pass_kind="prefill" if k==0 else "decode",source_index=n+k-1,
                consumed_token_id=source.token_id,consumed_piece=source.piece,predicted_index=n+k,
                predicted_token_id=token,predicted_piece=piece,target_count=n+k,array=name))
    if row_mass_error>.02:
        raise ValueError("Softmax normalization differs beyond BF16 tolerance")
    source_rows=[]; question_mass_error=0.0; question_future_mass=0.0
    for j,i in enumerate(question):
        values=np.ascontiguousarray(question_rows[0][:,:,j,:i+1])
        question_future_mass=max(question_future_mass,float(question_rows[0][:,:,j,i+1:].max()) if i+1<n else 0.0)
        question_mass_error=max(question_mass_error,float(np.abs(values.sum(axis=-1,dtype=np.float64)-1).max()))
        name=f"question_{i}"; arrays[name]=values
        source_rows.append(SourceRow(source_index=i,token_id=positions[i].token_id,piece=positions[i].piece,
            predicted_index=i+1,predicted_token_id=positions[i+1].token_id,predicted_piece=positions[i+1].piece,target_count=i+1,array=name))
    del question_rows
    if question_mass_error>.02 or question_future_mass>1e-6:
        raise ValueError("Question rows differ from normalized causal attention")
    full_record=None; full_coverage={}; full_provenance={}; full_validation={}; full_code={}
    if full_attention:
        layers=model.config.text_config.num_hidden_layers; heads=model.config.text_config.num_attention_heads
        arrays.update(full_arrays); full_arrays.clear()
        verify_full_attention(arrays,n,layers,question)
        verify_decode_rows(arrays,steps,n,layers,heads,len(output_ids))
        full_record=full_attention_record(arrays,n,layers,heads,len(steps))
        summary={"layout":full_record.layout,"position_count":n,"square_limit":full_record.square_limit,"bytes":full_record.bytes,
            "prediction_steps":"all","prediction_row_bytes":sum(npy_bytes(arrays[s.array]) for s in steps)}
        full_coverage={"full_attention":summary,"full_matrices_retained":True,"retained_prediction_steps":len(steps),
            "source_rows":"every prefill row in all layers/query heads (lossless BF16 causal triangle); last consumed position for every free-running prediction step (every generated token's row)",
            "uncaptured":"the final generated token's own row (it is never consumed); vision attention; pre-softmax scores"}
        full_provenance={"full_attention":{**summary,"requested":True,"storage":"raw BF16 bit patterns as uint16; causal lower triangle, rows in order; decode rows from every prediction step's array"}}
        full_validation={"full_attention_round_trip":"bit-exact for every layer: unpacked row n-1 equals step_0 and every unpacked question row equals its question array",
            "full_attention_max_mass_error":full_mass_error[0],
            "full_attention_decode_rows":"every generated step retained; step k is the row of position n+k-1 over n+k causal targets, exact BF16, finite, non-negative, mass within 0.02",
            "equivalence_scope":"identical eager decoder kernels and declared representation/intervention; every generated step's logit vector and complete greedy output token sequence"}
        full_code={"full_attention_packing":sha256(Path(__file__).with_name("full_attention.py").read_bytes()).hexdigest()}
    ip=processor.image_processor
    eos=model.generation_config.eos_token_id
    eos_set=set(eos if isinstance(eos,list) else [eos])
    modules=[import_module(type(item).__module__) for item in (model,processor,ip)]
    code_hashes={Path(m.__file__).name:sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules}
    text_axes=len(prefill_positions)==4
    insertion="vision window permutation inverted before language insertion" if profile.window_permutation else "merger output inserted in row-major order"
    deepstack={"visual_indexes":list(profile.deepstack_visual_indexes),"decoder_layers_after":list(range(len(profile.deepstack_visual_indexes))),
        "effect":"image-position hidden states gain these vision layers' features after the listed decoder layers, so keys/values at image positions in the following layers include them"} if profile.deepstack_visual_indexes else None
    return GenerationCapture(capture_id=identifier,case_id=case.case_id,retrieval_capture_id=retrieval.capture_id if retrieval else None,
        created_at=datetime.now(UTC).isoformat(),producer_version=__version__,dependencies=[case.case_id,*([retrieval.capture_id] if retrieval else []),*context.selected_page_ids],
        context=context,positions=positions,output_token_ids=output_ids,answer=processor.tokenizer.decode(output_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
        stop_reason="eos" if output_ids[-1] in eos_set else "token_limit",steps=steps,source_rows=source_rows,full_attention=full_record,
        layer_count=model.config.text_config.num_hidden_layers,query_head_count=model.config.text_config.num_attention_heads,kv_head_count=model.config.text_config.num_key_value_heads,
        provenance={"model_id":checkpoint["model_id"] if checkpoint else profile.model_id,"model_revision":checkpoint["revision"] if checkpoint else profile.revision,"model_registration_id":checkpoint["registration_id"] if checkpoint else profile.registration_id,"adapter_version":profile.adapter_version,"question_rows_version":profile.question_rows_version,"checkpoint_files":_checkpoint_files,
            "code_sha256":{**code_hashes,"adapter":sha256(Path(__file__).read_bytes()).hexdigest(),
                "document_serializer":sha256(Path(__file__).with_name("document_inputs.py").read_bytes()).hexdigest(),
                "steering_adapter":sha256(Path(__file__).with_name("steering.py").read_bytes()).hexdigest(),**full_code},
            "frameworks":{name:version(name) for name in ("torch","transformers","torchvision","numpy","Pillow")},"python":platform.python_version(),
            "gpu":torch.cuda.get_device_name(),"dtype":"bfloat16; lossless float32 storage","training":False,"dropout":0,"seed":0,"tf32":False,
            "language_attention":"eager","vision_attention":"sdpa; weights not captured" if representation=="image" else "not consumed; textual document representation","topology":f"causal language self-attention; GQA {profile.query_heads} query / {profile.kv_heads} KV heads"+("; merged images" if representation=="image" else "; serialized document text")+("; DeepStack visual features" if deepstack and representation=="image" else ""),
            **({"deepstack":deepstack} if deepstack else {}),
            "positional_configuration":{"rope_scaling":getattr(model.config.text_config,"rope_scaling",None),"rope_theta":getattr(model.config.text_config,"rope_theta",None),
                "prefill_axes":["text_causal","temporal","height","width"] if text_axes else ["temporal","height","width"],
                **({} if text_axes else {"text_positions":"cache positions 0..n-1; multimodal RoPE positions derived inside the model"}),"prefill_position_ids":prefill_positions,"prediction_source_position_ids":observed_positions,
                "rope_deltas":model.model.rope_deltas.cpu().tolist(),"query_head_to_kv_head":[i//model.model.language_model.layers[0].self_attn.num_key_value_groups for i in range(model.config.text_config.num_attention_heads)]},
            "image_processor":{"size":processor_size,"image_token_limit":image_token_limit,"mean":ip.image_mean,"std":ip.image_std,"rescale":ip.rescale_factor,"resample":int(ip.resample),
                "chain":f"RGB, aspect-preserving smart resize to multiples of {profile.cell} with at most {image_token_limit} merged positions per page, no crop/pad/tiles; {profile.patch_size}px patches, duplicated temporal frame, 2x2 merge, {insertion}"},
            "generation":{"max_new_tokens":MAX_NEW_TOKENS,"do_sample":False,"num_beams":1,"use_cache":True,"eos_token_ids":sorted(eos_set)},
            "document_representation":representation,"attention_intervention":steering,
            "peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated(),**full_provenance},
        coverage={"component":"language_self_attention","quantity":"post_softmax; dropout disabled","source_rows":"last consumed position for first eight free-running prediction steps; every question-token prefill row",
            "question_source_rows":len(source_rows),"question_rows_truncated":question_truncated,
            "captured_steps":len(steps),"generated_steps":len(output_ids),"all_layers":True,"all_query_heads":True,"all_causal_targets":True,
            "full_matrices_retained":False,"dense_prefill_acquired":True,"uncaptured":"prefill source rows other than question tokens; later output steps; vision attention; pre-softmax scores",
            "mapping":"row-major merged nominal image footprints" if representation=="image" else "deterministic serialized element character spans mapped to tokenizer offsets",
            "ownership":"exact cell partition for images; fractional character-span ownership for document text; residuals retained",
            "normalization":"original causal target domain; no image-only normalization",**full_coverage},
        validation={"status":"passed_for_this_capture","processor_fixture":"passed at runtime setup","actual_page_processor_max_error":processor_error if representation=="image" else None,"runtime_vision_window_inversion":("passed" if profile.window_permutation else "not applicable: no window permutation; row-major merger output verified") if representation=="image" else "not applicable: document text input; vision encoder not consumed",
            "consumed_predicted_alignment":"passed for every captured step","capture_on_off_token_equality":True,"capture_on_off_logits_max_error":logits_error,
            "prediction_logits_sha256":sha256(b"".join(a.tobytes() for a in sampled_logits)).hexdigest(),
            "equivalence_scope":"identical eager decoder kernels and declared representation/intervention; first eight logit vectors and complete greedy output token sequence",
            "max_attention_mass_error":row_mass_error,"question_rows_max_mass_error":question_mass_error,"question_rows_future_mass":question_future_mass,"attention_mass_tolerance":.02,"prediction_equivalence_tolerance":1e-5,
            "answer_correctness":"not evaluated","scope":"this pinned adapter/input; no general model-quality or causal-attribution claim",**full_validation}), arrays


def integrated_gradients(case, inputs, cache, interrupted=lambda:False, *, source, step, steps, baseline, system_prompt=SYSTEM_PROMPT, profile=QWEN25_VL_3B):
    """Integrated gradients of one captured prediction's token log-probability over consumed visual-token embeddings.

    Rebuilds the exact image inputs of `source` and checks their token IDs. The integration input is
    the vision encoder's output for each consumed visual token as it enters the language model; the
    baseline is the same encoder's output for a constant black or white image of the consumed size.
    The prompt and earlier output tokens stay fixed (teacher forcing). The pixel-space path through
    the vision tower oscillates too fast to integrate, and BF16 cannot resolve the cancelling terms,
    so weights are upcast exactly to float32 with gradient checkpointing; the float32 model is released
    afterwards so later generation reloads the validated BF16 runtime. The integration rule belongs to
    methods/integrated_gradients.py.
    """
    from docattentionlens.methods.integrated_gradients import integrate_with_check, token_totals
    if profile.deepstack_visual_indexes:
        raise ValueError("Integrated gradients does not support DeepStack generators: their visual features also enter later decoder layers")
    if [page.page_id for page,_raw,_elements in inputs]!=source.context.selected_page_ids or source.context.representation!="image":
        raise ValueError("Attribution inputs differ from the source capture's consumed page images")
    selected=next((s for s in source.steps if s.step==step),None)
    if selected is None or baseline not in ("black","white"):
        raise ValueError("Select a captured prediction step and a declared baseline")
    torch, processor, model = runtime(cache, None, profile)
    images=[]
    for page, raw, _elements in inputs:
        if sha256(raw).hexdigest()!=case.page_image_sha256[page.page_id] or page.image_sha256!=case.page_image_sha256[page.page_id]:
            raise ValueError("Page differs from frozen case input")
        image=Image.open(BytesIO(raw)); image.load()
        if image.size!=(page.width,page.height) or image.getexif().get(274,1)!=1:
            raise ValueError("Unsupported source image dimensions or orientation")
        images.append(image.convert("RGB"))
    content=[]
    for page,_raw,_elements in inputs:
        content += [{"type":"text","text":f"Document page {page.page_number}:"},{"type":"image"}]
    content += [{"type":"text","text":case.query}]
    prompt=processor.apply_chat_template([{"role":"system","content":system_prompt},{"role":"user","content":content}],tokenize=False,add_generation_prompt=True)
    with image_budget(processor.image_processor,profile,source.context.image_token_limit):
        processed=processor(text=[prompt],images=images,return_tensors="pt")
    tokens=processed.input_ids[0].tolist()
    if tokens!=source.context.input_token_ids or processed.image_grid_thw.tolist()!=[p.grid_thw for p in source.context.pages]:
        raise ValueError("Rebuilt model input differs from the source capture")
    target=source.output_token_ids[step]
    sequence=torch.tensor([tokens+source.output_token_ids[:step]],device="cuda")
    attention=torch.ones_like(sequence)
    grid=processed.image_grid_thw.to("cuda")
    pixels=processed.pixel_values.float()
    ip=processor.image_processor
    channels=len(ip.image_mean)
    if pixels.shape[1]%channels:
        raise ValueError("Pixel rows do not divide into colour channels")
    # A constant image of the consumed size, normalized like the input (rows hold each channel contiguously; see pack_pixels).
    level=(0.0 if baseline=="black" else 255.0)*ip.rescale_factor
    channel=(level-np.asarray(ip.image_mean,dtype=np.float64))/np.asarray(ip.image_std,dtype=np.float64)
    constant=torch.tensor(np.broadcast_to(channel[None,:,None],(pixels.shape[0],channels,pixels.shape[1]//channels)).reshape(tuple(pixels.shape)),dtype=torch.float32)
    image_positions=[i for page in source.context.pages for i in page.position_indices]
    count=sum(p.numel() for p in model.parameters())
    # Upcasting adds two bytes per parameter; checkpointed activations need a few GiB more.
    needed=count*2+4*1024**3
    if torch.cuda.mem_get_info()[0]<needed:
        model=None
        release_runtime()
        raise ValueError(f"Integrated gradients needs about {needed/1024**3:.0f} GiB free GPU memory beyond the BF16 weights to run in float32")
    if any(m.p for m in model.modules() if isinstance(m,torch.nn.Dropout)) or any(getattr(c,"attention_dropout",0) for c in (model.config.text_config,model.config.vision_config)):
        raise ValueError("Checkpointed float32 gradients need a dropout-free model")

    def image_embeddings(values):
        output=model.model.get_image_features(pixel_values=values.to("cuda"),image_grid_thw=grid)
        return torch.cat(list(output),dim=0) if isinstance(output,(list,tuple)) else output

    def log_probability(embeddings, grad):
        x=torch.tensor(embeddings,device="cuda",dtype=torch.float32,requires_grad=grad)
        embedded=base.clone(); embedded[0,positions]=x
        logits=model(inputs_embeds=embedded,attention_mask=attention,position_ids=rope,use_cache=False).logits[0,-1].float()
        return x,logits,torch.log_softmax(logits,dim=-1)[target]

    def gradient_at(point):
        with torch.enable_grad():
            x,_logits,value=log_probability(point,True)
            (gradient,)=torch.autograd.grad(value,x)
        return gradient.double().cpu().numpy()

    torch.cuda.reset_peak_memory_stats()
    try:
        model.requires_grad_(False)
        model.float()
        # Hugging Face checkpointing is active only in training mode; the model is dropout-free (checked above).
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.train()
        with torch.no_grad():
            model.model.rope_deltas=None
            native=torch.log_softmax(model(input_ids=sequence,attention_mask=attention,pixel_values=pixels.to("cuda"),image_grid_thw=grid,use_cache=False).logits[0,-1].float(),dim=-1)[target]
            actual=image_embeddings(pixels).double().cpu().numpy()
            reference=image_embeddings(constant).double().cpu().numpy()
            rope,_=model.model.get_rope_index(input_ids=sequence,image_grid_thw=grid,attention_mask=attention)
            base=model.model.get_input_embeddings()(sequence).float()
            positions=(sequence[0]==model.config.image_token_id).nonzero().flatten()
            if positions.tolist()!=image_positions or actual.shape!=reference.shape or actual.shape[0]!=len(image_positions):
                raise ValueError("Visual token embeddings differ from the capture's image positions")
            _,logits,value=log_probability(actual,False)
            target_log_probability=float(value)
            # The embedding route must reproduce the native pixel forward before anything is attributed.
            if abs(target_log_probability-float(native))>1e-4:
                raise ValueError("Embedding-space forward differs from the native pixel forward")
            greedy=int(torch.argmax(logits))==target
            baseline_log_probability=float(log_probability(reference,False)[2])
        attribution,halves=integrate_with_check(gradient_at,actual,reference,steps,interrupted)
        peak=torch.cuda.max_memory_allocated()
    finally:
        # Never leave a float32, training-mode model behind the validated BF16 generation runtime.
        model=None
        release_runtime()
    values=[None]*len(tokens)
    for index,value in zip(image_positions,token_totals(attribution),strict=True):
        values[index]=float(value)
    return {"attribution":values,"target_token_id":target,"target_log_probability":target_log_probability,
            "baseline_log_probability":baseline_log_probability,"target_is_greedy":greedy,"half_rule_totals":halves,
            "runtime":{"gpu":torch.cuda.get_device_name(),"peak_gpu_allocated_bytes":peak,"native_log_probability":float(native),
                       "input":"visual token embeddings entering the language model",
                       "dtype":"float32 forward and backward on checkpoint weights upcast exactly from BF16; gradient checkpointing; TF32 disabled; float64 accumulation",
                       "frameworks":{name:version(name) for name in ("torch","transformers")},
                       "adapter_sha256":sha256(Path(__file__).read_bytes()).hexdigest()}}
