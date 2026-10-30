"""Pinned Transformers ColPali adapter. Torch is an optional GPU dependency."""
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from uuid import uuid4
import platform
import numpy as np
from PIL import Image

from .contracts import ADAPTER_VERSION, MODEL_ID, MODEL_REVISION, Position, PageEvidence, RetrievalCapture
from .scoring import maxsim, ownership, contributions
from docattentionlens import __version__

_runtime = None
_checkpoint_files = {}


def release_runtime():
    global _runtime
    _runtime = None
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def runtime(cache):
    global _runtime, _checkpoint_files
    if _runtime is None:
        from docattentionlens.generation.qwen import release_runtime as release_generator
        release_generator()
        from docattentionlens.retrieval.colqwen import release_runtime as release_colqwen
        release_colqwen()
        import torch
        from transformers import ColPaliForRetrieval, ColPaliProcessor, GemmaTokenizerFast, SiglipImageProcessor
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("ColPali requires a BF16-capable CUDA device")
        free, _ = torch.cuda.mem_get_info()
        if free < 8*1024**3:
            raise RuntimeError("At least 8 GiB free GPU memory is required for this bounded plan")
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Select the two processor components explicitly: slow image processing
        # for the validated PIL resize and fast tokenization for exact offsets.
        processor = ColPaliProcessor(
            SiglipImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(cache)),
            GemmaTokenizerFast.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(cache)),
        )
        processor.query_prefix = "Query: "
        model = ColPaliForRetrieval.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(cache), dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        image_processor = processor.image_processor
        vision = model.config.vlm_config.vision_config
        if type(image_processor).__name__ != "SiglipImageProcessor" or image_processor.size != {"height":448,"width":448} or not image_processor.do_resize or vision.patch_size != 14 or vision.image_size != 448 or image_processor.image_seq_length != 1024:
            raise RuntimeError("Processor geometry does not match the validated direct-resize mapping")
        # An actual processor fixture checks spatial orientation, channel order,
        # affine resizing and normalization against the independently resized PIL image.
        fixture = np.zeros((96, 160, 3), dtype=np.uint8)
        fixture[:48,:80] = [255,0,0]; fixture[:48,80:] = [0,255,0]; fixture[48:,:80] = [0,0,255]; fixture[48:,80:] = [255,255,255]
        pil = Image.fromarray(fixture)
        actual = image_processor(images=pil, return_tensors="np")["pixel_values"][0]
        expected = np.asarray(pil.resize((448,448), resample=image_processor.resample), dtype=np.float32).transpose(2,0,1)
        expected = expected * image_processor.rescale_factor
        expected = (expected-np.asarray(image_processor.image_mean,dtype=np.float32)[:,None,None])/np.asarray(image_processor.image_std,dtype=np.float32)[:,None,None]
        if not np.allclose(actual,expected,atol=1e-6,rtol=0):
            raise RuntimeError("Processor fixture did not reproduce pixel geometry")
        # Verify SigLIP's actual patch flattening path with a positional sentinel.
        embedding = model.vlm.model.vision_tower.vision_model.embeddings
        with torch.inference_mode():
            pixels = torch.tensor(actual[None],device="cuda",dtype=torch.bfloat16)
            projected = embedding.patch_embedding(pixels)
            reconstructed = projected.flatten(2).transpose(1,2)+embedding.position_embedding(embedding.position_ids)
            if not torch.equal(embedding(pixels), reconstructed):
                raise RuntimeError("Vision patch ordering is not the validated row-major path")
        _runtime = (torch, processor, model)
        from huggingface_hub import snapshot_download
        folder=Path(snapshot_download(MODEL_ID,revision=MODEL_REVISION,cache_dir=str(cache),local_files_only=True))
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix in (".json",".safetensors"):
                digest=sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda:stream.read(8*1024*1024),b""):
                        digest.update(chunk)
                _checkpoint_files[path.name]=digest.hexdigest()
    return _runtime


def capture(case, inputs, cache: Path, interrupted=lambda:False, *, capture_attention=False, attention_page_id=None, intervention=None, force_eager=False, capture_joint_attention=False, joint_answer_source=None):
    if not intervention and not force_eager:
        return _capture(case,inputs,cache,interrupted,capture_attention=capture_attention,attention_page_id=attention_page_id,capture_joint_attention=capture_joint_attention,joint_answer_source=joint_answer_source)
    _torch,_processor,model=runtime(cache)
    language=model.vlm.model.language_model
    original=language.config._attn_implementation
    try:
        language.set_attn_implementation("eager")
        record,arrays=_capture(case,inputs,cache,interrupted,capture_attention=capture_attention,attention_page_id=attention_page_id,intervention=intervention,capture_joint_attention=capture_joint_attention,joint_answer_source=joint_answer_source)
        return record.model_copy(update={"provenance":{**record.provenance,"attention_implementation":"eager controlled baseline/intervention"}}),arrays
    finally:
        language.set_attn_implementation(original)


def _capture(case, inputs, cache: Path, interrupted=lambda:False, *, capture_attention=False, attention_page_id=None, intervention=None, capture_joint_attention=False, joint_answer_source=None):
    """Inputs contain verified image bytes, PageRecord and raw ElementRecords.

    Reference answers and support labels are intentionally absent from processing.
    Each page has a separate forward pass; the query never conditions that pass.
    """
    torch, processor, model = runtime(cache)
    query_input = processor(text=[case.query], return_tensors="pt", padding="longest")
    if query_input["input_ids"].shape[1] > 256:
        raise ValueError("Processed query exceeds the 256-position capture limit; no truncation was applied")
    formatted = processor.tokenizer.bos_token+processor.query_prefix+case.query+processor.tokenizer.pad_token*10+"\n"
    offset_input = processor.tokenizer(formatted, return_offsets_mapping=True, add_special_tokens=False)
    if offset_input["input_ids"] != query_input["input_ids"][0].tolist():
        raise ValueError("Query offset alignment failed")
    start = len(processor.tokenizer.bos_token+processor.query_prefix)
    end = start+len(case.query)
    qp = []
    for i, (token, mask, offsets) in enumerate(zip(query_input["input_ids"][0].tolist(),query_input["attention_mask"][0].tolist(),offset_input["offset_mapping"],strict=True)):
        role = "query_text" if offsets[1]>start and offsets[0]<end else "augmentation" if token==processor.tokenizer.pad_token_id and mask else "prefix_or_special"
        qp.append(Position(index=i,token_id=token,piece=processor.tokenizer.convert_ids_to_tokens(token),role=role,eligible=bool(mask),offsets=list(offsets)))
    with torch.inference_mode():
        q = model(**query_input.to("cuda")).embeddings[0].float().cpu()
    arrays = {"query":q.numpy()}
    encodings = []
    attention_validation = {}
    attention_scores = []
    eager_q = q.numpy()
    if capture_attention:
        from .encoding_capture import capture_encoding
        if not attention_page_id or attention_page_id not in {p.page_id for p,_,_ in inputs}:
            raise ValueError("Select one candidate page for document encoding attention")
        eager_q, encoding, error = capture_encoding(torch,model,query_input.to("cuda"),"query","query",arrays)
        encodings.append(encoding)
        attention_validation = {"max_row_sum_error":error,"capture_on_off_embeddings_equal":True,
            "runtime_mask_verified":True,"embedding_kernel_tolerance":.02,"score_kernel_tolerance":.1,
            "max_embedding_kernel_error":float(np.abs(eager_q-q.numpy()).max())}
        if attention_validation["max_embedding_kernel_error"] > .02:
            raise ValueError("Retriever eager/SDPA embeddings exceed tolerance")
    evidence = []
    repeat_error = 0.0
    for page, raw, elements in inputs:
        if interrupted():
            raise RuntimeError("Capture interrupted before page processing")
        if sha256(raw).hexdigest()!=case.page_image_sha256[page.page_id]:
            raise ValueError("Page image differs from frozen case")
        with Image.open(BytesIO(raw)) as opened:
            if opened.size!=(page.width,page.height):
                raise ValueError("Source dimensions differ from the frozen page")
            if opened.getexif().get(274,1)!=1:
                raise ValueError("Unvalidated EXIF orientation")
            image = opened.convert("RGB")
            batch = processor(images=[image], return_tensors="pt")
        positions = []
        visual_index = 0
        mappings = []
        for i,(token, mask) in enumerate(zip(batch["input_ids"][0].tolist(),batch["attention_mask"][0].tolist(),strict=True)):
            bbox = None
            role = "bos" if token==processor.tokenizer.bos_token_id else "newline" if processor.tokenizer.decode([token])=="\n" else "visual_prompt"
            if token==processor.image_token_id:
                y,x=divmod(visual_index,32)
                bbox=[x*page.width/32,y*page.height/32,(x+1)*page.width/32,(y+1)*page.height/32]
                visual_index+=1; role="visual_patch"
            positions.append(Position(index=i,token_id=token,piece=processor.tokenizer.convert_ids_to_tokens(token),role=role,eligible=bool(mask),bbox=bbox))
            mappings.append(ownership(bbox,elements) if bbox else {"non_page":1.0})
        if visual_index!=1024 or tuple(batch["pixel_values"].shape)!=(1,3,448,448):
            raise ValueError("Page processing changed the validated grid")
        from ..attention_interventions import apply as apply_attention_intervention
        active_intervention=intervention if intervention and intervention["page_id"]==page.page_id else None
        with torch.inference_mode(), apply_attention_intervention(model,torch,active_intervention,positions):
            d = model(**batch.to("cuda")).embeddings[0].float().cpu()
            if not evidence:
                plain=model(**batch.to("cuda")).embeddings[0].float().cpu()
                repeat_error=float((plain-d).abs().max())
                if repeat_error!=0:
                    raise ValueError("Repeated plain forward differed from captured embeddings")
        score,winners,maxima,_ = maxsim(q.numpy(),d.numpy(),[p.eligible for p in qp],[p.eligible for p in positions])
        if capture_attention:
            eager_d = d.numpy()
            if page.page_id == attention_page_id:
                eager_d, encoding, error = capture_encoding(torch,model,batch.to("cuda"),"document","document",arrays,page.page_id)
                encodings.append(encoding)
                attention_validation["max_row_sum_error"] = max(attention_validation["max_row_sum_error"],error)
                attention_validation["max_embedding_kernel_error"] = max(attention_validation["max_embedding_kernel_error"],float(np.abs(eager_d-d.numpy()).max()))
                if attention_validation["max_embedding_kernel_error"] > .02:
                    raise ValueError("Retriever eager/SDPA embeddings exceed tolerance")
                if capture_joint_attention:
                    from .joint_attention import capture_joint
                    if interrupted():
                        raise RuntimeError("Capture interrupted before diagnostic joint processing")
                    joint,joint_error=capture_joint(torch,model,processor,image,case.query,positions,mappings,arrays,page.page_id,False)
                    encodings.append(joint)
                    attention_validation["max_row_sum_error"] = max(attention_validation["max_row_sum_error"],joint_error)
                    if joint_answer_source=="reference":
                        if interrupted():
                            raise RuntimeError("Capture interrupted before diagnostic joint answer processing")
                        # The only place a case annotation enters the model; excluded from scoring.
                        answered,answer_error=capture_joint(torch,model,processor,image,case.query,positions,mappings,arrays,page.page_id,False,
                                                            answer=case.reference_answer,encoding_id="joint_answer")
                        encodings.append(answered)
                        attention_validation["max_row_sum_error"] = max(attention_validation["max_row_sum_error"],answer_error)
            attention_score = maxsim(eager_q,eager_d,[p.eligible for p in qp],[p.eligible for p in positions])[0]
            attention_scores.append(attention_score)
            if abs(attention_score-score) > .1:
                raise ValueError("Retriever eager/SDPA scores exceed tolerance")
        # Invoke the upstream scorer on precisely the eligible, unpadded FP32 domain.
        reference = float(processor.score_retrieval([q[[p.eligible for p in qp]]],[d[[p.eligible for p in positions]]],batch_size=1)[0,0])
        error = abs(score-reference)
        if error>1e-4:
            raise ValueError(f"Scorer reconstruction failed: {error}")
        name="page_"+str(len(evidence)); arrays[name]=d.numpy()
        ledger=contributions(winners,maxima,mappings)
        ledger["scorer_rounding"]=score-sum(ledger.values())
        evidence.append(PageEvidence(page_id=page.page_id,image_sha256=page.image_sha256,width=page.width,height=page.height,page_number=page.page_number,score=score,reference_score=reference,reconstruction_error=error,positions=positions,vector_array=name,ownership=mappings,contributions=ledger,winner_indices=winners,query_maxima=maxima))
    record = RetrievalCapture(capture_id="capture_"+uuid4().hex,case_id=case.case_id,created_at=datetime.now(UTC).isoformat(),producer_version=__version__,dependencies=[case.case_id,*[p.page_id for p in evidence]],dataset_version=case.dataset_version,annotation_version=case.annotation_version,query=case.query,query_positions=qp,query_array="query",pages=evidence,
        provenance={"adapter_version":ADAPTER_VERSION,"model_id":MODEL_ID,"model_revision":MODEL_REVISION,"processor_revision":MODEL_REVISION,"query_prefix":"Query: ","query_augmentation_pad_tokens":10,"formatted_query":formatted,"visual_prompt":processor.visual_prompt_prefix,"model_input":"RGB page image only; separate query pass","evaluation_labels_consumed":False,"torch":version("torch"),"transformers":version("transformers"),"numpy":version("numpy"),"pillow":version("Pillow"),"python":platform.python_version(),"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"eval_mode":True,"encoder_dtype":"bfloat16","stored_dtype":"float32 lossless cast of BF16 outputs","attention_implementation":"sdpa","seed":0,"tf32":False},
        scorer={"quantity":"late_interaction_similarity","formula":"sum_i max_j dot(q_i,d_j)","precision":"float32 CPU","normalization":"encoder L2 normalization in BF16; no subsequent normalization","eligible_domain":"all active positions, including query prefix/augmentation and non-page visual prompt","padding":"inactive positions excluded; no zero padding in max domain","tie_policy":"first eligible position","candidate_scope":"explicit selected pages; not corpus retrieval"},
        geometry={"policy":"siglip_direct_resize_row_major_v1","input_size":[448,448],"patch_size":14,"grid":[32,32],"order":"row-major; x changes fastest","operations":["RGB conversion","anisotropic direct resize; no crop/pad/tile","rescale and channel normalization","14x14 convolution; flatten spatial axes row-major","one-to-one projection into 1024 image-token positions"],"resample":int(processor.image_processor.resample),"ownership_policy":"exact_cell_equal_overlap_v1","flagged_geometry":"positive raw intersection allocated to unreviewed_geometry; no silent repair","residuals":["background","non_page","unreviewed_geometry"],"coverage":"full source page; rectangles describe nominal patch footprint, not receptive-field attribution"},
        validation={"processor_fixture":True,"patch_order_runtime_check":True,"source_dimensions_and_hashes":True,"score_tolerance_absolute":1e-4,"max_reconstruction_error":max(p.reconstruction_error for p in evidence),"ownership_conserved":True,"signed_values_preserved":True})
    if capture_attention:
        baseline_scores = [p.score for p in evidence]
        record = record.model_copy(update={"schema_version":"0.9.0"})
        if sorted(range(len(evidence)), key=lambda i:-baseline_scores[i]) != sorted(range(len(evidence)), key=lambda i:-attention_scores[i]):
            raise ValueError("Retriever eager/SDPA candidate rankings differ")
        attention_validation.update({"candidate_ranking_equal":True,
            "max_score_kernel_error":max(abs(a-b) for a,b in zip(baseline_scores,attention_scores,strict=True)),
            "baseline_scores":baseline_scores,"eager_attention_scores":attention_scores})
    record=record.model_copy(update={"attention_encodings":encodings,"provenance":{**record.provenance,"checkpoint_file_sha256":_checkpoint_files,"adapter_source_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),"scoring_source_sha256":sha256(Path(__file__).with_name("scoring.py").read_bytes()).hexdigest(),"image_processor":type(processor.image_processor).__name__,"tokenizer":type(processor.tokenizer).__name__,
        **({"attention_adapter_version":"colpali_language_attention_v1","attention_implementation":"eager language extraction; SDPA baseline scoring", "stored_dtype":"embeddings: FP32 lossless cast of BF16 outputs; attention: eager FP32 post-softmax weights", "attention_adapter_source_sha256":sha256(Path(__file__).with_name("encoding_capture.py").read_bytes()).hexdigest(),"attention_coverage":"query and one selected document; full language encoding matrices; vision tower excluded"} if capture_attention else {})},"scorer":{**record.scorer,"ledger_precision":"float64 allocation; scorer_rounding reconciles FP32 score reduction"},"validation":{**record.validation,"repeat_plain_forward_max_abs_error":repeat_error,**({"attention":attention_validation} if capture_attention else {})}})
    if capture_joint_attention:
        record=record.model_copy(update={"schema_version":"0.11.0","provenance":{**record.provenance,
            "diagnostic_joint_pass":True,"joint_capture_policy":"retain_document_v1",
            "joint_attention_source_sha256":sha256(Path(__file__).with_name("joint_attention.py").read_bytes()).hexdigest(),
            "attention_coverage":"separate query and document encodings plus diagnostic joint page-plus-question language encoding; joint pass is excluded from retrieval scoring; vision tower excluded"}})
    if capture_attention and capture_joint_attention and joint_answer_source=="reference":
        record=record.model_copy(update={"schema_version":"0.12.0","provenance":{**record.provenance,"diagnostic_joint_answer_source":"reference",
            "evaluation_labels_consumed":True,
            "evaluation_labels_consumed_by":"diagnostic joint_answer encoding only (reference answer in its prompt); query, page and scores use no labels"}})
    return record,arrays
