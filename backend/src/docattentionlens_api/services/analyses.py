"""Resolve verified saved inputs; scientific calculations belong to method modules."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from pydantic import ValidationError
from docattentionlens_api.repositories.analyses import AnalysisRepository
from docattentionlens_api.repositories.generations import GenerationRepository
from docattentionlens_api.errors import fail, readable
from docattentionlens.datasets.adapter import canonical
from docattentionlens.methods.contracts import AnalysisResult, Context
from docattentionlens.methods.registry import get_definition, prepare_parameters, prediction_capabilities
from docattentionlens.methods._shared import valid_support
from docattentionlens.generation.inspection import source_view
import docattentionlens.generation.inspection as inspection
import docattentionlens.methods as methods_package
import docattentionlens.methods._full_matrix as full_matrix


@lru_cache
def calculation_sha256():
    """Hash of the code that computes saved-row analyses.

    Results are cached by identity, so a changed formula must never reuse an earlier
    result under an unchanged descriptor or implementation version. Line endings are
    normalised so identical code hashes identically across checkouts.
    """
    digest = sha256()
    for path in [*sorted(Path(methods_package.__file__).parent.glob("*.py")), Path(inspection.__file__)]:
        digest.update(path.name.encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def calculate(jobs, body):
    if body.source_index is not None and (body.sample_capture_ids or body.comparison_capture_id):
        fail("invalid_parameters","Multi-capture analyses require prediction sources; arbitrary question-token indices are not aligned across captures")
    repo = GenerationRepository(jobs)
    record = repo.get(body.generation_capture_id)
    try:
        definition = get_definition(body.method_id, body.variant_id)
    except KeyError:
        fail("unknown_method", "Unknown method variant", 404)
    if full_matrix.applies(definition.descriptor, record):
        return calculate_full_matrix(jobs, repo, record, definition, body)
    original_record = record
    parameters_in = body.parameters
    try:
        if body.source_index is not None:
            record, _ = source_view(record, body.source_index)
            parameters_in = {**body.parameters, "step": 0}
        parameters = prepare_parameters(body.method_id, body.variant_id, parameters_in, record)
        needs_document=definition.descriptor.perspective=="document_aware" or bool(body.sample_capture_ids) or body.comparison_capture_id is not None
        needs_table=definition.descriptor.perspective=="table_aware"
        from .document_context import annotations
        from .table_context import table_inputs
        doc=annotations(jobs,record) if needs_document else None
        table=table_inputs(jobs,record) if needs_table else None
        capabilities=prediction_capabilities(record)
        if table:
            capabilities.add("table_structure_annotations")
            if table.grounding is not None: capabilities.add("table_grounding_annotations")
        if doc:
            if doc.elements: capabilities.add("element_type_annotations")
            if any(e.reading_order is not None for e in doc.elements): capabilities.add("reading_order_annotations")
            if doc.case.support_element_ids: capabilities.add("designated_support_annotations")
        retrieval=None
        if record.retrieval_capture_id and record.retrieval_capture_id.startswith("capture_"):
            from docattentionlens_api.repositories.captures import CaptureRepository
            retrieval=CaptureRepository(jobs).get(record.retrieval_capture_id)
            if retrieval.case_id!=record.case_id or retrieval.query!=record.context.query:
                fail("invalid_reference","Retrieval and generation case/query lineage differs")
            capabilities.add("retrieval_evidence")
        comparison=None; alignment=None; experiment=None
        if body.comparison_capture_id:
            comparison_record=repo.get(body.comparison_capture_id)
            comparison_doc=annotations(jobs,comparison_record)
            comparison_step=next((s for s in comparison_record.steps if s.step==parameters.step),None)
            if comparison_step is None: fail("unsupported_capture","Comparison prediction step is not captured")
            comparison_table=table_inputs(jobs,comparison_record) if needs_table else None
            comparison=Context(comparison_record,repo.store.array(comparison_record,comparison_step.array),comparison_doc,table=comparison_table)
            capabilities.add("paired_capture")
            # Table contrasts compare different questions about one table; the method checks the table revision and geometry.
            same_table=bool(table and comparison_table and comparison_table.table.table_id==table.table.table_id)
            if doc.case.case_id!=comparison_doc.case.case_id and not body.experiment_result_id and not same_table:
                fail("invalid_reference","Changed cases require verified experiment lineage")
        if body.experiment_result_id:
            if body.experiment_result_id.startswith("document_study_"):
                from docattentionlens_api.repositories.document_studies import DocumentStudyRepository
                experiment=DocumentStudyRepository(jobs).get(body.experiment_result_id)
                if experiment.kind!="result" or not any(o.capture_id==record.capture_id for o in experiment.outcomes):
                    fail("invalid_reference","Select a capture from this completed document experiment")
                capabilities.add("experiment_outcomes")
            else:
                from docattentionlens_api.repositories.experiments import ExperimentRepository
                experiment=ExperimentRepository(jobs).get(body.experiment_result_id)
                if not comparison: fail("invalid_reference","Select two captures from the layout experiment")
                base=next((a for a in experiment.arms if a.case.case_id==record.case_id),None)
                other=next((a for a in experiment.arms if a.case.case_id==comparison.record.case_id),None)
                refs={o.get("generation",{}).get("capture_id") for o in experiment.outcomes}
                if experiment.record_kind!="experiment_result" or base is None or other is None or not {record.capture_id,comparison.record.capture_id}<=refs:
                    fail("invalid_reference","Captures do not belong to the selected completed experiment")
                alignment={base.element_lineage[i]:other.element_lineage[i] for i in base.element_lineage if i in other.element_lineage}
                # Unchanged pages retain their source identities.
                alignment.update({e.element_id:e.element_id for e in doc.elements if e.page_id!=base.page.page_id})
                capabilities.add("element_lineage")
        sample_ids=body.sample_capture_ids
        if len(sample_ids)!=len(set(sample_ids)): fail("invalid_parameters","Sample captures must be distinct")
        samples=[]
        for identifier in sample_ids:
            if identifier==record.capture_id: continue
            cap=repo.get(identifier); cap_doc=annotations(jobs,cap)
            selected_sample=next((s for s in cap.steps if s.step==parameters.step),None)
            if selected_sample is None: fail("unsupported_capture","A sample prediction step is not captured")
            cap_retrieval=None
            if cap.retrieval_capture_id and cap.retrieval_capture_id.startswith("capture_"):
                from docattentionlens_api.repositories.captures import CaptureRepository
                cap_retrieval=CaptureRepository(jobs).get(cap.retrieval_capture_id)
                if cap_retrieval.case_id!=cap.case_id or cap_retrieval.query!=cap.context.query:
                    fail("invalid_reference","A sample retrieval capture has mismatched case/query lineage")
            samples.append(Context(cap,repo.store.array(cap,selected_sample.array),cap_doc,cap_retrieval,table=table_inputs(jobs,cap) if needs_table else None))
        if sample_ids: capabilities.add("corpus_sample_manifest")
        missing = set(definition.descriptor.requires) - capabilities
        if missing:
            fail("unsupported_capture", "Method requirements are absent from this capture: " + ", ".join(sorted(missing)))
        selected = next(s for s in record.steps if s.step == parameters.step)
        mask = valid_support(record, parameters.step)
    except KeyError:
        fail("unknown_method", "Unknown method variant", 404)
    except OSError:
        fail("artifact_incomplete","A comparison/sample payload is missing or failed integrity verification",409)
    except NotImplementedError as exc:
        fail("unsupported_method", str(exc), 422)
    except (ValueError, ValidationError) as exc:
        fail("invalid_parameters", readable(exc), 422)
    # A retained row's array hash, or for a full-attention prompt row the hashes of the triangles it is read from.
    source_sha = repo.store.source_sha256(record, selected.array)
    if source_sha is None or not record.content_sha256:
        fail("artifact_incomplete", "Capture is missing verified array/manifest identities", 409)
    frame_count = max(1,len(getattr(parameters, "layers", [])))*max(1,len(getattr(parameters, "heads", [])))
    multiple_sources = bool({"multiple_source_rows", "question_source_rows"} & set(definition.descriptor.requires))
    budget=4_000_000 if needs_document or needs_table or multiple_sources else 65536
    if selected.target_count * frame_count * (len(samples)+1) > budget:
        fail("analysis_budget_exceeded", "Reduce the compared layer/head population", 422)
    specification = {"method_id": body.method_id, "variant_id": body.variant_id,
        "descriptor_sha256": sha256(canonical(definition.descriptor.model_dump(mode="json"))).hexdigest(),
        "implementation_version": definition.descriptor.implementation_version,
        "calculation_sha256": calculation_sha256(),
        "capture_id": record.capture_id, "capture_sha256": record.content_sha256, "array_sha256": source_sha,
        "parameters": parameters.model_dump(mode="json"),
        **({"source_index": body.source_index} if body.source_index is not None else {})}
    inputs={}
    if doc: inputs[record.capture_id]={"capture_sha256":record.content_sha256,"annotation_sha256":doc.annotation_sha256}
    if table: inputs.setdefault(record.capture_id,{"capture_sha256":record.content_sha256})["table_annotation_sha256"]=table.annotation_sha256
    for c in [*samples,*([comparison] if comparison else [])]:
        inputs[c.record.capture_id]={"capture_sha256":c.record.content_sha256,"annotation_sha256":c.document.annotation_sha256,
                                     **({"table_annotation_sha256":c.table.annotation_sha256} if c.table else {})}
        if c.retrieval: inputs[c.retrieval.capture_id]={"capture_sha256":c.retrieval.content_sha256}
    if retrieval: inputs[retrieval.capture_id]={"capture_sha256":retrieval.content_sha256}
    if experiment: inputs[body.experiment_result_id]={"manifest_sha256":experiment.content_sha256}
    if inputs:
        specification["input_manifest"]={"inputs":inputs,"sample_capture_ids":sample_ids,"comparison_capture_id":body.comparison_capture_id,"alignment":alignment}
    identifier = "analysis_" + sha256(canonical(specification)).hexdigest()
    store = AnalysisRepository(jobs)
    existing = store.find(identifier)
    if existing is not None:
        return existing
    try:
        values = repo.store.source_array(record, selected.array)
    except (OSError, ValueError):
        fail("artifact_incomplete", "Attention payload is missing or failed integrity verification", 409)
    try:
        @lru_cache(maxsize=256)
        def read_source(index):
            source = next((s for s in original_record.steps if s.source_index == index), None)
            source_record = original_record
            if source is None:
                source_record, source = source_view(original_record, index)
            else:
                source_record = original_record.model_copy(update={"steps": [source]})
            return repo.store.source_array(original_record, source.array), valid_support(source_record, source.step)
        context=Context(record,values,doc,retrieval,comparison,alignment=alignment,experiment=experiment,
                        read_array=lambda name:repo.store.array(record,name), read_source=read_source, source_record=original_record,
                        table=table)
        if samples or sample_ids:
            from dataclasses import replace
            context=replace(context,samples=(context,*samples))
        calculated = definition.calculate(context, parameters)
    except ValueError as exc:
        fail("invalid_parameters", str(exc), 422)
    result = AnalysisResult(**calculated.model_dump(), analysis_id=identifier, method_id=body.method_id,
        variant_id=body.variant_id, implementation_version=definition.descriptor.implementation_version,
        generation_capture_id=record.capture_id, capture_sha256=record.content_sha256, array_sha256=source_sha,
        parameters=specification["parameters"], source_index=selected.source_index, predicted_index=selected.predicted_index,
        target_count=selected.target_count, valid_target_indices=[int(i) for i in mask.nonzero()[0]],
        quantity=definition.descriptor.quantity, units=definition.descriptor.output.units,
        input_manifest=specification.get("input_manifest",{}),
        coverage={**record.coverage, "selection": (f"{len(calculated.source_indices)} explicit captured source rows; per-source declared causal targets" if calculated.source_indices else
                  ("one retained question-token row" if body.source_index is not None else "one retained prediction") + "; complete declared causal targets"),
                  "source_count": len(calculated.source_indices) or 1,
                  "source_indices": calculated.source_indices,
                  "layer_count": record.layer_count, "query_head_count": record.query_head_count})
    return store.put(result)


class _Incomplete(Exception):
    """A saved payload is missing or failed integrity verification (not a scientific error)."""


def calculate_full_matrix(jobs, repo, record, definition, body):
    """Square-matrix and image-source methods on a full-attention generator capture.

    The capture is analysed as a causal encoding truncated at the selected source
    (`methods/_full_matrix.py`); formulas stay in each method module.
    """
    if body.sample_capture_ids or body.comparison_capture_id or body.experiment_result_id:
        fail("invalid_parameters", "Full-matrix analyses read one capture; comparison, sample and experiment captures are not supported")
    try:
        missing = set(definition.descriptor.requires) - prediction_capabilities(record)
        if missing:
            fail("unsupported_capture", "Method requirements are absent from this capture: " + ", ".join(sorted(missing)))
        source, predicted, retained, view, parameters, saved = full_matrix.prepare(
            record, definition, body.parameters, body.source_index)
        identity = full_matrix.identity(record, source)
        mask = valid_support(view, 0)
    except NotImplementedError as exc:
        fail("unsupported_method", str(exc), 422)
    except (ValueError, ValidationError) as exc:
        fail("invalid_parameters", readable(exc), 422)
    refs = {a.name: a.sha256 for a in record.arrays}
    if (retained is not None and retained not in refs) or not record.content_sha256:
        fail("artifact_incomplete", "Capture is missing verified array/manifest identities", 409)
    # A retained row keeps its own array identity; any other source is identified by the full arrays.
    array_sha256 = refs[retained] if retained is not None else sha256(canonical(identity)).hexdigest()
    specification = {"method_id": body.method_id, "variant_id": body.variant_id,
        "descriptor_sha256": sha256(canonical(definition.descriptor.model_dump(mode="json"))).hexdigest(),
        "implementation_version": definition.descriptor.implementation_version,
        "calculation_sha256": calculation_sha256(),
        "capture_id": record.capture_id, "capture_sha256": record.content_sha256, "array_sha256": array_sha256,
        "parameters": saved, "full_attention": identity,
        **({"source_index": body.source_index} if body.source_index is not None else {})}
    identifier = "analysis_" + sha256(canonical(specification)).hexdigest()
    store = AnalysisRepository(jobs)
    existing = store.find(identifier)
    if existing is not None:
        return existing

    def read(name):
        try:
            return repo.store.array(record, name)
        except (OSError, ValueError) as exc:
            raise _Incomplete(name) from exc
    try:
        context, view = full_matrix.context(record, source, read, full_matrix.used_layers(parameters))
        calculated = definition.calculate(context, parameters)
    except _Incomplete as exc:
        fail("artifact_incomplete", f"Full attention payload {exc} is missing or failed integrity verification", 409)
    except ValueError as exc:
        fail("invalid_parameters", str(exc), 422)
    n = record.full_attention.position_count
    calculated = calculated.model_copy(update={"notices": [*calculated.notices, *full_matrix.notices(view, source, n)]})
    if len(calculated.model_dump_json().encode()) > 4*1024**2:
        fail("analysis_budget_exceeded", "Analysis result exceeds four MiB; reduce compared axes, frames or edges", 422)
    result = AnalysisResult(**calculated.model_dump(), analysis_id=identifier, method_id=body.method_id,
        variant_id=body.variant_id, implementation_version=definition.descriptor.implementation_version,
        generation_capture_id=record.capture_id, capture_sha256=record.content_sha256, array_sha256=array_sha256,
        parameters=saved, source_index=source, predicted_index=predicted, target_count=source + 1,
        valid_target_indices=[int(i) for i in mask.nonzero()[0]],
        quantity=definition.descriptor.quantity, units=definition.descriptor.output.units,
        coverage={**record.coverage,
                  "selection": (f"{len(calculated.source_indices)} source rows" if len(calculated.source_indices) > 1 else "one source row")
                               + f" of a causal square over positions 0–{source}, rebuilt from saved full attention",
                  "source_count": len(calculated.source_indices) or 1, "source_indices": calculated.source_indices,
                  "layer_count": record.layer_count, "query_head_count": record.query_head_count,
                  "full_attention": {**record.coverage.get("full_attention", {}),
                                     **{k: identity[k] for k in ("layout", "position_count", "square_limit", "source_index")}}})
    return store.put(result)
