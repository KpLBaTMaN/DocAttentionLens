"""Layout change comparison: canonical definition and calculation ownership."""
from .contracts import MethodDefinition, MethodDescriptor

DESCRIPTOR = MethodDescriptor.model_validate({'method_id': 'layout.move_group',
 'variant_id': 'reviewed_intact_group_v1',
 'descriptor_version': '0.1.2',
 'status': 'planned',
 'implementation_version': None,
 'perspective': 'document_aware',
 'family': 'perturbations_and_interventions',
 'question': 'Does a consuming RAG stage preserve its outcome after a reviewed intact-group move?',
 'evidence_kind': 'input_perturbation_effect',
 'quantity': 'paired_stage_outcome_delta',
 'definition': 'Compare baseline/variant outcomes under an explicit reviewed manipulation and controls.',
 'source_references': ['design/experiments-and-evaluation.md'],
 'requires': ['reviewed_group_relation',
              'variant_assets',
              'element_lineage',
              'expected_answer_policy',
              'consuming_stage',
              'controls'],
 'execution_requirement': 'paired_execution',
 'target_requirement': 'Declared stage-specific outcome',
 'parameters_schema': {'type': 'object',
                       'properties': {'stage': {'enum': ['retrieval',
                                                         'generation_fixed_evidence',
                                                         'rag']}},
                       'additionalProperties': False},
 'output': {'schema': 'paired_experiment_outcomes',
            'units': 'outcome-specific units',
            'signed': True,
            'support': 'reviewed paired cases',
            'normalization': 'declared outcome protocol'},
 'displays': ['paired_pages', 'evidence_junction'],
 'comparison_constraints': ['Same paired protocol, resources and outcome definitions.'],
 'limitations': ['Movement may alter distance, reading order and rendering; isolate or report '
                 'confounds.'],
 'validation_evidence': [],
 'name': 'Layout change comparison',
 'summary': 'Compare measured outcomes before and after moving a reviewed document group, preserving '
            'its content and associations under a declared experiment protocol.'})

from .attention_comparison import Parameters, compute
DESCRIPTOR=DESCRIPTOR.model_copy(update={"status":"implemented","implementation_version":"reviewed_layout_element_comparison_v1",
    "descriptor_version":"0.2.0","parameters_schema":Parameters.model_json_schema(),
    "requires":["attention_weights","generator_prediction_row_capture","complete_declared_support","validated_ownership_map","mapped_elements","element_type_annotations","paired_capture","element_lineage"],
    "execution_requirement":"saved_capture","quantity":"aligned_layout_element_comparison",
    "output":DESCRIPTOR.output.model_copy(update={"result_schema":"method_analysis","units":"mass difference; divergence bits; overlap fraction",
        "support":"verified aligned elements; unaligned and residual support separate",
        "normalization":"mass differences retain original domains; divergence separately normalizes aligned-element vectors"}),
    "displays":["table","paired_pages"],
    "validation_evidence":["library/tests/test_experiments.py: reviewed pixel moves; library/tests/methods/test_document_methods.py: aligned comparisons"],
    "summary":"Compare saved baseline and moved-group captures using verified experiment element lineage and compatible prediction histories."})
DEFINITION=MethodDefinition(DESCRIPTOR,Parameters,compute,True)
