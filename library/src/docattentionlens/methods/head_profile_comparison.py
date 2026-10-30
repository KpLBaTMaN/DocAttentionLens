"""Head profile comparison: canonical definition and calculation ownership.

The declared profile is the head preference catalogue's document-balanced
element-type profile (`head_specialisation.py`): per row, the share of image
attention on each annotated element type plus unreviewed background, averaged
within each document and then equally across documents of the declared sample.
Heads are compared by cosine similarity of those profiles. A head whose profile
cannot be formed on every sample row (no image mass, no mapped element) is
unavailable, never a zero vector.
"""
import numpy as np
from pydantic import Field
from .contracts import AxisIndex, Calculation, MethodDefinition, MethodDescriptor, RowParameters, ResearchPlot, ResearchPoint
from ._document import REQUIRES, document, sample_contexts, table
from ._shared import parameter_schema, selection
from .head_specialisation import document_balanced_profile, vocabulary_of

UNDEFINED = ("Type profiles require positive image mass", "No annotated element is mapped to this selection")


class ProfileParameters(RowParameters):
    """`head` is unused: the compared population is `layers` × `heads`."""
    layers: list[AxisIndex] = Field(default_factory=lambda: [0], max_length=128)
    heads: list[AxisIndex] = Field(default_factory=list, max_length=128,
                                   description="Empty selects all captured query heads.")


DESCRIPTOR = MethodDescriptor.model_validate({'method_id': 'heads.profile_similarity',
 'variant_id': 'reviewed_profile_cosine_v1',
 'descriptor_version': '0.2.0',
 'status': 'implemented',
 'implementation_version': 'document_type_profile_cosine_v1',
 'perspective': 'document_aware',
 'family': 'head_and_representation',
 'question': 'Which heads have similar declared document-behaviour profiles?',
 'evidence_kind': 'representation',
 'quantity': 'head_profile_similarity',
 'definition': 'Cosine between nonzero declared profiles sampled over the same reviewed corpus and '
               'profile dimensions. Profile: per-row image-attention share on each annotated element type '
               'plus unreviewed background, averaged within each document and then equally across the '
               'declared sample documents.',
 'source_references': ['definition:cosine similarity',
                       'library/src/docattentionlens/methods/head_specialisation.py: declared profile'],
 'requires': [*REQUIRES, 'corpus_sample_manifest'],
 'execution_requirement': 'saved_capture',
 'target_requirement': 'None',
 'parameters_schema': parameter_schema(ProfileParameters),
 'output': {'schema': 'head_similarity_matrix',
            'units': 'cosine similarity',
            'signed': True,
            'support': 'declared head profiles',
            'normalization': 'unit-length nonzero profiles'},
 'displays': ['matrix', 'head_grid'],
 'comparison_constraints': ['Same sampling, profile definitions and reviewed relation versions.'],
 'limitations': ['Clusters and behaviour names need separate evidence; zero profiles are unavailable.',
                 'Profiles describe the declared sample; stable head behaviour needs held-out replication.',
                 'Profiles are non-negative shares, so cosine lies in [0, 1] here even though the output '
                 'contract is signed.'],
 'validation_evidence': ['library/tests/methods/test_planned_methods.py: independent cosine and '
                         'profile fixtures, unavailable heads and document balancing'],
 'name': 'Head profile comparison',
 'summary': 'Compare attention heads using cosine similarity between declared profiles over a selected '
            'corpus. Profile similarity does not measure output importance.'})


def cosine(p, q):
    return float(p @ q / (np.linalg.norm(p)*np.linalg.norm(q)))


def calculate(context, parameters):
    samples = sample_contexts(context)
    vocabulary = vocabulary_of(samples)
    heads = [(layer, head) for layer in selection(parameters.layers, context.record.layer_count)
             for head in selection(parameters.heads, context.record.query_head_count)]
    if len(heads) < 2:
        raise ValueError("Select at least two heads to compare")
    profiles = {}
    unavailable = []
    for key in heads:
        try:
            profile = document_balanced_profile(samples, parameters.step, *key, vocabulary)
        except ValueError as exc:
            if not str(exc).startswith(UNDEFINED):
                raise
            unavailable.append(key)
            continue
        if not np.isfinite(profile).all() or np.linalg.norm(profile) <= 0:
            unavailable.append(key)
            continue
        profiles[key] = profile
    label = lambda key: f"L{key[0]} · H{key[1]}"
    labels = [label(k) for k in heads]
    values = [[cosine(profiles[a], profiles[b]) if a in profiles and b in profiles else None for b in heads] for a in heads]
    last = max(len(heads)-1, 1)
    points = [ResearchPoint(label=f"{label(a)} vs {label(b)}", x=j/last, y=i/last, value=v)
              for i, a in enumerate(heads) for j, b in enumerate(heads) if (v := values[i][j]) is not None]
    return Calculation(
        table=table("Head profile cosine similarity", labels, labels, values, "cosine similarity of declared profiles"),
        tables=[table("Declared head profiles", [*vocabulary, "Background/unreviewed image"], [label(k) for k in profiles],
                      [profiles[k].tolist() for k in profiles], "image-attention fraction")],
        plots=[ResearchPlot(kind="grid", title="Head profile similarity", x_label="Compared head", y_label="Head", points=points)],
        normalization="each profile divided by its Euclidean norm; profiles are equal-document means of per-row image-attention type shares",
        notices=[f"{len(samples)} saved captures across {len({document(c).case.document_id for c in samples})} documents; similarity describes this declared sample.",
                 *([f"Unavailable (profile undefined on a sample row): {', '.join(label(k) for k in unavailable)}."] if unavailable else []),
                 "Similar profiles do not establish shared function, output importance or causation."])


DEFINITION = MethodDefinition(DESCRIPTOR, ProfileParameters, calculate, True)
