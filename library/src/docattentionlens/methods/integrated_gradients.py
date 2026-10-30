"""Integrated gradients attribution: canonical definition and calculation ownership.

Sundararajan, Taly & Yan (2017), "Axiomatic Attribution for Deep Networks":
IG_i(x) = (x_i − x'_i) · ∫₀¹ ∂F(x' + α(x − x'))/∂x_i dα, approximated here by a
midpoint Riemann sum with m steps. Completeness (Σ_i IG_i = F(x) − F(x')) is the
method's own check and is always reported. F is the log-probability of the token a
saved capture predicted at one step, with the prompt and earlier output tokens held
fixed (teacher forcing). The input x is each consumed visual token's embedding as it
enters the language model (the vision encoder's output); the baseline x′ is the same
encoder's output for a constant black or white image of the consumed size. Token IDs
are not differentiable, and the pixel-space path through the vision tower does not converge.
Gradients come from a Document studies GPU run; this module owns the integration
rule, the result contract and its presentation.
"""
from typing import Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from .contracts import Calculation, MethodDefinition, MethodDescriptor, ResearchPlot, ResearchPoint, ResultTable
from ._shared import parameter_schema, statistic
from ..generation.contracts import MAX_NEW_TOKENS

COMPLETENESS_TOLERANCE = 0.05  # relative gap above which the result is flagged, never hidden


class IGParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    step: int = Field(default=0, ge=0, le=MAX_NEW_TOKENS - 1, strict=True, description="Captured prediction step whose token log-probability is attributed.")
    steps: int = Field(default=128, ge=2, le=1024, strict=True, title="Riemann steps",
                       description="Midpoint Riemann steps along the straight input-baseline path.")
    baseline: Literal["black", "white"] = Field(default="black", title="Baseline image",
                                                description="Constant RGB image of the consumed size; its vision-encoder output is the baseline embedding.")


DESCRIPTOR = MethodDescriptor.model_validate({'method_id': 'attribution.integrated_gradients',
 'variant_id': 'input_path_v1',
 'descriptor_version': '0.2.0',
 'status': 'implemented',
 'implementation_version': 'qwen_visual_embedding_path_v1',
 'perspective': 'general',
 'family': 'output_attribution',
 'question': 'What attribution is assigned along a declared input-baseline path for this target?',
 'evidence_kind': 'output_attribution',
 'quantity': 'integrated_gradient_attribution',
 'definition': 'Path-integrated input gradients times input-minus-baseline, with explicit target and '
               'numerical approximation, computed in float32: (x − x′)·(1/m)·Σ_k ∂F(x′ + α_k(x − x′))/∂x with α_k = (k + ½)/m. '
               'F is the log-probability of a saved capture\'s predicted token at one step under teacher '
               'forcing; x are the consumed visual-token embeddings entering the language model and x′ the '
               'vision encoder\'s output for a constant black or white image of the consumed size; per-token '
               'values sum the embedding dimensions.',
 'source_references': ['https://arxiv.org/abs/1703.01365',
                       'paper-workspace/research/interpretability methods/016-axiomatic-attribution-for-deep-networks.pdf'],
 'requires': ['generator_prediction_row_capture', 'experiment_outcomes'],
 'execution_requirement': 'model_execution',
 'target_requirement': 'Explicit differentiable score/log-probability target',
 'parameters_schema': parameter_schema(IGParameters),
 'output': {'schema': 'input_attribution',
            'units': 'target-dependent attribution',
            'signed': True,
            'support': 'declared differentiable input axes',
            'normalization': 'baseline/path-dependent'},
 'displays': ['patch_grid', 'element_table'],
 'comparison_constraints': ['Same output target, baseline, path and approximation.'],
 'limitations': ['Not universal model support; token IDs are not differentiable inputs.',
                 'Output attribution for the declared target and baseline, not attention; a different '
                 'baseline or target gives a different attribution.',
                 'Forward and backward passes run in float32 on the checkpoint weights upcast exactly from '
                 'BF16; BF16 arithmetic cannot resolve the path integral. Logits can differ slightly from the '
                 'BF16 capture; the target is re-checked as the float32 greedy token.',
                 'The path integrand can oscillate strongly for vision-language models; completeness and the '
                 'interleaved half-rule disagreement are reported with every result.',
                 'Requires free GPU memory for float32 weights (about 4 bytes per parameter plus activations); '
                 'larger generators are refused.',
                 'Pinned built-in Qwen2-VL and Qwen2.5-VL generators with image inputs only; text and earlier '
                 'output tokens are held fixed. DeepStack generators (Qwen3-VL) are refused because their '
                 'visual features also enter later decoder layers.',
                 'The vision encoder is outside the integral: attribution is to the visual-token embeddings it '
                 'produced, not to pixels. The pixel path did not converge on a tested capture by 512 steps.'],
 'validation_evidence': ['library/tests/methods/test_planned_methods.py: midpoint rule exactness on '
                         'quadratic and completeness on non-separable fixtures; aggregation and '
                         'experiment-lineage checks. Real-capture completeness is reported per result.',
                         'scripts/completed_methods_check.py on saved single-page captures (RTX 4090, '
                         '128 steps, black baseline): Qwen2.5-VL-3B step 0 relative completeness gap 0.36% with '
                         'half rules 1.932/1.925 nats and the embedding forward equal to the native pixel forward; '
                         'Qwen2-VL-2B step 0 gap 0.0013%; Qwen2.5-VL-3B teacher-forced step 1 gap 1.3% with a '
                         '5.6% half-rule disagreement, which the result flags. Qwen3-VL is refused.'],
 'name': 'Integrated gradients attribution',
 'summary': 'Attribute a specified model output to differentiable inputs relative to a declared '
            'baseline and integration path. This is output attribution, not attention.'})


def integrate_with_check(gradient_at, inputs, baseline, steps, interrupted=lambda: False):
    """Midpoint Riemann integrated gradients plus a convergence check, accumulated in float64.

    The m midpoints split into two interleaved m/2-point rules (even and odd k) whose average is
    the full rule. Their attribution totals are returned: a large disagreement means the
    integrand oscillates faster than the step count resolves.
    """
    x = np.asarray(inputs, dtype=np.float64)
    reference = np.asarray(baseline, dtype=np.float64)
    if x.shape != reference.shape or not np.isfinite(x).all() or not np.isfinite(reference).all():
        raise ValueError("Input and baseline must be finite arrays with identical shape")
    if not isinstance(steps, int) or steps < 2:
        raise ValueError("Integrated gradients needs at least two Riemann steps")
    halves = [np.zeros_like(x), np.zeros_like(x)]
    for k in range(steps):
        if interrupted():
            raise RuntimeError("Integrated gradients interrupted")
        gradient = np.asarray(gradient_at(reference + (k + .5)/steps*(x - reference)), dtype=np.float64)
        if gradient.shape != x.shape or not np.isfinite(gradient).all():
            raise ValueError("Gradient differs from the input axes or is not finite")
        halves[k % 2] += gradient
    difference = x - reference
    attribution = difference*(halves[0] + halves[1])/steps
    check = [float((difference*h).sum()/(steps/2)) for h in halves] if steps % 2 == 0 else None
    return attribution, check


def riemann_integrate(gradient_at, inputs, baseline, steps, interrupted=lambda: False):
    """Midpoint Riemann integrated gradients for any gradient callback."""
    return integrate_with_check(gradient_at, inputs, baseline, steps, interrupted)[0]


def token_totals(attribution):
    """Per visual token: the sum of its attribution over embedding dimensions."""
    values = np.asarray(attribution, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Attribution must be one embedding vector per visual token")
    return values.sum(axis=1)


def outcome_for(context):
    study = context.experiment
    if study is None or getattr(study, "method_id", None) != DESCRIPTOR.method_id or getattr(study, "kind", None) != "result":
        raise ValueError("Run and select a saved integrated-gradients experiment")
    outcome = next((o for o in study.outcomes if o.capture_id == context.record.capture_id), None)
    if outcome is None or outcome.attribution is None:
        raise ValueError("The experiment holds no attribution for this capture")
    if len(outcome.attribution) != len(context.record.context.input_token_ids):
        raise ValueError("Attribution length differs from the capture's input positions")
    return study, outcome


def calculate(context, parameters):
    study, outcome = outcome_for(context)
    if study.parameters.get("step") != parameters.step:
        raise ValueError("The selected step differs from the experiment's attributed step")
    record = context.record
    values = outcome.attribution
    positions = record.positions[:len(values)]
    image = [p for p in positions if p.role == "image" and p.bbox]
    if any(values[p.index] is None for p in image) or any(v is not None for p, v in zip(positions, values) if not (p.role == "image" and p.bbox)):
        raise ValueError("Attribution must cover exactly the consumed image positions")
    total = float(sum(values[p.index] for p in image))
    difference = outcome.target_log_probability - outcome.baseline_log_probability
    gap = total - difference
    relative = abs(gap)/abs(difference) if difference else None
    elements = {}
    for p in image:
        for bucket, fraction in p.ownership.items():
            elements[bucket, p.page_id] = elements.get((bucket, p.page_id), 0.0) + values[p.index]*fraction
    ranked = sorted(elements.items(), key=lambda item: -abs(item[1]))
    pages = {page.page_id: page for page in record.context.pages}
    plots = []
    for page_id, page in pages.items():
        cells = [p for p in image if p.page_id == page_id]
        if cells:
            plots.append(ResearchPlot(kind="grid", title=f"Integrated gradients · page {page.page_number} · signed visual-token attribution",
                x_label="Page x (fraction of width)", y_label="Page y (fraction of height)",
                points=[ResearchPoint(label=f"Patch {p.index}", x=(p.bbox[0]+p.bbox[2])/2/page.source_width,
                                      y=(p.bbox[1]+p.bbox[3])/2/page.source_height, value=values[p.index], page_id=page_id) for p in cells]))
    target = next((s for s in record.steps if s.step == parameters.step), None)
    piece = target.predicted_piece if target else outcome.target_token_id
    # The float32 attribution forward can rank tokens differently from the BF16 capture; never leave that silent.
    not_greedy = ([f"The explained target “{piece}” is not the float32 forward's greedy prediction at this step "
                   f"(target log-probability {outcome.target_log_probability:.4g} nats, probability {float(np.exp(outcome.target_log_probability)):.3g}). "
                   "The attribution explains this token, not the token the model would choose."]
                  if getattr(outcome, "target_is_greedy", None) is False else [])
    return Calculation(
        statistics=[statistic("Target log-probability F(x)", outcome.target_log_probability, "nats"),
                    statistic("Baseline log-probability F(x′)", outcome.baseline_log_probability, "nats"),
                    statistic("F(x) − F(x′)", difference, "nats"),
                    statistic("Sum of patch attributions", total, "nats"),
                    statistic("Completeness gap", gap, "nats"),
                    statistic("Relative completeness gap", relative),
                    statistic("Positive attribution", sum(max(values[p.index], 0) for p in image), "nats"),
                    statistic("Negative attribution", sum(min(values[p.index], 0) for p in image), "nats"),
                    statistic("Riemann steps", study.parameters.get("steps"), "steps"),
                    statistic("Interleaved half-rule disagreement", abs(outcome.half_rule_totals[0]-outcome.half_rule_totals[1]) if outcome.half_rule_totals else None, "nats")],
        table=ResultTable(title="Attribution by element and residual bucket", columns=["Attribution", "Share of |total|"],
                          labels=[bucket for (bucket, _), _ in ranked],
                          values=[[value, abs(value)/sum(abs(v) for _, v in ranked) if ranked else None] for _, value in ranked],
                          units="nats; signed", element_ids=[bucket if not bucket.startswith(("background", "unreviewed", "non_")) else None for (bucket, _), _ in ranked],
                          page_ids=[page for (_, page), _ in ranked]),
        plots=plots,
        normalization=f"attributions sum toward F(x) − F(x′); {study.parameters.get('baseline')} baseline; midpoint Riemann sum",
        notices=[f"Target: log-probability of “{piece}” at prediction step {parameters.step}, with the prompt and earlier output fixed.",
                 *not_greedy,
                 *([f"Completeness gap is {relative:.1%} of F(x) − F(x′); the integral has not converged at this step count. Increase Riemann steps or treat values as unreliable."] if relative is not None and relative > COMPLETENESS_TOLERANCE else []),
                 *([f"The two interleaved half-step rules disagree by {abs(outcome.half_rule_totals[0]-outcome.half_rule_totals[1]):.3g} nats: the path integrand oscillates faster than the steps resolve."] if outcome.half_rule_totals and difference and abs(outcome.half_rule_totals[0]-outcome.half_rule_totals[1]) > COMPLETENESS_TOLERANCE*abs(difference) else []),
                 "Output attribution for this target and baseline; it is not attention and does not establish causation."])


DEFINITION = MethodDefinition(DESCRIPTOR, IGParameters, calculate, True)
