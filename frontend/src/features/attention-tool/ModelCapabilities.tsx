import type { components } from '../../api/generated/schema';
import { fullAttentionSteps } from './fullAttentionEstimate';

export type Model = components['schemas']['ModelRegistration'];

/** Which answer tokens keep their attention row; undefined for a model without answer rows (a retriever). */
export function answerRowCoverage(limits: Record<string, number | undefined>, fullAttention = false, fullSteps?: number) {
  const ordinary = limits.captured_steps, maxTokens = limits.max_new_tokens;
  if (fullAttention && fullSteps) return maxTokens === undefined || fullSteps >= maxTokens ? `Every answer token's attention row (up to ${fullSteps})` : `Attention rows for the first ${fullSteps} answer tokens`;
  return ordinary ? `Attention rows for the first ${ordinary} answer tokens (prediction steps)` : undefined;
}

export function captureSummary(model?: Model, {fullAttention = false}: {fullAttention?: boolean} = {}) {
  if (!model) return 'Choose a model to see capture coverage.';
  const limits=model.capabilities.limits??{};
  const text=model.architecture.components?.find(c=>c.name==='Language');
  const parts=[];
  const answerRows=answerRowCoverage(limits,fullAttention,fullAttentionSteps(model));
  if (answerRows) parts.push(answerRows);
  if (model.capabilities.attention_available&&text?.layers&&text.query_heads) parts.push(`all ${text.layers} layers / ${text.query_heads} query heads`);
  if (limits.max_new_tokens) parts.push(`maximum ${limits.max_new_tokens} output tokens`);
  if (limits.attention_page_limit) parts.push(`query encoding + ${limits.attention_page_limit} document encoding`);
  if (limits.candidate_limit) parts.push(`up to ${limits.candidate_limit} candidate pages`);
  if (!model.capabilities.attention_available) parts.push(model.capabilities.signal_description);
  return parts.join(' · ');
}

export function ModelCapabilities({model,fullAttention=false}:{model?:Model;fullAttention?:boolean}) {
  if (!model) return null;
  // The summary line and any launch blocker stay visible; the full description is one click away.
  return <section className="docattentionlens-setup-capability" aria-label="Selected model details">
    <strong>{model.capabilities.signal_description}</strong>
    <p data-testid="capture-summary">{captureSummary(model,{fullAttention})}</p>
    {!model.capabilities.can_run&&<p role="status">{model.capabilities.reason}</p>}
    <details><summary>Model details</summary>
    <p>{model.capabilities.input_description}</p>
    <h3>Architecture and dimensions</h3><dl className="docattentionlens-model-metadata">
      {model.architecture.components?.map(component=><div key={component.name}>
        <dt>{component.name}</dt><dd>{[
          component.layers?`${component.layers} layers`:undefined,
          component.query_heads?`${component.query_heads} ${component.name==='Language'?'query ':''}heads`:undefined,
          component.kv_heads?`${component.kv_heads} KV head${component.kv_heads===1?'':'s'}`:undefined,
          component.patch_size?`${component.patch_size} px patches`:undefined,
          component.spatial_merge_size?`${component.spatial_merge_size} × ${component.spatial_merge_size} spatial merge`:undefined,
          component.image_size?`${component.image_size} × ${component.image_size} input`:undefined,
        ].filter(Boolean).join(' · ')}</dd>
      </div>)}
      {model.architecture.embedding_dim&&<div><dt>Retrieval vectors</dt><dd>{model.architecture.embedding_dim} dimensions</dd></div>}
    </dl>
    {!model.architecture.components?.length&&<p>Layer, head and patch dimensions were not declared in this checkpoint’s configuration.</p>}
    <p><strong>Available evidence</strong><br/>{model.capabilities.summary}</p>
    <small>{model.validation}</small>
    <h3>Model identity</h3><p>{model.model_id} · {model.architecture.model_type}</p><p>Revision: {model.revision}</p>
    </details>
  </section>;
}
