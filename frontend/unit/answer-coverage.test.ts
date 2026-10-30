import { describe, expect, it } from 'vitest';
import type { components } from '../src/api/generated/schema';
import { answerTokenCoverage } from '../src/features/document/CaptureMenu';
import { answerRowCoverage, captureSummary } from '../src/features/attention-tool/ModelCapabilities';

type Model = components['schemas']['ModelRegistration'];
const steps = (n: number) => Array.from({length: n}, (_, step) => ({step})) as unknown as components['schemas']['PredictionStep'][];
const capture = (kept: number, tokens: number) => ({steps: steps(kept), output_token_ids: Array.from({length: tokens}, (_, i) => i)});
// The catalogue's limits for the Qwen generators.
const LIMITS = {captured_steps: 8, max_new_tokens: 64, full_attention_captured_steps: 64};
const model = (limits: Record<string, number>): Model => ({registration_id: 'builtin', role: 'generation', name: 'Qwen', model_id: 'q', revision: 'r', validation: '',
  architecture: {model_type: 'qwen', architectures: [], components: [{name: 'Language', layers: 36, query_heads: 16}], embedding_dim: null, dtype: 'bfloat16'},
  capabilities: {can_run: true, attention_available: true, signal_description: '', input_description: '', summary: '', reason: '', limits}} as unknown as Model);

describe('answer-token coverage', () => {
  it('names a full-attention capture that kept every answer token, or its first N', () => {
    expect(answerTokenCoverage(capture(12, 12))).toBe('every answer token');
    expect(answerTokenCoverage(capture(64, 64))).toBe('every answer token');
    // Full-attention captures made before every decode row was kept (generation schema 0.7.0) kept eight.
    expect(answerTokenCoverage(capture(8, 36))).toBe('first 8 answer tokens');
    expect(answerTokenCoverage(capture(1, 2))).toBe('first 1 answer token');
  });

  it('reads the ordinary and full-attention limits instead of a fixed eight', () => {
    expect(answerRowCoverage(LIMITS)).toBe('Attention rows for the first 8 answer tokens (prediction steps)');
    expect(answerRowCoverage({...LIMITS, captured_steps: 4})).toBe('Attention rows for the first 4 answer tokens (prediction steps)');
    expect(answerRowCoverage(LIMITS, true, 64)).toBe("Every answer token's attention row (up to 64)");
    // A full-attention limit below the output limit is stated as such.
    expect(answerRowCoverage(LIMITS, true, 32)).toBe('Attention rows for the first 32 answer tokens');
    expect(answerRowCoverage({})).toBeUndefined();
  });

  it('follows the option in the setup summary', () => {
    expect(captureSummary(model(LIMITS))).toMatch(/^Attention rows for the first 8 answer tokens \(prediction steps\) · all 36 layers \/ 16 query heads · maximum 64 output tokens/);
    expect(captureSummary(model(LIMITS), {fullAttention: true})).toMatch(/^Every answer token's attention row \(up to 64\) · /);
    // A catalogue without the full-attention field falls back to the output limit.
    expect(captureSummary(model({captured_steps: 8, max_new_tokens: 64}), {fullAttention: true})).toMatch(/^Every answer token's attention row \(up to 64\)/);
  });
});
