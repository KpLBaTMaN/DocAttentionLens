import { describe, expect, it } from 'vitest';
import { rolloutIntervalGap } from '../src/features/document/rolloutInterval';

describe('captured rollout layer intervals', () => {
  it('accepts full and single-layer intervals at the actual capture bounds', () => {
    expect(rolloutIntervalGap('attention.partial_rollout',36,35,0)).toBeUndefined();
    expect(rolloutIntervalGap('attention.partial_rollout',36,35,35)).toBeUndefined();
    expect(rolloutIntervalGap('attention.partial_rollout',1,0,0)).toBeUndefined();
    expect(rolloutIntervalGap('attention.rollout',36,35)).toBeUndefined();
  });
  it('refuses bad saved indices and reversed intervals without changing them', () => {
    expect(rolloutIntervalGap('attention.partial_rollout',36,20,127)).toContain('127–20');
    expect(rolloutIntervalGap('attention.partial_rollout',18,18,0)).toContain('0 to 17');
    expect(rolloutIntervalGap('attention.partial_rollout',36,10,20)).toContain('follows');
    expect(rolloutIntervalGap('attention.partial_rollout',36,10,1.5)).toContain('outside');
    expect(rolloutIntervalGap('attention.partial_rollout',36,10,'2')).toContain('outside');
  });
  it('does not apply the two-endpoint rule to unrelated method parameters', () => {
    expect(rolloutIntervalGap('attention.entropy',36,10,20)).toBeUndefined();
  });
});
