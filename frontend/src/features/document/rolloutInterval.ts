export const isRolloutInterval = (methodId?: string) => methodId === 'attention.rollout' || methodId === 'attention.partial_rollout';

/** Reject invalid saved selections before requesting a calculation; never change them on load. */
export function rolloutIntervalGap(methodId: string | undefined, layerCount: number, end: number, start: unknown = 0) {
  if (!isRolloutInterval(methodId)) return undefined;
  const first = methodId === 'attention.partial_rollout' ? start : 0;
  if (!Number.isInteger(first) || !Number.isInteger(end) || (first as number) < 0 || end < 0 || (first as number) >= layerCount || end >= layerCount)
    return `Choose captured layers from 0 to ${layerCount - 1}. The saved interval ${String(first)}–${end} is outside this capture.`;
  if ((first as number) > end) return `Start layer ${first} follows end layer ${end}. Choose a valid layer interval.`;
  return undefined;
}
