import { useEffect, useState } from 'react';

/**
 * A record identifier shown as secondary information: muted, monospace and
 * truncated, with the full value in `title` and a copy button beside it.
 */
export function RecordId({value, label = 'ID'}: {value: string; label?: string}) {
  const [copied, setCopied] = useState<'done' | 'failed'>();
  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(undefined), 1600);
    return () => window.clearTimeout(timer);
  }, [copied]);
  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopied('done');
    } catch {
      setCopied('failed');
    }
  }
  return <span className="docattentionlens-record-id">
    <code title={value}>{value}</code>
    <button type="button" className="docattentionlens-record-copy" aria-label={`Copy ${label} ${value}`} onClick={() => void copy()}>{copied === 'done' ? 'Copied' : copied === 'failed' ? 'Copy failed' : 'Copy'}</button>
  </span>;
}
