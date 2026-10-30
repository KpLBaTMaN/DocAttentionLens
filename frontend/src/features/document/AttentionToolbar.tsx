import { useRef, type ReactNode } from 'react';
import type { DisplaySettings } from '../../api/client';
import { ATTENTION_MODES, type AttentionMode } from './attentionDisplay';
import { useMenuDismiss } from './WorkspaceControls';

/** Scale choices, shared with Display settings; an encoder's row is its encoding row, not a causal one. */
export const scaleOptions = (encoding: boolean) => [
  {id: 'auto', label: 'Image contrast'}, {id: 'causal', label: encoding ? 'Full encoding row' : 'Full causal row'}, {id: 'fixed', label: 'Fixed'}] as const;

/** Display-only attention controls for comparison maps; captured values and normalization never change. */
export function AttentionToolbar({mode, settings, enabled, encoding = false, onMode, onView, children}: {
  mode: AttentionMode; settings: DisplaySettings; enabled: boolean; encoding?: boolean;
  onMode: (mode: AttentionMode) => void; onView: (next: DisplaySettings) => void;
  /** Shown after the controls, such as the page colour key. */
  children?: ReactNode;
}) {
  const scale = settings.attention_scale_mode ?? 'auto';
  const opacity = Math.round((settings.attention_opacity ?? .65) * 100);
  const more = useRef<HTMLDetailsElement>(null);
  const dismiss = useMenuDismiss(more);
  return <fieldset className="docattentionlens-view-toolbar" aria-label="Attention display" disabled={!enabled}>
    <div className="docattentionlens-segmented" role="radiogroup" aria-label="Attention display mode">
      {ATTENTION_MODES.map(m => <button key={m.id} type="button" role="radio" aria-checked={mode === m.id} onClick={() => onMode(m.id)}>{m.label}</button>)}
    </div>
    {(mode === 'overlay' || mode === 'blend') && <label className="docattentionlens-toggle">Opacity<input aria-label="Attention opacity" type="range" min={0} max={100} step={1} value={opacity}
      onChange={e => onView({...settings, attention_opacity: Number(e.target.value) / 100})}/><output>{opacity}%</output></label>}
    <label className="docattentionlens-toggle">Scale<select aria-label="Heatmap scale" value={scale} onChange={e => onView({...settings, attention_scale_mode: e.target.value as DisplaySettings['attention_scale_mode']})}>
      {scaleOptions(encoding).map(o => <option key={o.id} value={o.id}>{o.label}</option>)}
    </select></label>
    {scale === 'fixed' && <label className="docattentionlens-toggle">Maximum<input type="number" aria-label="Fixed scale maximum" min={0.00000001} max={100000} step="any" value={settings.attention_scale_max ?? 1}
      onChange={e => {const value = e.target.valueAsNumber; if (Number.isFinite(value) && value > 0 && value <= 100000) onView({...settings, attention_scale_max: value});}}/></label>}
    <details ref={more} className="docattentionlens-display-more" {...dismiss}>
      <summary aria-label="More display options" title="More display options"><span aria-hidden="true">⚙</span></summary>
      <div>
        <label className="docattentionlens-toggle"><input type="checkbox" checked={settings.attention_show_patches ?? false} onChange={e => onView({...settings, attention_show_patches: e.target.checked})}/>Patches</label>
        <label className="docattentionlens-toggle" title="Interpolate between patch centres instead of drawing square patches"><input type="checkbox" checked={settings.attention_smooth ?? false} onChange={e => onView({...settings, attention_smooth: e.target.checked})}/>Smooth</label>
      </div>
    </details>
    {children}
  </fieldset>;
}
