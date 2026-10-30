import type { DisplaySettings } from '../../api/client';
import type { View, UpdateView } from './useAcquisition';

/** MaxSim similarity display. Attention display controls sit above the page (AttentionToolbar). */
export function SignalDisplay({enabled, state, settings, update, onView}: {
  enabled: boolean; state: View; settings: DisplaySettings;
  update: UpdateView; onView: (next: DisplaySettings) => void;
}) {
  const scale = settings.retrieval_scale_mode ?? 'auto';
  const opacity = settings.similarity_opacity ?? .65;
  return <section aria-label="Similarity display"><h2>Similarity display</h2>
    <fieldset className="docattentionlens-signal-controls" disabled={!enabled}>
      <label className="docattentionlens-checkbox"><input type="checkbox" aria-label="Similarity overlay" checked={state.similarity_overlay !== false}
        onChange={e => update(s => ({...s, similarity_overlay: e.target.checked}))}/>Similarity overlay</label>
      <label className="docattentionlens-opacity-control"><span>Opacity <output>{Math.round(opacity * 100)}%</output></span><input aria-label="Similarity opacity" type="range" min={0} max={100} step={1} value={Math.round(opacity * 100)}
        onChange={e => onView({...settings, similarity_opacity: Number(e.target.value) / 100})}/></label>
      <label>Heatmap scale<select aria-label="Similarity scale" value={scale} onChange={e => onView({...settings, retrieval_scale_mode: e.target.value as DisplaySettings['retrieval_scale_mode']})}>
        <option value="auto">Automatic per row</option><option value="fixed">Fixed</option>
      </select></label>
      {scale === 'fixed' && <label>Fixed maximum<input type="number" aria-label="Fixed similarity scale maximum" min={0.00000001} max={100000} step="any" value={settings.retrieval_scale_max ?? 1}
        onChange={e => {const value = e.target.valueAsNumber; if (Number.isFinite(value) && value > 0 && value <= 100000) onView({...settings, retrieval_scale_max: value});}}/></label>}
    </fieldset>
    <p>Display only: opacity and scale preserve captured values and normalization.</p>
  </section>;
}
