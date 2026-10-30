import type { ReactNode } from 'react';
import type { DisplaySettings } from '../../api/client';
import { ATTENTION_MODES, COLORMAPS, mapGradient, type AttentionMode } from './attentionDisplay';
import { scaleOptions } from './AttentionToolbar';

type Style = Pick<DisplaySettings, 'attention_display_mode' | 'attention_colormap' | 'attention_range_floor' | 'attention_opacity' | 'attention_smooth' | 'attention_show_patches'>;

/** Default: amber overlay from zero. Glow: the colour map blended over the whole page, stretched from its minimum, soft and gridded. */
const PRESETS: {id: string; label: string; hint: string; style: Style}[] = [
  {id: 'default', label: 'Default', hint: 'Amber overlay; transparent where attention is weak', style: {attention_display_mode: 'overlay', attention_colormap: 'inferno',
    attention_range_floor: 'zero', attention_opacity: .65, attention_smooth: false, attention_show_patches: false}},
  {id: 'glow', label: 'Glow', hint: 'Colour map over the whole page: weak areas darken, strong areas glow', style: {attention_display_mode: 'blend', attention_colormap: 'inferno',
    attention_range_floor: 'minimum', attention_opacity: .45, attention_smooth: true, attention_show_patches: true}},
];
const OVERLAYS = [['show_boxes', 'Layout boxes'], ['show_reading_order', 'Reading order'], ['show_geometry_flags', 'Geometry flags'], ['show_support', 'Designated support']] as const;
/** A table grounding page has a table, not layout elements: the same settings draw its cell grid and the question's lookup path. */
const TABLE_OVERLAYS = [['show_boxes', 'Table cell grid'], ['show_support', 'Question grounding (evaluation only)']] as const;

/** Details › Display shows either its settings or the colour scale readout. */
export type DisplayView = 'settings' | 'scale';
export const isDisplayView = (value: unknown): value is DisplayView => value === 'settings' || value === 'scale';
const VIEWS: {id: DisplayView; label: string; title: string}[] = [
  {id: 'settings', label: 'Settings', title: 'Colour mode, opacity, colour map, scale, range, detail and page overlays'},
  {id: 'scale', label: 'Scale', title: 'What each colour means, where the row mass goes and the row totals'}];

const matches = (settings: DisplaySettings, style: Style) =>
  (Object.keys(style) as (keyof Style)[]).every(k => (settings[k] ?? PRESETS[0].style[k]) === style[k]);

/** Details › Display: every display setting and the colour scale readout. Display only. */
export function DisplaySettingsTab({settings, hasAttention, encoding = false, readout, pageColouring, inputView, onMode, onView, view = 'settings', onDisplayView, structure = 'layout'}: {
  settings: DisplaySettings; hasAttention: boolean; encoding?: boolean;
  /** What the page's annotations describe: layout elements or a table. */
  structure?: 'layout' | 'table';
  /** The colour scales, image and non-image shares and row totals of the shown row. */
  readout?: ReactNode;
  /** Element totals or patches, when the method has element rows. */
  pageColouring?: ReactNode;
  /** Original, consumed or both model inputs (generator captures with a consumed image). */
  inputView?: ReactNode;
  /** Changing the mode also turns a hidden attention overlay back on. */
  onMode?: (mode: AttentionMode) => void;
  onView: (next: DisplaySettings) => void;
  /** Which view is shown; the choice is the viewer's, not saved with the investigation. */
  view?: DisplayView; onDisplayView?: (view: DisplayView) => void;
}) {
  const mode = settings.attention_display_mode ?? 'overlay';
  const colormap = settings.attention_colormap ?? 'inferno';
  const opacity = Math.round((settings.attention_opacity ?? .65) * 100);
  const scale = settings.attention_scale_mode ?? 'auto';
  const floor = settings.attention_range_floor ?? 'zero';
  const preset = PRESETS.find(p => matches(settings, p.style));
  const set = (next: Partial<DisplaySettings>) => onView({...settings, ...next});
  const switcher = onDisplayView && <div className="docattentionlens-display-views" role="tablist" aria-label="Display view" onKeyDown={e => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
    e.preventDefault();
    const next = VIEWS.find(v => v.id !== view)!.id;
    onDisplayView(next);
    requestAnimationFrame(() => document.getElementById(`display-view-${next}`)?.focus());
  }}>
    {VIEWS.map(v => <button key={v.id} type="button" role="tab" id={`display-view-${v.id}`} aria-selected={view === v.id} aria-controls="display-view-panel"
      tabIndex={view === v.id ? 0 : -1} title={v.title} onClick={() => onDisplayView(v.id)}>{v.label}</button>)}
  </div>;
  if (view === 'scale') return <div className="docattentionlens-display-tab" data-testid="display-settings" data-view="scale">
    {switcher}
    <div id="display-view-panel" role="tabpanel" aria-labelledby="display-view-scale" className="docattentionlens-display-readout">
      {readout ?? <p className="docattentionlens-display-hint">No colour scale for the current view.</p>}</div>
  </div>;
  return <div className="docattentionlens-display-tab" data-testid="display-settings" data-view="settings">
    {switcher}
    <div id="display-view-panel" role="tabpanel" aria-labelledby={switcher ? 'display-view-settings' : undefined} className="docattentionlens-display-settings">
    {hasAttention && <section className="docattentionlens-display-group" aria-label="Look">
      <h3>Look</h3>
      <div className="docattentionlens-reading-buttons is-pair" role="group" aria-label="Style presets">
        {PRESETS.map(p => <button key={p.id} type="button" className="docattentionlens-reading-button" title={p.hint} aria-pressed={preset?.id === p.id}
          onClick={() => set(p.style)}>{p.label}</button>)}
      </div>
      <p className="docattentionlens-display-hint">{preset ? preset.hint : 'Custom: your own combination of the settings below.'}</p>
      <h4>Colour mode <kbd className="docattentionlens-display-key" title="Press V to cycle the colour mode">V</kbd></h4>
      <div className="docattentionlens-reading-buttons is-four" role="radiogroup" aria-label="Colour mode">
        {ATTENTION_MODES.map(m => <button key={m.id} type="button" role="radio" className="docattentionlens-reading-button" aria-checked={mode === m.id} aria-pressed={mode === m.id}
          onClick={() => onMode ? onMode(m.id) : set({attention_display_mode: m.id})}>{m.label}</button>)}
      </div>
      <h4>Opacity <small>Overlay and Blend</small></h4>
      <label className="docattentionlens-display-slider"><input type="range" aria-label="Attention opacity" min={0} max={100} step={1} value={opacity}
        onChange={e => set({attention_opacity: Number(e.target.value) / 100})}/><output>{opacity}%</output></label>
    </section>}
    <section className="docattentionlens-display-group" aria-label="Colour and scale">
      <h3>Colour and scale</h3>
      {hasAttention ? <>
        <h4>Colour map <small>Overlay stays amber</small></h4>
        <div className="docattentionlens-colormaps" role="radiogroup" aria-label="Colour map">
          {COLORMAPS.map(c => <button key={c.id} type="button" role="radio" aria-checked={colormap === c.id} className="docattentionlens-colormap" onClick={() => set({attention_colormap: c.id})}>
            <i style={{background: mapGradient(c.id)}} aria-hidden="true"/>{c.label}</button>)}
        </div>
        <h4>Scale <small>What the brightest colour means</small></h4>
        <select aria-label="Heatmap scale" value={scale} onChange={e => set({attention_scale_mode: e.target.value as DisplaySettings['attention_scale_mode']})}>
          {scaleOptions(encoding).map(o => <option key={o.id} value={o.id}>{o.label}</option>)}
        </select>
        {scale === 'fixed' && <label className="docattentionlens-display-slider">Maximum<input type="number" aria-label="Fixed scale maximum" min={0.00000001} max={100000} step="any" value={settings.attention_scale_max ?? 1}
          onChange={e => {const value = e.target.valueAsNumber; if (Number.isFinite(value) && value > 0 && value <= 100000) set({attention_scale_max: value});}}/></label>}
        <p className="docattentionlens-display-hint">{scale === 'auto' ? 'Stretches to this page’s own peak, so weak pages still show contrast.' : scale === 'causal' ? `The largest weight in the whole ${encoding ? 'encoding' : 'causal'} row, text tokens included, is the brightest colour, so the page can look dim.` : 'The maximum you set stays the brightest colour across pages and steps.'}</p>
        <h4>Range</h4>
        <div className="docattentionlens-reading-buttons is-pair" role="radiogroup" aria-label="Range start">
          <button type="button" role="radio" className="docattentionlens-reading-button" aria-checked={floor === 'zero'} aria-pressed={floor === 'zero'}
            onClick={() => set({attention_range_floor: 'zero'})}>Start at 0</button>
          <button type="button" role="radio" className="docattentionlens-reading-button" aria-checked={floor === 'minimum'} aria-pressed={floor === 'minimum'}
            onClick={() => set({attention_range_floor: 'minimum'})}>Page minimum → max</button>
        </div>
      </> : <p className="docattentionlens-display-hint">MaxSim similarity opacity and scale are in Details › Result.</p>}
      {onDisplayView ? <p className="docattentionlens-display-hint">What each colour means and where the row mass goes: <button type="button" className="docattentionlens-link-button" onClick={() => onDisplayView('scale')}>Scale</button></p>
        : <div className="docattentionlens-display-readout">{readout ?? <p className="docattentionlens-display-hint">No colour scale for the current view.</p>}</div>}
    </section>
    {hasAttention && <section className="docattentionlens-display-group" aria-label="Detail">
      <h3>Detail</h3>
      <div className="docattentionlens-display-checks">
        <label className="docattentionlens-checkbox" title="Interpolate between patch centres (display only)"><input type="checkbox" checked={settings.attention_smooth ?? false} onChange={e => set({attention_smooth: e.target.checked})}/>Smooth</label>
        <label className="docattentionlens-checkbox"><input type="checkbox" checked={settings.attention_show_patches ?? false} onChange={e => set({attention_show_patches: e.target.checked})}/>Patch grid</label>
      </div>
    </section>}
    <section className="docattentionlens-display-group" aria-label="Page">
      <h3>Page</h3>
      {inputView && <><h4>Input view <small>What the model consumed</small></h4>{inputView}</>}
      {pageColouring && <div className="docattentionlens-display-page-colouring"><h4>Page colouring</h4>{pageColouring}</div>}
      <h4>Document overlays</h4>
      <div className="docattentionlens-display-checks" role="group" aria-label="Document overlays">
        {(structure === 'table' ? TABLE_OVERLAYS : OVERLAYS).map(([key, label]) => <label className="docattentionlens-checkbox" key={key}><input type="checkbox" checked={settings[key] ?? (key === 'show_boxes')} onChange={e => set({[key]: e.target.checked})}/>{label}</label>)}
      </div>
      <p className="docattentionlens-display-hint">{structure === 'table'
        ? 'Purple: the table, and dashed its caption and notes · gold: answer cell · orange dashed: row header path · violet dashed: column header path · brown dotted: support cells.'
        : 'Red: selection · cyan dashed: inspection source · purple dotted: inspection target · amber dashed: flagged geometry · green: designated support.'}</p>
    </section>
    <p className="docattentionlens-display-hint docattentionlens-display-footer">Changes preview live and are saved with the investigation. Stored attention values never change.</p>
    </div>
  </div>;
}
