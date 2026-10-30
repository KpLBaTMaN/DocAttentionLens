import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from 'react';
import { rolloutIntervalGap } from './rolloutInterval';
import '../../styles/layer-scrubber.css';

/** Optional "All" head cell for methods that support a mean over heads. */
export type AllHeads = {active: boolean; onChange: (active: boolean) => void; label?: string};

/** Two endpoints on the layer track (rollout interval), following the rules of `RolloutLayerControls`. */
export type LayerRange = {
  from: unknown; to: number; onChange: (from: number, to: number) => void;
  fixedStart?: boolean;
  /** Full rollout only: offers "Choose a layer range" (switching to partial rollout). */
  onChooseRange?: () => void;
};

export type LayerScrubberProps = {
  layer?: number; layers?: number; onLayer?: (layer: number) => void;
  head?: number; heads?: number; onHead?: (head: number) => void;
  /** Optional per-layer bar heights 0..1; the track is flat when absent. */
  bars?: number[];
  /** What the bars show (for example the selected element's layer means); the track's tooltip. */
  barsTitle?: string;
  allHeads?: AllHeads;
  range?: LayerRange;
  showLayer?: boolean; showHead?: boolean;
  /** `true` disables; a string disables and is shown as the reason. */
  disabled?: boolean | string;
  className?: string;
};

/** More head cells than this collapse to a compact select and slider. */
export const HEAD_CELL_LIMIT = 24;

const clamp = (n: number, count: number) => Math.max(0, Math.min(count - 1, n));
const pageStep = (count: number) => Math.max(1, Math.round(count / 10));

/** The index a slider key moves to, or undefined for keys a slider does not handle. */
export function keyIndex(key: string, value: number, count: number): number | undefined {
  switch (key) {
    case 'ArrowLeft': case 'ArrowDown': return clamp(value - 1, count);
    case 'ArrowRight': case 'ArrowUp': return clamp(value + 1, count);
    case 'Home': return 0;
    case 'End': return count - 1;
    case 'PageDown': return clamp(value - pageStep(count), count);
    case 'PageUp': return clamp(value + pageStep(count), count);
    default: return undefined;
  }
}

/** The layer under a pointer on a track of `count` equal cells. */
export function indexAt(clientX: number, rect: {left: number; width: number}, count: number): number | undefined {
  if (!(rect.width > 0) || count < 1) return undefined;
  return clamp(Math.floor((clientX - rect.left) / rect.width * count), count);
}

const capture = (e: ReactPointerEvent<HTMLElement>) => {
  try {e.currentTarget.setPointerCapture?.(e.pointerId);} catch {/* pointer already released */}
};
const pointedAt = (e: ReactPointerEvent<HTMLElement>, track: HTMLElement | null, count: number) =>
  track ? indexAt(e.clientX, track.getBoundingClientRect(), count) : undefined;

function Bars({count, bars, title, cell}: {count: number; bars?: number[]; title?: string; cell: (i: number) => string}) {
  return <div className={`docattentionlens-scrubber-bars${bars ? ' has-values' : ''}`} aria-hidden="true" title={bars ? title : undefined}>
    {Array.from({length: count}, (_, i) => {
      const h = bars?.[i];
      const height = typeof h === 'number' && Number.isFinite(h) ? `${Math.round(20 + 80 * Math.max(0, Math.min(1, h)))}%` : undefined;
      return <span key={i} className={`docattentionlens-scrubber-bar ${cell(i)}`} style={height ? {height} : undefined}/>;
    })}
  </div>;
}

function IndexSelect({label, value, count, disabled, onChange}: {label: string; value: number; count: number; disabled: boolean; onChange: (v: number) => void}) {
  return <span className="docattentionlens-scrubber-readout">
    <select aria-label={label} value={value} disabled={disabled} onChange={e => onChange(Number(e.target.value))}>
      {Array.from({length: count}, (_, i) => <option key={i} value={i}>{i}</option>)}
    </select><span className="docattentionlens-scrubber-of">/ {count - 1}</span>
  </span>;
}

/** One layer: the whole track is the slider. Dragging previews locally and commits on release. */
function LayerTrack({layer, layers, bars, barsTitle, disabled, onLayer}: {layer: number; layers: number; bars?: number[]; barsTitle?: string; disabled: boolean; onLayer: (v: number) => void}) {
  const track = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const [draft, setDraft] = useState<number | null>(null);
  const current = clamp(layer, layers);
  const shown = draft ?? current;
  const commit = (v: number | undefined) => {if (v !== undefined && v !== layer) onLayer(v);};
  return <div className="docattentionlens-scrubber-layer">
    <span className="docattentionlens-scrubber-label" aria-hidden="true">Layer</span>
    <div ref={track} className="docattentionlens-scrubber-track" role="slider" tabIndex={disabled ? -1 : 0}
      aria-label="Attention layer slider" aria-orientation="horizontal" aria-valuemin={0} aria-valuemax={layers - 1}
      aria-valuenow={shown} aria-valuetext={`Layer ${shown} / ${layers - 1}`} aria-disabled={disabled || undefined}
      data-testid="layer-scrubber-track"
      onKeyDown={e => {
        if (disabled) return;
        const next = keyIndex(e.key, current, layers);
        if (next === undefined) return;
        e.preventDefault(); commit(next);
      }}
      onPointerDown={e => {
        if (disabled || e.button !== 0) return;
        const v = pointedAt(e, track.current, layers);
        if (v === undefined) return;
        capture(e); dragging.current = true; setDraft(v);
      }}
      onPointerMove={e => {
        if (!dragging.current) return;
        const v = pointedAt(e, track.current, layers);
        if (v !== undefined) setDraft(v);
      }}
      onPointerUp={e => {
        if (!dragging.current) return;
        dragging.current = false;
        const v = pointedAt(e, track.current, layers) ?? draft ?? undefined;
        setDraft(null); commit(v);
      }}
      onPointerCancel={() => {dragging.current = false; setDraft(null);}}>
      <Bars count={layers} bars={bars} title={barsTitle} cell={i => i === shown ? 'is-current' : i < shown ? 'is-before' : ''}/>
    </div>
    <IndexSelect label="Attention layer" value={shown} count={layers} disabled={disabled} onChange={commit}/>
  </div>;
}

type Handle = 'from' | 'to';

/** Rollout interval: two handles on one track, with the paired rules of RolloutLayerControls. */
function RangeTrack({range, layers, bars, barsTitle, disabled}: {range: LayerRange; layers: number; bars?: number[]; barsTitle?: string; disabled: boolean}) {
  const fixed = !!range.fixedStart;
  const gap = rolloutIntervalGap(fixed ? 'attention.rollout' : 'attention.partial_rollout', layers, range.to, fixed ? 0 : range.from);
  const captured = (n: unknown) => clamp(typeof n === 'number' && Number.isInteger(n) ? n : 0, layers);
  const last = captured(range.to), first = fixed ? 0 : Math.min(captured(range.from), last);
  // Moving one endpoint keeps a valid interval by moving the other one with it.
  const apply = (handle: Handle, v: number): [number, number] => handle === 'from' ? [v, Math.max(v, last)] : [fixed ? 0 : Math.min(first, v), v];
  const track = useRef<HTMLDivElement>(null);
  const dragging = useRef<Handle | null>(null);
  const [draft, setDraft] = useState<[number, number] | null>(null);
  const [a, b] = draft ?? [first, last];
  const commit = ([from, to]: [number, number]) => {if (gap || from !== first || to !== last) range.onChange(from, to);};
  const nearest = (v: number): Handle => {
    if (fixed) return 'to';
    const df = Math.abs(v - first), dt = Math.abs(v - last);
    return df < dt ? 'from' : dt < df ? 'to' : v < first ? 'from' : 'to';
  };
  const handle = (which: Handle) => {
    const value = which === 'from' ? a : b;
    const label = which === 'from' ? 'Start layer slider' : 'End layer slider';
    return <div className={`docattentionlens-scrubber-handle is-${which}`} role="slider" tabIndex={disabled ? -1 : 0} aria-label={label}
      aria-orientation="horizontal" aria-valuemin={0} aria-valuemax={layers - 1} aria-valuenow={value}
      aria-valuetext={`Layer ${value}`} aria-disabled={disabled || undefined}
      style={{left: `${(value + .5) / layers * 100}%`}}
      onKeyDown={(e: KeyboardEvent) => {
        if (disabled) return;
        const next = keyIndex(e.key, which === 'from' ? first : last, layers);
        if (next === undefined) return;
        e.preventDefault(); commit(apply(which, next));
      }}/>;
  };
  const count = b - a + 1;
  return <div className="docattentionlens-scrubber-layer is-range" role="group" aria-label="Rollout layer interval">
    <span className="docattentionlens-scrubber-label" aria-hidden="true">Layers</span>
    <div ref={track} className="docattentionlens-scrubber-track" data-testid="layer-scrubber-track"
      onPointerDown={e => {
        if (disabled || e.button !== 0) return;
        const v = pointedAt(e, track.current, layers);
        if (v === undefined) return;
        const which = nearest(v);
        capture(e); dragging.current = which; setDraft(apply(which, v));
      }}
      onPointerMove={e => {
        if (!dragging.current) return;
        const v = pointedAt(e, track.current, layers);
        if (v !== undefined) setDraft(apply(dragging.current, v));
      }}
      onPointerUp={e => {
        const which = dragging.current;
        if (!which) return;
        dragging.current = null;
        const v = pointedAt(e, track.current, layers);
        const pair = v !== undefined ? apply(which, v) : draft;
        setDraft(null); if (pair) commit(pair);
      }}
      onPointerCancel={() => {dragging.current = null; setDraft(null);}}>
      <Bars count={layers} bars={bars} title={barsTitle} cell={i => i === a || i === b ? 'is-current' : i > a && i < b ? 'is-in-range' : ''}/>
      {!fixed && handle('from')}{handle('to')}
    </div>
    <span className="docattentionlens-scrubber-readout">
      {!fixed && <><select aria-label="Start layer" value={first} disabled={disabled}
        onChange={e => {const v = Number(e.target.value); range.onChange(v, Math.max(v, last));}}>
        {Array.from({length: layers}, (_, i) => <option key={i} value={i}>{i}</option>)}</select><span className="docattentionlens-scrubber-of">–</span></>}
      <select aria-label="End layer" value={last} disabled={disabled}
        onChange={e => {const v = Number(e.target.value); range.onChange(fixed ? 0 : Math.min(first, v), v);}}>
        {Array.from({length: layers}, (_, i) => <option key={i} value={i}>{i}</option>)}</select>
      <span className="docattentionlens-scrubber-of">/ {layers - 1}</span>
    </span>
    {gap ? <span className="docattentionlens-scrubber-range-note is-warning" role="alert">{gap}{' '}
      <button type="button" className="docattentionlens-scrubber-link" disabled={disabled} onClick={() => range.onChange(first, last)}>Use layers {first}–{last}</button></span>
      : <span className="docattentionlens-scrubber-range-note" role="status"
        title={a === 0 ? 'Starts at layer 0 · traces back to model input positions' : `Traces back to inputs of layer ${a}`}>
        Layers {a}–{b} · {count} {count === 1 ? 'layer' : 'layers'} composed</span>}
    {fixed && range.onChooseRange && <button type="button" className="docattentionlens-scrubber-link" disabled={disabled} onClick={range.onChooseRange}>Choose a layer range</button>}
  </div>;
}

/** Compact head slider: previews while dragging, commits on release, keys, blur or a short pause. */
function HeadSlider({value, count, disabled, onChange}: {value: number; count: number; disabled: boolean; onChange: (v: number) => void}) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  useEffect(() => {
    if (draft === value || disabled) return;
    const timer = setTimeout(() => onChange(draft), 150);
    return () => clearTimeout(timer);
  }, [draft, value, onChange, disabled]);
  const commit = () => {if (draft !== value && !disabled) onChange(draft);};
  return <input type="range" className="docattentionlens-scrubber-range-input" aria-label="Attention head slider" min={0} max={count - 1} step={1}
    value={draft} disabled={disabled} onChange={e => setDraft(Number(e.target.value))} onPointerUp={commit} onKeyUp={commit} onBlur={commit}/>;
}

function HeadPicker({head, heads, allHeads, disabled, onHead}: {head: number; heads: number; allHeads?: AllHeads; disabled: boolean; onHead: (v: number) => void}) {
  const cellsRef = useRef<(HTMLButtonElement | null)[]>([]);
  const current = clamp(head, heads);
  const all = !!allHeads?.active;
  const choose = (v: number) => {
    if (disabled) return;
    if (all) allHeads?.onChange(false);
    if (v !== head || all) onHead(v);
  };
  const compact = heads > HEAD_CELL_LIMIT;
  const allLabel = allHeads?.label ?? 'All';
  return <div className={`docattentionlens-scrubber-head${compact ? ' is-compact' : ''}`} data-testid="layer-scrubber-head">
    <span className="docattentionlens-scrubber-label" aria-hidden="true">Head</span>
    <span className="docattentionlens-scrubber-readout">
      <select aria-label="Attention head" value={all ? 'all' : current} disabled={disabled}
        onChange={e => e.target.value === 'all' ? allHeads?.onChange(true) : choose(Number(e.target.value))}>
        {/* The select lists the heads; the All cell (or A) chooses All, and the select reads it while it is shown. */}
        {all && <option value="all">{allLabel}</option>}
        {Array.from({length: heads}, (_, i) => <option key={i} value={i}>{i}</option>)}
      </select><span className="docattentionlens-scrubber-of">/ {heads - 1}</span>
    </span>
    {allHeads && <button type="button" className="docattentionlens-scrubber-cell docattentionlens-scrubber-all" aria-pressed={all} disabled={disabled}
      title="Mean over all heads (A)" onClick={() => allHeads.onChange(!all)}>{allLabel}</button>}
    {compact ? <HeadSlider value={current} count={heads} disabled={disabled} onChange={choose}/>
      : <div className="docattentionlens-scrubber-cells" role="radiogroup" aria-label="Attention head slider" tabIndex={-1}
        aria-disabled={disabled || undefined}
        onKeyDown={e => {
          const next = keyIndex(e.key, current, heads);
          if (next === undefined || disabled) return;
          e.preventDefault(); choose(next);
          if (e.target !== e.currentTarget) cellsRef.current[next]?.focus();
        }}>
        {Array.from({length: heads}, (_, i) => <button key={i} ref={el => {cellsRef.current[i] = el;}} type="button" role="radio"
          className={`docattentionlens-scrubber-cell${i === current ? ' is-current' : ''}`} aria-checked={!all && i === current}
          aria-label={`Head ${i}`} tabIndex={i === current && !disabled ? 0 : -1} disabled={disabled} onClick={() => choose(i)}>{i}</button>)}
      </div>}
  </div>;
}

/** Layer and head selection under the page; keeps the accessible names of the controls it replaced. */
export function LayerScrubber({layer = 0, layers = 0, onLayer, head = 0, heads = 0, onHead, bars, barsTitle, allHeads, range,
  showLayer = true, showHead = true, disabled = false, className}: LayerScrubberProps) {
  const off = !!disabled;
  const reason = typeof disabled === 'string' ? disabled : undefined;
  const layerShown = showLayer && layers > 0 && (!!range || !!onLayer);
  const headShown = showHead && heads > 0 && !!onHead;
  if (!layerShown && !headShown && !reason) return null;
  const hint = [layerShown && !range ? '[ ] layer' : '', headShown ? '{ } head' : ''].filter(Boolean).join(' · ');
  return <div className={`docattentionlens-layer-scrubber${off ? ' is-disabled' : ''}${className ? ` ${className}` : ''}`}
    role="group" aria-label="Layer and head" data-testid="layer-scrubber">
    {layerShown && (range ? <RangeTrack range={range} layers={layers} bars={bars} barsTitle={barsTitle} disabled={off}/>
      : <LayerTrack layer={layer} layers={layers} bars={bars} barsTitle={barsTitle} disabled={off} onLayer={v => onLayer?.(v)}/>)}
    {headShown && <HeadPicker head={head} heads={heads} allHeads={allHeads} disabled={off} onHead={v => onHead?.(v)}/>}
    {reason && <span className="docattentionlens-scrubber-reason" role="status">{reason}</span>}
    {hint && <span className="docattentionlens-scrubber-hint" aria-hidden="true">{hint}</span>}
  </div>;
}
