import type { CSSProperties, ReactNode } from 'react';
import { ATTENTION_FILL, attentionWeightLabel, mapGradient, roundedTimesLabel, scalePosition, sharePercent, type Colormap } from './attentionDisplay';
import type { MassGroup } from './massBreakdown';
import { displayPiece } from './TokenStrip';

/**
 * How a scale's colours are drawn: amber opacity (tokens and page overlay), an opaque colour map
 * (heatmap), a colour map blended over the page at the opacity (blend), or signed similarity.
 */
export type Ramp = {kind: 'amber' | 'heatmap' | 'blend' | 'signed'; opacity: number; onPage?: boolean; colormap?: Colormap};

export type ScaleLine = {
  key: string;
  /** What carries the colour: a token strip or the page. */
  where: string;
  /** The page's quantity, when it is not implied by `where`. */
  what?: ReactNode;
  ramp: Ramp;
  /** Range and what sets it, kept as one claim: "0 → 0.829077 · token maximum". */
  range: string; rangeTestId?: string;
  /** The item holding the maximum, shown only when the scale is that maximum. */
  peak?: ReactNode;
};

const channels = (hex: string) => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16)).join(',');
const AMBER = channels(ATTENTION_FILL), POSITIVE = channels('#b080d1'), NEGATIVE = channels('#247ed5');

// Overlays are drawn over the page image, so a page ramp starts from white paper; token heat sits on the panel.
export function rampStyle({kind, opacity, onPage, colormap}: Ramp): CSSProperties {
  const paper = onPage ? ', #fff' : '';
  if (kind === 'heatmap') return {background: mapGradient(colormap)};
  // A blend shows the map at the opacity over white paper.
  if (kind === 'blend') return {background: `linear-gradient(rgba(255,255,255,${1 - opacity}), rgba(255,255,255,${1 - opacity})), ${mapGradient(colormap)}`};
  if (kind === 'signed') return {background: `linear-gradient(to right, rgba(${NEGATIVE},${opacity}), rgba(${NEGATIVE},0) 50%, rgba(${POSITIVE},0) 50%, rgba(${POSITIVE},${opacity}))${paper}`};
  return {background: `linear-gradient(to right, rgba(${AMBER},0), rgba(${AMBER},${opacity}))${paper}`};
}

/** The token with the largest magnitude across one or more strips; values are indexed by token index. */
export function strongestToken<T extends {index: number}>(strips: [T[], (number | null | undefined)[] | undefined][]) {
  let best: {token: T; value: number} | undefined;
  for (const [tokens, values] of strips) for (const token of tokens) {
    const value = values?.[token.index];
    if (typeof value === 'number' && value !== 0 && (!best || Math.abs(value) > Math.abs(best.value))) best = {token, value};
  }
  return best;
}

/**
 * Whether one colour means one value on the tokens and on the page. Image contrast gives each its own
 * maximum, so the same amber can stand for very different weights.
 */
export function scaleAgreement(tokenMax: number, pageMax: number, shared: boolean, rounded = false) {
  if (shared || tokenMax === pageMax) return 'Tokens and page share one scale: the same colour is the same value.';
  const ratio = tokenMax / pageMax, larger = ratio >= 1 ? ratio : 1 / ratio;
  const times = rounded ? roundedTimesLabel(larger) : larger >= 10 ? Math.round(larger).toLocaleString() : larger.toFixed(1);
  if (larger < 1.5) return `Separate scales of similar size (${times}×).`;
  return ratio > 1
    ? `Separate scales: the token maximum is ${times}× the page maximum, so the same colour is a larger value on a token.`
    : `Separate scales: the page maximum is ${times}× the token maximum, so the same colour is a larger value on the page.`;
}

/** One key for every colour scale in view: each token strip and the page. */
export function ScaleKey({placement, mode, lines, note, massShare, mass, breakdown, open = true, onToggle, exact, onExact}: {
  placement: 'panel' | 'legend';
  /** The Scale choice under the page. */
  mode: string;
  lines: ScaleLine[]; note?: string;
  /** Image share of the shown row, drawn as a bar beside the stated shares. */
  massShare?: number; mass?: ReactNode;
  /** The shown row by target role, under the shares. */
  breakdown?: Parameters<typeof MassTable>[0];
  open?: boolean; onToggle?: (open: boolean) => void;
  /** Whether the lines state exact values or rounded ones; the caller formats them. */
  exact?: boolean; onExact?: (exact: boolean) => void;
}) {
  if (!lines.length && !mass) return null;
  const detailed = open || placement === 'legend';
  return <section className="docattentionlens-scale-key" data-placement={placement} data-open={detailed} aria-label="Colour scale">
    <div className="docattentionlens-scale-head"><span className="docattentionlens-row-label">Scale</span><span className="docattentionlens-scale-mode">{mode}</span>
      {onToggle && placement === 'panel' && <button type="button" className="docattentionlens-link-button docattentionlens-scale-toggle" aria-expanded={open} onClick={() => onToggle(!open)}>{open ? 'Fewer details' : 'More details'}</button>}
      {onExact && <NumberStyle exact={!!exact} onChange={onExact}/>}</div>
    {lines.map(line => <div key={line.key} className="docattentionlens-scale-line">
      <span className="docattentionlens-scale-ramp" style={rampStyle(line.ramp)} aria-hidden="true"/>
      <div className="docattentionlens-scale-text">
        <span className="docattentionlens-scale-name"><strong>{line.where}</strong>{line.what && <span className="docattentionlens-scale-what"> · {line.what}</span>}</span>
        {line.peak && <span className="docattentionlens-scale-peak">Peak {line.peak}</span>}
        <span className="docattentionlens-scale-range" data-testid={line.rangeTestId}>{line.range}</span>
      </div>
    </div>)}
    {note && <p className="docattentionlens-scale-note" data-testid="scale-agreement">{note}</p>}
    {mass && <div className="docattentionlens-scale-mass"><span className="docattentionlens-row-label">Mass</span>
      {massShare !== undefined && <span className="docattentionlens-scale-split" aria-hidden="true"><i style={{width: `${Math.max(0, Math.min(1, massShare)) * 100}%`}}/></span>}
      {mass}</div>}
    {breakdown && detailed && <MassTable {...breakdown}/>}
  </section>;
}

/**
 * Every target group of the shown row with its share and value, so the non-image mass is not one lump:
 * the question, the answer so far, the prompt template and special tokens (often an attention sink).
 */
export function MassTable({groups, total, whole, format}: {groups: MassGroup[]; total: number; whole: string; format: (value: number) => string}) {
  if (!groups.length || !(total > 0)) return null;
  return <table className="docattentionlens-mass-breakdown" data-testid="mass-breakdown">
    <caption>Where the {whole} goes</caption>
    <tbody>{groups.map(g => {
      const share = Math.max(0, g.mass / total);
      return <tr key={g.key} data-group={g.key}>
        <th scope="row" title={`${g.positions} target position${g.positions === 1 ? '' : 's'}`}>{g.label}
          {g.top && g.positions > 1 && g.top.mass >= .5 * g.mass && <span className="docattentionlens-mass-top" title={`Position ${g.top.index} · ${attentionWeightLabel(g.top.mass)}`}> · mostly “{displayPiece(g.top.piece)}”</span>}</th>
        <td className="docattentionlens-mass-bar" aria-hidden="true"><i data-page={g.key === 'page' || g.key === 'image' || undefined} style={{width: `${Math.min(1, share) * 100}%`}}/></td>
        <td className="docattentionlens-mass-share">{sharePercent(share)}</td>
        <td className="docattentionlens-mass-value" title={attentionWeightLabel(g.mass)}>{format(g.mass)}</td>
      </tr>;
    })}</tbody>
  </table>;
}

/** Rounded values for a quick read, or the exact ones (scientific notation for tiny weights). */
export function NumberStyle({exact, onChange}: {exact: boolean; onChange: (exact: boolean) => void}) {
  return <div className="docattentionlens-segmented docattentionlens-number-style" role="radiogroup" aria-label="Values" onKeyDown={event => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const next = event.currentTarget.querySelector<HTMLButtonElement>(`button[aria-checked=false]`);
    next?.click(); next?.focus();
  }}>
    <button type="button" role="radio" aria-checked={!exact} tabIndex={exact ? -1 : 0} title="Values rounded to three decimals" onClick={() => onChange(false)}>Rounded</button>
    <button type="button" role="radio" aria-checked={exact} tabIndex={exact ? 0 : -1} title="Values as calculated, in scientific notation when tiny" onClick={() => onChange(true)}>Exact</button>
  </div>;
}

/** What the pointer is over on the page, with its shown value. */
export type PageHover = {label: string; value: number};

/** The page's colour scale in the canvas legend, with the hovered value. Display only. */
export function PageKey({ramp, floor, maximum, unit, hover}: {
  ramp: Ramp; floor: number; maximum: number; unit: string; hover?: PageHover;
}) {
  const at = hover && Math.max(0, Math.min(1, scalePosition(hover.value, maximum, floor)));
  const clipped = !!hover && hover.value > maximum;
  return <div className="docattentionlens-page-key" data-testid="page-key" role="group" aria-label={`Page colour key: ${unit}`}
    title={`${unit} · ${attentionWeightLabel(floor)} → ${attentionWeightLabel(maximum)} · hover the page to read a value`}>
    {/* One line: floor, ramp (with the hover marker), bound. */}
    <span className="docattentionlens-page-key-scale">
      <span className="docattentionlens-page-key-end" data-testid="page-key-floor">{floor === 0 ? '0' : attentionWeightLabel(floor)}</span>
      <span className="docattentionlens-page-key-ramp" style={rampStyle({...ramp, onPage: true})} aria-hidden="true">
        {at !== undefined && <i data-testid="page-key-marker" style={{left: `${at * 100}%`}}/>}</span>
      <span className="docattentionlens-page-key-end" data-testid="page-key-max">{attentionWeightLabel(maximum)}</span>
    </span>
    {/* The value under the pointer appears inline only while hovering; otherwise the invitation is for screen readers. */}
    <output className="docattentionlens-page-key-readout" data-testid="page-key-readout" aria-live="polite" data-empty={!hover}
      title={hover && `${hover.label}: ${attentionWeightLabel(hover.value)}, ${clipped ? 'above the scale' : `${Math.round(hover.value / maximum * 100)}% of the maximum`}`}>
      {hover ? `${clipped ? '>100' : Math.round(hover.value / maximum * 100)}% · ${attentionWeightLabel(hover.value)}` : 'Hover the page'}</output>
  </div>;
}
