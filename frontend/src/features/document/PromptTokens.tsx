import { Fragment, useEffect, useId, useRef, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from 'react';
import { ChevronDown } from '../../app/icons';
import { attentionWeightLabel, sharePercent } from './attentionDisplay';
import { SPECIAL_ROLES, continuesWord, displayPiece, tokenHeat, type Token } from './TokenStrip';
import { ResizeHandle } from './WorkspaceControls';
import '../../styles/prompt-tokens.css';

export type PromptPart = 'question' | 'answer' | 'template' | 'image';
/** `page` names the page an image patch belongs to ("Page 3"), for the page-image chip. */
export type PromptToken = Token & {part: PromptPart; value?: number | null; selectable?: boolean; page?: string};
/**
 * A part of the prompt the shown reading does not contain (the question in a Page reading, say). It is
 * drawn dimmed where it would sit, so the order stays legible, with the reading that contains it.
 */
export type PromptGhost = {part: PromptPart; text?: string; reason: string; action?: {label: string; onClick: () => void}};

// Word joining and heat colours are TokenStrip's, so a token reads and shades the same in both views.
const NEWLINE = /Ċ|\n/;
// A template run longer than this folds into one chip, so a long system prompt does not push the question out of view.
const TEMPLATE_FOLD = 10;
// Dragging the bottom edge above this height hides the prompt to its header (one keyboard step from hidden opens it);
// shown rows are at least two lines tall.
const HIDE_BELOW = 16, MIN_ROWS = 40;

type Item = {kind: 'token'; token: PromptToken; gap: boolean} | {kind: 'image'; run: PromptToken[]; gap: boolean};
type Run = {part: PromptPart; items: Item[]; tokens: PromptToken[]; sum: number};
type Segment = Run | {part: PromptPart; ghost: PromptGhost};
/** One labelled row: consecutive template and page-image runs share a Context row; question and answer each have their own. */
type Row = {kind: string; label: ReactNode; segments: Segment[]};

/** The whole prompt as the model read it, in sequence order, shaded by attention. */
export function PromptTokens({tokens, scaleMax = 1, opacity = .65, signed = false, selected, onSelect, last, label = 'Prompt', answerLabel = 'reference answer', ghosts = [], note, controls, open = true, onOpen, height, onHeight}: {
  /** The whole sequence in position order, image patches included. */
  tokens: PromptToken[]; scaleMax?: number; opacity?: number;
  /** Signed MaxSim similarity uses the page's lilac/blue similarity mapping instead of amber attention. */
  signed?: boolean; selected?: (token: PromptToken) => boolean; onSelect?: (token: PromptToken) => void;
  /** Position of the last answer token, outlined as the target. */
  last?: number; label?: string;
  /** Key name for 'answer' tokens: the case's reference answer, or the model's generated answer. */
  answerLabel?: string;
  /** Parts missing from this reading, shown dimmed in their place. */
  ghosts?: PromptGhost[];
  note?: string;
  /** The source line's own controls (view switch and options), placed at the end of the header. */
  controls?: ReactNode;
  /** Whether the rows are shown; the header stays either way. */
  open?: boolean; onOpen?: (open: boolean) => void;
  /** Height of the rows in pixels; undefined keeps the default of a few lines. */
  height?: number; onHeight?: (height: number) => void;
}) {
  const selectable = (t: PromptToken) => !!onSelect && t.part !== 'image' && t.selectable !== false;
  const enabled = tokens.filter(t => selectable(t) && t.eligible !== false);
  const chosen = selected ? enabled.find(selected) : undefined;
  const current = chosen?.index;
  // A page patch can be the source too (clicked on the page); the readout names it, though its chip is the whole image.
  const patch = !chosen && selected ? tokens.find(t => t.part === 'image' && selected(t)) : undefined;
  const focusIndex = current ?? enabled[0]?.index;
  const rowsId = useId();
  // The rows scroll inside their height; keep the selected token in view without scrolling the page.
  const text = useRef<HTMLDivElement>(null);
  const [unfolded, setUnfolded] = useState<Set<number>>(new Set());
  // The rows' shown and full heights, for the resize handle; the last values are kept while the prompt is hidden.
  const [size, setSize] = useState({shown: 0, full: 240});
  useEffect(() => {
    const box = text.current;
    if (!box) return;
    const reveal = () => {
      setSize(previous => previous.shown === box.clientHeight && previous.full === box.scrollHeight ? previous : {shown: box.clientHeight, full: box.scrollHeight});
      const chip = current === undefined ? null : box.querySelector<HTMLElement>(`[data-token-index="${current}"]`);
      if (!chip) return;
      const outer = box.getBoundingClientRect(), own = chip.getBoundingClientRect();
      if (own.top < outer.top) box.scrollTop -= outer.top - own.top + 4;
      else if (own.bottom > outer.bottom) box.scrollTop += own.bottom - outer.bottom + 4;
    };
    reveal();
    const observer = new ResizeObserver(reveal);
    observer.observe(box);
    return () => observer.disconnect();
  }, [current, open, tokens.length]);
  if (!tokens.length) return null;
  // Arrow keys and the step buttons move through selectable tokens, as in TokenStrip.
  function move(by: number, focus: boolean) {
    if (!onSelect || !enabled.length) return;
    const at = current === undefined ? (by > 0 ? -1 : enabled.length) : enabled.findIndex(t => t.index === current);
    const next = enabled[Math.max(0, Math.min(enabled.length - 1, at + by))];
    if (!next) return;
    onSelect(next);
    if (focus) text.current?.querySelector<HTMLElement>(`[data-token-index="${next.index}"]`)?.focus();
  }
  function step(event: KeyboardEvent) {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    move(event.key === 'ArrowRight' ? 1 : -1, true);
  }
  // Group consecutive image tokens into one run; a token joins its neighbour without a gap when it continues a word.
  const items: Item[] = [];
  let broke = true;
  tokens.forEach((token, i) => {
    const prev = tokens[i - 1];
    if (token.part === 'image') {
      const run = items.at(-1);
      if (run?.kind === 'image') run.run.push(token);
      else items.push({kind: 'image', run: [token], gap: !broke});
      broke = false;
      return;
    }
    const continuation = !!prev && prev.part === token.part && continuesWord(token, prev);
    items.push({kind: 'token', token, gap: !broke && !continuation});
    broke = NEWLINE.test(token.piece);
  });
  // Consecutive items of one part form a run, with the weight its tokens receive.
  const weight = (t: PromptToken) => typeof t.value === 'number' ? (signed ? Math.abs(t.value) : t.value) : 0;
  const segments: Segment[] = [];
  for (const item of items) {
    const part = item.kind === 'image' ? 'image' : item.token.part;
    const members = item.kind === 'image' ? item.run : [item.token];
    const open = segments.at(-1);
    const run = open && 'items' in open && open.part === part ? open : undefined;
    if (run) { run.items.push(item); run.tokens.push(...members); }
    else segments.push({part, items: [item], tokens: [...members], sum: 0});
  }
  const runs = segments.filter((s): s is Run => 'items' in s);
  runs.forEach(run => { run.sum = run.tokens.reduce((sum, t) => sum + weight(t), 0); });
  // A missing part sits where the joint reading puts it: after the page image, else before the question.
  if (ghosts.length) {
    const afterImage = segments.map(s => s.part).lastIndexOf('image') + 1;
    const beforeQuestion = segments.findIndex(s => s.part === 'question');
    const at = afterImage > 0 ? afterImage : beforeQuestion >= 0 ? beforeQuestion : segments.length;
    segments.splice(at, 0, ...ghosts.map(ghost => ({part: ghost.part, ghost})));
  }
  const heatStyle = (value: number | null | undefined) => tokenHeat(value, scaleMax, opacity, signed);
  // Each run's share of the weight on the shown tokens. Shares are stated only for a chosen source's row:
  // incoming columns and similarities do not divide one source's attention.
  const total = runs.reduce((sum, run) => sum + run.sum, 0);
  const largest = Math.max(0, ...runs.map(run => run.sum));
  const shares = !signed && (!!chosen || !!patch) && total > 0;
  const shareLabel = (run: Run) => sharePercent(run.sum / total);
  // A run's weight as attention heat against the largest run (amber even for signed similarity: sums are of magnitudes).
  const runHeat = (run: Run): CSSProperties | undefined => tokenHeat(run.sum, largest, opacity, false);
  const holdsSource = (run: Run) => run.tokens.some(t => t.index === current || t.index === patch?.index);
  const partName = (part: PromptPart) => part === 'question' ? 'Question' : part === 'answer' ? answerLabel[0].toUpperCase() + answerLabel.slice(1) : part === 'image' ? 'Page image' : 'Template';
  const parts = new Set(tokens.map(t => t.part));
  const hasLast = last !== undefined && tokens.some(t => t.index === last && t.part !== 'image');
  const key: ReactNode[] = [
    parts.has('question') && <span key="q" className="docattentionlens-prompt-token is-question">question</span>,
    parts.has('answer') && <span key="a" className="docattentionlens-prompt-token is-answer">{answerLabel}</span>,
    hasLast && <span key="t" className="docattentionlens-prompt-token is-answer is-last">last answer token</span>,
    (chosen || patch) && <span key="s" className="docattentionlens-prompt-token is-source-key">source</span>,
  ].filter(Boolean);
  const renderItem = (item: Item) => {
    if (item.kind === 'image') {
      const {run} = item, values = run.map(t => t.value).filter((v): v is number => typeof v === 'number');
      const sum = values.reduce((a, b) => a + b, 0);
      const title = [`Positions ${run[0].index}–${run.at(-1)!.index} · ${run[0].page ?? 'page image'} · ${run.length} patches`,
        values.length ? `${signed ? 'image similarity' : 'image weight'} Σ ${attentionWeightLabel(sum)}` : '',
        onSelect ? 'click a patch on the page to attend from it' : ''].filter(Boolean).join(' · ');
      return <Fragment key={`image-${run[0].index}`}>{item.gap && ' '}<span className="docattentionlens-prompt-image" title={title} data-testid="prompt-image">
        <span aria-hidden="true">▣ </span>{run[0].page ?? 'Page image'} · {run.length} patches{values.length > 0 && <small> · Σ {attentionWeightLabel(sum)}</small>}</span></Fragment>;
    }
    const {token} = item, value = token.value;
    const piece = displayPiece(token.piece);
    // A lone space reads as a gap in running text; the strips' "·" placeholder would look like punctuation.
    const blank = piece === '·' && token.piece !== '·';
    const title = [token.title ?? `Position ${token.index} · ${token.piece} · ${token.role}`, typeof value === 'number' ? `${signed ? 'similarity' : 'attention weight'} ${attentionWeightLabel(value)}` : ''].filter(Boolean).join(' · ');
    const className = `docattentionlens-prompt-token is-${token.part} ${item.gap ? '' : 'is-joined'} ${token.index === last ? 'is-last' : ''} ${SPECIAL_ROLES.has(token.role) ? 'is-special' : ''}`;
    const chip = selectable(token)
      ? <button type="button" className={className} style={heatStyle(value)} title={title} data-token-index={token.index} data-testid="token-chip"
          aria-label={`Token ${token.index}: ${blank ? 'space' : piece}`} aria-pressed={selected?.(token) ?? false} disabled={token.eligible === false}
          tabIndex={focusIndex === token.index ? 0 : -1}
          onClick={() => onSelect!(token)} onKeyDown={step}>{blank ? ' ' : piece}</button>
      : <span className={className} style={heatStyle(value)} title={title} data-testid="token-heat">{blank ? ' ' : piece}</span>;
    return <Fragment key={token.index}>{item.gap && ' '}{chip}</Fragment>;
  };
  // A long template run folds into one chip with its weight; it opens on request, or while it holds the source.
  const fold = (run: Run) => {
    const first = run.tokens[0].index;
    if (run.part !== 'template' || run.items.length <= TEMPLATE_FOLD || holdsSource(run)) return run.items.map(renderItem);
    const open = unfolded.has(first), count = run.tokens.length;
    const flip = () => setUnfolded(previous => { const next = new Set(previous); if (open) next.delete(first); else next.add(first); return next; });
    if (open) return [...run.items.map(renderItem), <button key="fold" type="button" className="docattentionlens-prompt-fold" aria-expanded="true"
      aria-label="Fold the template tokens" title="Fold the template tokens" onClick={flip}>less</button>];
    return <button type="button" className="docattentionlens-prompt-fold is-run" aria-expanded="false" style={runHeat(run)}
      aria-label={`Show ${count} template tokens`} title={`Show the ${count} template tokens${total > 0 ? ` · weight Σ ${attentionWeightLabel(run.sum)}` : ''}`} onClick={flip}>
      <span aria-hidden="true">▸ </span>Template · {count} tokens{total > 0 && <small> · Σ {attentionWeightLabel(run.sum)}</small>}</button>;
  };
  // Rows: consecutive template and image runs form one Context row; question, answer and missing parts have their own.
  const rows: Row[] = [];
  for (const segment of segments) {
    const kind = 'ghost' in segment ? `ghost-${segment.part}` : segment.part === 'template' || segment.part === 'image' ? 'context' : segment.part;
    const row = rows.at(-1);
    if (row && row.kind === kind && kind === 'context') row.segments.push(segment);
    else rows.push({kind, label: '', segments: [segment]});
  }
  rows.forEach(row => {
    row.label = row.kind === 'context' ? (row.segments.some(s => s.part === 'image') ? 'Context' : 'Template')
      : 'ghost' in row.segments[0] ? <>{partName(row.segments[0].part)} <em>not in this reading</em></> : partName(row.segments[0].part);
  });
  const renderSegment = (segment: Segment, n: number) => 'ghost' in segment
    ? <span key={`ghost-${segment.part}`} className="docattentionlens-prompt-segment is-ghost" data-part={segment.part} data-testid="prompt-ghost" title={segment.ghost.reason}>
        <span className="docattentionlens-prompt-token is-ghost">{segment.ghost.text || segment.ghost.reason}</span>
        {segment.ghost.action && <button type="button" className="docattentionlens-link-button docattentionlens-prompt-ghost-action" onClick={segment.ghost.action.onClick}>{segment.ghost.action.label}</button>}
      </span>
    : <span key={`${segment.part}-${n}`} className="docattentionlens-prompt-segment" data-part={segment.part}>{fold(segment)}</span>;
  // The bar: one cell per run in sequence order, widths eased so short question and answer runs stay visible.
  const tickName = (run: Run) => run.part === 'image' ? (run.tokens[0].page ?? 'page image').toLowerCase()
    : run.part === 'template' ? (run.tokens.length > TEMPLATE_FOLD ? 'template' : '') : run.part === 'answer' ? answerLabel.replace(/^generated |^reference /, '') : 'question';
  const cells = runs.map((run, n) => ({run, n, grow: Math.sqrt(run.tokens.length)}));
  const summary = cells.filter(c => tickName(c.run)).map(c => `${tickName(c.run)}${shares ? ` ${shareLabel(c.run)}` : ''}`).join(', ');
  const bar = (compact: boolean) => <div className={`docattentionlens-prompt-bar${compact ? ' is-compact' : ''}`} aria-hidden="true">
    {cells.map(({run, n, grow}) => <span key={n} className={`is-${run.part}${holdsSource(run) ? ' has-source' : ''}`} style={{flexGrow: grow, ...runHeat(run)}}
      title={`${partName(run.part)} · ${run.tokens.length} ${run.part === 'image' ? 'patches' : 'tokens'}${shares ? ` · ${shareLabel(run)} of the weight` : ''}`}/>)}
  </div>;
  // The header names the source the page shows, with steps through the selectable tokens.
  const steps = onSelect && enabled.length > 1 && <span className="docattentionlens-prompt-step" role="group" aria-label="Step through tokens">
    <button type="button" className="docattentionlens-button" aria-label="Previous token" title="Previous token (←)" disabled={chosen !== undefined && enabled[0].index === current} onClick={() => move(-1, false)}>‹</button>
    <button type="button" className="docattentionlens-button" aria-label="Next token" title="Next token (→)" disabled={chosen !== undefined && enabled.at(-1)!.index === current} onClick={() => move(1, false)}>›</button>
  </span>;
  const from = signed ? 'Similarity from' : 'Attention from';
  const readout = chosen ? <span className="docattentionlens-prompt-from">{from}<span className="docattentionlens-prompt-source" data-testid="prompt-readout">
      <b>“{displayPiece(chosen.piece)}”</b><small>{chosen.part === 'answer' ? answerLabel : chosen.part} · position {chosen.index}</small>{steps}</span></span>
    : patch ? <span className="docattentionlens-prompt-from">{from}<span className="docattentionlens-prompt-source" data-testid="prompt-readout"><b>page patch</b><small>position {patch.index}</small>{steps}</span></span>
    : onSelect ? <span className="docattentionlens-prompt-readout">Click a token, or a patch on the page, to choose the source{steps}</span> : null;
  const toggle = () => onOpen?.(!open);
  return <div className="docattentionlens-prompt-tokens" role="group" aria-label={label} data-open={open}>
    <div className="docattentionlens-prompt-head">
      {onOpen && <button type="button" className="docattentionlens-icon-button docattentionlens-prompt-toggle" aria-expanded={open} aria-controls={rowsId}
        aria-label={open ? `Hide ${label.toLowerCase()}` : `Show ${label.toLowerCase()}`} title={`${open ? 'Hide' : 'Show'} ${label.toLowerCase()} (P)`} onClick={toggle}><ChevronDown size={14}/></button>}
      <span className="docattentionlens-prompt-title"><strong>{label}</strong><span>what the model read · {tokens.length.toLocaleString()} tokens</span></span>
      {!open && <span className="docattentionlens-prompt-hidden-note">{bar(true)}<span>hidden · P or drag the edge to show</span></span>}
      <span className="docattentionlens-prompt-head-gap"/>
      {readout}
      {controls}
    </div>
    {open && <>
      <div className="docattentionlens-prompt-map" role="img" aria-label={`Prompt parts in order: ${summary}`}>
        {bar(false)}
        <div className="docattentionlens-prompt-ticks" aria-hidden="true">
          {cells.map(({run, n, grow}) => <span key={n} style={{flexGrow: grow}}>{tickName(run)}{shares && tickName(run) && <em> {shareLabel(run)}</em>}</span>)}
        </div>
      </div>
      <div ref={text} id={rowsId} className="docattentionlens-prompt-tokens-text" style={height === undefined ? undefined : {maxHeight: height}}>
        {rows.map((row, r) => <Fragment key={`${row.kind}-${r}`}>
          <span className={`docattentionlens-prompt-row-label is-row-${row.kind}`}>{row.label}</span>
          <div className={`docattentionlens-prompt-row${row.kind.startsWith('ghost') ? ' is-ghost' : ''}`} data-row={row.kind}>{row.segments.map(renderSegment)}</div>
        </Fragment>)}
      </div>
      <div className="docattentionlens-prompt-tokens-foot">
        <span className="docattentionlens-prompt-scale">Shading: {signed ? 'similarity' : chosen || patch ? 'attention from the source' : 'attention weight'} <i aria-hidden="true"/> high</span>
        {key.length > 0 && <p className="docattentionlens-prompt-tokens-key">{key.map((swatch, i) => <Fragment key={i}>{i > 0 && ' · '}{swatch}</Fragment>)}</p>}
        {onSelect && <span className="docattentionlens-prompt-hint">Click a token to make it the source · ← → step{onOpen ? ' · P hides' : ''}{onHeight ? ' · drag the edge to resize' : ''}</span>}
        {note && <details className="docattentionlens-token-note"><summary aria-label={`Read ${label.toLowerCase()}`} title={`Read ${label.toLowerCase()}`}><span aria-hidden="true">ⓘ</span></summary><p>{note}</p></details>}
      </div>
    </>}
    {onHeight && <div className="docattentionlens-prompt-grip" title="Drag to resize · double-click to hide or show" onDoubleClick={toggle}>
      <ResizeHandle horizontal label={`Resize ${label.toLowerCase()}`} value={open ? Math.round(height ?? size.shown) : 0} min={0} max={Math.max(size.full, HIDE_BELOW * 2)}
        onChange={value => {
          if (value < HIDE_BELOW) { if (open) onOpen?.(false); return; }
          if (!open) onOpen?.(true);
          onHeight(Math.max(value, MIN_ROWS));
        }}/>
    </div>}
  </div>;
}
