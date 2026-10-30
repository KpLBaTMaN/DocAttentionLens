import { useEffect, useRef, type CSSProperties, type KeyboardEvent, type ReactNode } from 'react';
import { ATTENTION_FILL, attentionAlpha, attentionWeightLabel } from './attentionDisplay';

export type Token = {index: number; piece: string; role: string; eligible?: boolean; title?: string};

// Word joining and heat colours are shared with PromptTokens, so a token reads and shades the same in both views.
export const SPECIAL_ROLES = new Set(['special', 'prefix_or_special', 'augmentation', 'bos', 'newline', 'prompt', 'visual_prompt']);
const WORD_START = /^[▁Ġ\s]/;
const POSITIVE = '#b080d1', NEGATIVE = '#247ed5';
const rgb = (hex: string) => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16)).join(',');

/** A word piece that continues the previous token's word; special tokens never join. */
export const continuesWord = (token: Token, previous: Token | undefined) =>
  !!previous && !WORD_START.test(token.piece) && !SPECIAL_ROLES.has(token.role) && !SPECIAL_ROLES.has(previous.role);

/**
 * A token's heat on the page heatmap's colour and intensity mapping: amber attention, or for signed MaxSim
 * similarity the page's lilac/blue similarity mapping. No fill without a value or at zero heat.
 */
export function tokenHeat(value: number | null | undefined, scaleMax: number, opacity: number, signed: boolean): CSSProperties | undefined {
  if (typeof value !== 'number') return undefined;
  const heat = attentionAlpha(signed ? Math.abs(value) : value, scaleMax, opacity);
  const colour = signed && value < 0 ? NEGATIVE : signed ? POSITIVE : ATTENTION_FILL;
  return heat > 0 ? {background: `rgba(${rgb(colour)},${heat})`} : undefined;
}

// Byte-level BPE (Qwen) draws every byte as one printable character: printable Latin-1
// bytes as themselves, the rest from U+0100 upward in byte order.
const BYTE_OF = (() => {
  const bytes = new Map<string, number>();
  let shifted = 0;
  for (let b = 0; b < 256; b++) {
    const printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
    bytes.set(String.fromCharCode(printable ? b : 256 + shifted++), b);
  }
  return bytes;
})();
const UTF8 = new TextDecoder('utf-8', {fatal: true});

/** A byte-level piece's text ("âĢĻs" → "’s"). Other pieces, and bytes split across tokens, are kept as they are. */
export function decodePiece(piece: string) {
  const chars = [...piece];
  if (!chars.some(c => c.charCodeAt(0) > 127) || !chars.every(c => BYTE_OF.has(c))) return piece;
  try { return UTF8.decode(Uint8Array.from(chars, c => BYTE_OF.get(c)!)); } catch { return piece; }
}

/** Readable token text: byte-level pieces decoded, word markers removed and newlines a visible symbol. */
export function displayPiece(piece: string) {
  // Newlines become visible first, so a newline token is not taken for a word-start space.
  return decodePiece(piece).replace(/Ċ|\n/g, '↵').replace(WORD_START, '') || '·';
}

/** One row of sequence tokens; clicking a token selects it as the attention source. */
export function TokenStrip({label, caption, tokens, values, scaleMax = 1, opacity = .65, signed = false, marked, selected, onSelect, note, children}: {
  /** The group's accessible name; `caption` is a shorter visible label (defaults to the name). */
  label: string; caption?: string; tokens: Token[]; values?: (number | null | undefined)[]; scaleMax?: number; opacity?: number;
  /** Signed MaxSim similarity uses the page's lilac/blue similarity mapping instead of amber attention. */
  signed?: boolean; marked?: Set<number>;
  selected?: number; onSelect?: (index: number) => void; note?: string; children?: ReactNode;
}) {
  const enabled = tokens.filter(t => t.eligible !== false);
  const focusIndex = enabled.some(t => t.index === selected) ? selected : enabled[0]?.index;
  // Narrow strips scroll sideways; keep the selected token in view without scrolling the page.
  const chips = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const strip = chips.current;
    if (!strip || selected === undefined) return;
    const reveal = () => {
      const chip = strip.querySelector<HTMLElement>(`[data-token-index="${selected}"]`);
      if (!chip) return;
      const box = strip.getBoundingClientRect(), own = chip.getBoundingClientRect();
      if (own.left < box.left) strip.scrollLeft -= box.left - own.left + 8;
      else if (own.right > box.right) strip.scrollLeft += own.right - box.right + 8;
    };
    reveal();
    const observer = new ResizeObserver(reveal);
    observer.observe(strip);
    return () => observer.disconnect();
  }, [selected]);
  // Arrow keys step through selectable tokens, as the source-position select does.
  function step(event: KeyboardEvent, current: number) {
    if (!onSelect || (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight')) return;
    event.preventDefault();
    const at = enabled.findIndex(t => t.index === current);
    const next = enabled[Math.max(0, Math.min(enabled.length - 1, at + (event.key === 'ArrowRight' ? 1 : -1)))];
    if (!next) return;
    onSelect(next.index);
    (event.currentTarget.parentElement?.querySelector(`[data-token-index="${next.index}"]`) as HTMLElement | null)?.focus();
  }
  return <div className="docattentionlens-token-row" role="group" aria-label={label}>
    <span className="docattentionlens-token-row-label" aria-hidden={caption !== undefined || undefined}>{caption ?? label}</span>
    <div ref={chips} className="docattentionlens-token-chips">{tokens.map((token, i) => {
      const value = values?.[token.index];
      const continuation = continuesWord(token, tokens[i - 1]);
      const text = displayPiece(token.piece);
      const title = [token.title ?? `Position ${token.index} · ${token.piece} · ${token.role}`, typeof value === 'number' ? `${signed ? 'similarity' : 'attention weight'} ${attentionWeightLabel(value)}` : '', marked?.has(token.index) ? 'MaxSim winner is the selected patch' : ''].filter(Boolean).join(' · ');
      const style = tokenHeat(value, scaleMax, opacity, signed);
      const className = `docattentionlens-token ${continuation ? 'is-continuation' : ''} ${SPECIAL_ROLES.has(token.role) ? 'is-special' : ''} ${marked?.has(token.index) ? 'is-marked' : ''}`;
      return onSelect
        ? <button key={token.index} type="button" className={className} style={style} title={title} data-token-index={token.index} data-testid="token-chip"
            aria-label={`Token ${token.index}: ${text}`} aria-pressed={selected === token.index} disabled={token.eligible === false}
            tabIndex={focusIndex === token.index ? 0 : -1}
            onClick={() => onSelect(token.index)} onKeyDown={e => step(e, token.index)}>{text}</button>
        : <span key={token.index} className={className} style={style} title={title} data-testid="token-heat">{text}</span>;
    })}</div>
    {children}
    {note && <details className="docattentionlens-token-note"><summary aria-label={`Read ${label.toLowerCase()}`} title={`Read ${label.toLowerCase()}`}><span aria-hidden="true">ⓘ</span></summary><p>{note}</p></details>}
  </div>;
}
