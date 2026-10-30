import { useRef, useState, type KeyboardEvent } from 'react';
import { attentionWeightLabel } from './attentionDisplay';

/** One page in the filmstrip. `state` is short ("Supplied", "Omitted", "#2"); `detail` is its full description. */
export type FilmstripPage = {
  pageId: string; pageNumber: number;
  /** Thumbnail image; a numbered tile is shown when it is missing or fails to load. */
  image?: string;
  state: string;
  /** Retrieval score as shown (signed MaxSim), kept exact in `detail`. */
  score?: number;
  /** Attention mass of the shown row on this page's patches (context lens); undefined when not measured. */
  mass?: number;
  /** Omitted pages have no captured image attention: not measured, never zero. */
  omitted?: boolean;
  detail: string;
};

function Thumb({page, load}: {page: FilmstripPage; load: boolean}) {
  const [failed, setFailed] = useState(false);
  return <span className="docattentionlens-film-thumb" aria-hidden="true">
    {page.image && load && !failed ? <img src={page.image} alt="" loading="lazy" decoding="async" onError={() => setFailed(true)}/> : <span className="docattentionlens-film-tile">{page.pageNumber}</span>}
  </span>;
}

/** The page filmstrip: the capture's pages as thumbnails, shown when there is more than one page. */
export function Filmstrip({pages, current, onSelect, label, massLabel, orientation = 'vertical', loadImages = true}: {
  pages: FilmstripPage[]; current: string; onSelect: (pageId: string) => void;
  /** Accessible name, e.g. "Generator context pages". */
  label: string;
  /** What the mass is, for its tooltip ("Original-domain attention mass"). */
  massLabel?: string;
  /** Vertical in the desktop column; a wrapping grid in the phone sheet. */
  orientation?: 'vertical' | 'grid';
  /**
   * Thumbnails are full page images: they wait until the workspace's own evidence has loaded, so they never
   * hold the browser's few connections while the page and its attention are fetched. Numbered tiles show meanwhile.
   */
  loadImages?: boolean;
}) {
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  const at = Math.max(0, pages.findIndex(p => p.pageId === current));
  const [focusIndex, setFocusIndex] = useState<number | null>(null);
  const tabStop = focusIndex ?? at;
  const largest = Math.max(...pages.map(p => p.mass ?? 0), 1e-12);
  const move = (e: KeyboardEvent, from: number) => {
    const next = e.key === 'ArrowDown' || e.key === 'ArrowRight' ? from + 1 : e.key === 'ArrowUp' || e.key === 'ArrowLeft' ? from - 1
      : e.key === 'Home' ? 0 : e.key === 'End' ? pages.length - 1 : undefined;
    if (next === undefined) return;
    e.preventDefault(); e.stopPropagation();
    const index = Math.max(0, Math.min(pages.length - 1, next));
    setFocusIndex(index); buttons.current[index]?.focus();
  };
  return <nav className="docattentionlens-filmstrip" data-orientation={orientation} aria-label={label} data-testid="filmstrip">
    <ol>{pages.map((p, i) => {
      const selected = p.pageId === current;
      const massText = p.omitted ? 'not supplied' : p.mass === undefined ? undefined : attentionWeightLabel(p.mass);
      return <li key={p.pageId}>
        <button ref={el => {buttons.current[i] = el;}} type="button" className="docattentionlens-film-page" data-page-id={p.pageId} aria-current={selected ? 'page' : undefined}
          aria-pressed={selected} tabIndex={i === tabStop ? 0 : -1} title={p.detail}
          aria-label={`Page ${p.pageNumber} · ${p.detail}${p.mass !== undefined ? ` · ${massLabel ?? 'attention mass'} ${attentionWeightLabel(p.mass)}` : ''}`}
          onClick={() => onSelect(p.pageId)} onKeyDown={e => move(e, i)} onFocus={() => setFocusIndex(i)} onBlur={() => setFocusIndex(null)}>
          <Thumb page={p} load={loadImages}/>
          <span className="docattentionlens-film-number">{p.pageNumber}</span>
          <span className={`docattentionlens-film-state${p.omitted ? ' is-omitted' : ''}`}>{p.state}{p.score !== undefined && <> · {p.score.toFixed(3)}</>}</span>
          {massText !== undefined && <span className="docattentionlens-film-mass" data-testid="filmstrip-mass" title={p.mass !== undefined ? `${massLabel ?? 'Attention mass'} ${attentionWeightLabel(p.mass)}` : 'Omitted from the model input: no captured image attention'}>
            {p.mass !== undefined && <i style={{width: `${Math.max(2, p.mass / largest * 100)}%`}}/>}<span>{massText}</span></span>}
        </button></li>;})}</ol>
  </nav>;
}
