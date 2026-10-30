import { Link } from 'react-router-dom';
import type { Element } from '../../api/client';
import type { Attention } from './GenerationPanel';
import { attentionWeightLabel } from './attentionDisplay';
import { elementQuantityLabel } from './documentGrouping';
import { elementCoverage, shareLabel, type SourceReach } from './ElementSummary';
import { formatProfile } from './ElementProfile';
import { enrichmentPosition, pageEnrichment, type Enrichment } from './elementEnrichment';
import { useBrowserChoice, useBrowserPreference } from './WorkspaceControls';

type Position = {index: number; role?: string; bbox?: number[] | null; page_id?: string | null; ownership?: Record<string, number> | null};
/**
 * What an element's value is: its share of the row's image attention (every supplied page), its share of the
 * whole row, or its Enrichment, the image share over its share of image positions.
 */
type ShareOf = 'image' | 'row' | 'enrichment';
const isShareOf = (value: unknown): value is ShareOf => value === 'image' || value === 'row' || value === 'enrichment';

/** The ratio in words: what is divided by what, the most it can reach, and how little area it may rest on. */
function enrichmentTitle(x: Enrichment) {
  return `${shareLabel(x.imageShare, 1)} of the row's image attention ÷ ${shareLabel(x.areaShare, 1)} of the image positions = ${formatProfile(x.value, 'enrichment')}`
    + `; at most ${formatProfile(x.ceiling, 'enrichment')}, with all image attention on it`
    + (x.belowOnePosition ? ` · covers ${x.positions.toFixed(2)} of one image position, so the ratio rests on a fragment of one position` : '');
}

/** A log-scale bar about 1× (attention in proportion to area); hatching marks values beyond the element's reach. */
function EnrichmentBar({x}: {x: Enrichment}) {
  const at = enrichmentPosition(x.value!), cap = x.ceiling !== null && x.ceiling < 4 ? enrichmentPosition(x.ceiling) : undefined;
  return <i className="docattentionlens-enrichment-bar" aria-hidden="true" data-testid="enrichment-bar">
    {cap !== undefined && <i className="docattentionlens-enrichment-ceiling" style={{left: `${cap}%`}}/>}
    <i className={`docattentionlens-enrichment-fill ${x.value! >= 1 ? 'is-above' : 'is-below'}`} style={{left: `${Math.min(at, 50)}%`, width: `${Math.abs(at - 50)}%`}}/>
  </i>;
}

/** Short names for attention that no annotated element owns, as the rail is narrow. */
function outsideLabel(bucket: string) {
  if (bucket === 'background') return 'background';
  if (bucket === 'non_page') return 'prompt and special tokens';
  if (bucket === 'unreviewed_geometry') return 'unreviewed geometry';
  if (bucket.startsWith('non_image:')) return `non-image ${bucket.slice('non_image:'.length)}`;
  return bucket.replaceAll('_', ' ');
}

/** The page's element list (ledger or Details › Element), with each element's share of the shown row. */
export function PageElements({elements, selectedId, onSelect, row, positions, pageId, reach, support, onDetails, markHref}: {
  /** The page's elements in reading order. */
  elements: Element[]; selectedId?: string | null; onSelect: (element: Element) => void;
  /** The shown row; omitted when it has no element values for this page (column views, MaxSim, an unsupplied page). */
  row?: Attention; positions: Position[]; pageId: string; reach?: SourceReach;
  /** Designated supporting elements: an evaluation annotation shown for orientation only. */
  support?: Set<string>; onDetails?: () => void;
  /** An uploaded document's Mark elements screen, offered while the page has no elements. */
  markHref?: string;
}) {
  const [byMass, setByMass] = useBrowserPreference('docattentionlens-element-order-mass', false);
  const [shareOf, setShareOf] = useBrowserChoice<ShareOf>('docattentionlens-element-share', 'image', isShareOf);
  const coverage = row ? elementCoverage(row, positions, pageId, reach) : undefined;
  const values = coverage?.masses.size ? coverage : undefined;
  const largest = values ? Math.max(...[...values.masses.values()].map(Math.abs), 1e-12) : 1;
  const quantity = row ? elementQuantityLabel(row.quantity) : '';
  // A row with no image attention at all (text-target blocks) has only the row share.
  const imageMass = row && row.image_mass > 0 ? row.image_mass : undefined;
  // Enrichment uses the element profile's denominators: image attention and the measured image positions.
  const enrichment = values && imageMass !== undefined ? pageEnrichment(row!, positions, pageId, reach) : undefined;
  const ofEnrichment = shareOf === 'enrichment' && enrichment !== undefined;
  const ofImage = shareOf === 'image' && imageMass !== undefined;
  const whole = ofImage ? imageMass! : row?.total_mass ?? 0;
  const shown = values && byMass
    ? [...elements].sort((a, b) => (values.masses.has(b.element_id) ? 1 : 0) - (values.masses.has(a.element_id) ? 1 : 0)
      || (values.masses.get(b.element_id) ?? 0) - (values.masses.get(a.element_id) ?? 0))
    : elements;
  // Attention no element owns: page background and unreviewed geometry, positions off every page, other pages.
  const outside = values && row ? [
    ...row.allocations.filter(a => !a.element_id && (a.page_id === pageId || !a.page_id)).map(a => ({key: a.bucket, label: outsideLabel(a.bucket), mass: a.mass})),
    ...(() => {const other = row.allocations.filter(a => a.page_id && a.page_id !== pageId).reduce((sum, a) => sum + a.mass, 0);
      return other ? [{key: 'other_pages', label: 'other pages', mass: other}] : [];})(),
  ] : [];
  const value = (e: Element) => {
    if (!values) return null;
    const mass = values.masses.get(e.element_id);
    if (mass !== undefined) {
      const reached = values.reached.get(e.element_id) ?? 0, owned = values.owned.get(e.element_id) ?? 0;
      const title = `${quantity} ${attentionWeightLabel(mass)} · ${imageMass !== undefined ? `${shareLabel(mass, imageMass)} of the row's image attention · ` : ''}${shareLabel(mass, row!.total_mass)} of the row${values.partial(e.element_id) ? ` · only ${reached} of its ${owned} patches come before source ${reach!.source}; the mass covers those only` : ''}`;
      if (ofEnrichment) {
        const x = enrichment!.get(e.element_id);
        return x?.value != null ? {text: formatProfile(x.value, 'enrichment'), enrichment: x, title: `${enrichmentTitle(x)} · ${title}`}
          : {text: '—', title: `No measured image position: no Enrichment · ${title}`};
      }
      return {text: shareLabel(mass, whole), width: Math.abs(mass) / largest, title};
    }
    if (e.geometry_flags.length) return {text: '—', title: 'Geometry flagged: its patches are held in Unreviewed element geometry'};
    if (values.owned.has(e.element_id)) return {text: 'not reached', unreached: true,
      title: `Its patches lie after ${reach ? `source ${reach.source}` : 'the source'}, which cannot attend to them: not measured, not zero`};
    return {text: '—', title: 'No mapped positions: missing, not zero'};
  };
  return <>
    <div className="docattentionlens-elements-head"><h2>Page elements</h2>
      {values && <div className="docattentionlens-element-order" role="group" aria-label="Element order">
        <button type="button" className="docattentionlens-button" aria-pressed={!byMass} onClick={() => setByMass(false)}>Reading</button>
        <button type="button" className="docattentionlens-button" aria-pressed={byMass} onClick={() => setByMass(true)}>Mass</button></div>}</div>
    {values && imageMass !== undefined && <div className="docattentionlens-element-order docattentionlens-element-share" role="group" aria-label="Element values">
      <button type="button" className="docattentionlens-button" aria-pressed={ofImage} title="Each element's share of the attention that reaches the supplied page images"
        onClick={() => setShareOf('image')}>Image share</button>
      <button type="button" className="docattentionlens-button" aria-pressed={!ofImage && !ofEnrichment} title="Each element's share of the whole attention row, text tokens included"
        onClick={() => setShareOf('row')}>Row share</button>
      <button type="button" className="docattentionlens-button" aria-pressed={ofEnrichment} title="Each element's share of image attention ÷ its share of image positions; 1× is attention in proportion to area"
        onClick={() => setShareOf('enrichment')}>Enrichment</button></div>}
    <p>{values ? ofEnrichment
      ? "Enrichment: share of the row's image attention ÷ share of image positions. 1× is in proportion to area; no element can exceed 1 ÷ its area share. "
      : `${quantity}: share of the ${ofImage ? "row's image attention (every supplied page)" : 'shown row'}. ` : ''}Native depth does not assert parentage.</p>
    {!elements.length && markHref && <p data-testid="mark-elements-link">This page has no marked elements. <Link to={markHref}>Mark elements</Link> to read attention by heading, table and text; marked elements apply to questions asked after marking.</p>}
    {reach && values && reach.seen < reach.total && <p className="docattentionlens-source-reach" data-testid="rail-reach">
      Source {reach.source} sees {reach.seen} of {reach.total} patches{reach.onWholePage && <> · <button className="docattentionlens-link-button" onClick={reach.onWholePage}>Use the last token</button></>}</p>}
    <ol>{shown.map(e => {
      const isSelected = selectedId === e.element_id, v = value(e);
      return <li key={e.key}><button data-testid="outline-element" data-native-id={e.native_id}
        className={`docattentionlens-element-choice ${isSelected ? 'is-selected' : ''} ${v?.unreached ? 'is-unreached' : ''} ${v?.enrichment?.belowOnePosition ? 'is-fragment' : ''}`}
        style={{paddingLeft: `${8 + (byMass && values ? 0 : Math.min(e.level ?? 0, 5) * 8)}px`}} aria-pressed={isSelected} onClick={() => onSelect(e)}>
        <strong className="docattentionlens-element-head-line"><b>{e.reading_order ?? '—'} · {e.element_type}</b>
          {v && <em className="docattentionlens-element-value" data-testid="element-value" title={v.title}>{v.text}</em>}</strong>
        <span>Element {e.native_id} · {e.element_category}{e.geometry_flags.length ? ' · flagged' : ''}
          {support?.has(e.element_id) && <em className="docattentionlens-support-tag" title="Designated supporting element: an evaluation annotation, never supplied to the model">support</em>}
          {values?.partial(e.element_id) && <em className="docattentionlens-partial-tag" data-testid="partial-tag">partial {values.reached.get(e.element_id) ?? 0}/{values.owned.get(e.element_id)}</em>}
          {v?.enrichment?.belowOnePosition && <em className="docattentionlens-fragment-tag" data-testid="fragment-tag"
            title={`Owns ${v.enrichment.positions.toFixed(2)} of one image position: its Enrichment divides by almost no area and is unstable`}>under 1 position</em>}</span>
        {isSelected && v?.enrichment && <span className="docattentionlens-enrichment-ratio" data-testid="enrichment-ratio">
          {shareLabel(v.enrichment.imageShare, 1)} of image attention ÷ {shareLabel(v.enrichment.areaShare, 1)} of image positions · at most {formatProfile(v.enrichment.ceiling, 'enrichment')}
          {v.enrichment.belowOnePosition && ` · covers ${v.enrichment.positions.toFixed(2)} of one position`}</span>}
        {v?.enrichment && <EnrichmentBar x={v.enrichment}/>}
        {v?.width !== undefined && <i className="docattentionlens-element-bar" aria-hidden="true" style={{width: `${Math.max(v.width * 100, v.width > 0 ? 2 : 0)}%`}}/>}
      </button></li>;})}</ol>
    {!!outside.length && <p className="docattentionlens-elements-outside" data-testid="outside-elements">Outside elements · {outside.map((o, i) => {
      const x = ofEnrichment ? enrichment!.get(o.key) : undefined;
      return <span key={o.key} title={x?.value != null ? `${enrichmentTitle(x)} · ${attentionWeightLabel(o.mass)}` : `${attentionWeightLabel(o.mass)}`}>{i ? ' · ' : ''}{o.label}
        {!ofEnrichment ? <> <strong>{shareLabel(o.mass, row!.total_mass)}</strong></>
          : x?.value != null ? <> <strong>{formatProfile(x.value, 'enrichment')}</strong></>
          : o.key === 'other_pages' ? <> <strong>—</strong></> : ': not image positions'}</span>;})}</p>}
    {values && onDetails && <button className="docattentionlens-link-button docattentionlens-elements-details" onClick={onDetails}>Patch contributions and full table ↓</button>}
  </>;
}
