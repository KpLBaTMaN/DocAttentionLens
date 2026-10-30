import { useState } from 'react';
import type { components } from '../../api/generated/schema';
import { heatColor, rgb, type Colormap } from '../document/attentionDisplay';
import { cellText, groundingRoles, type Table, type TableGrounding } from './tableGeometry';

export type TableAllocation = components['schemas']['TableAllocation'];
type CellMass = components['schemas']['TableCellMass'];
type Measure = 'mass' | 'enrichment' | 'resolution';

const MEASURES: {id: Measure; label: string; title: string}[] = [
  {id: 'mass', label: 'Mass', title: 'Attention mass on each cell (original captured domain)'},
  {id: 'enrichment', label: 'Enrichment', title: "Each cell's share of image attention over its share of page area; 1 is uniform"},
  {id: 'resolution', label: 'Resolution', title: 'Whether an image position lies at least half inside the cell'},
];

const value = (cell: CellMass, measure: Measure) => measure === 'mass' ? cell.mass : measure === 'enrichment' ? cell.enrichment : cell.largest_position_share;
const format = (v?: number | null) => v === null || v === undefined ? 'unmeasured' : v === 0 ? '0' : Math.abs(v) >= .01 ? v.toFixed(3) : v.toExponential(2);
const span = (indices: number[]) => ({start: Math.min(...indices) + 1, end: Math.max(...indices) + 2});

/** Table-aware result drawn as the logical table, one block per cell, with row and column marginals. */
export function TableHeatmap({table, allocation, grounding, showGrounding, colormap = 'inferno', selected, onSelect}: {
  table: Table; allocation: TableAllocation; grounding?: TableGrounding | null; showGrounding: boolean; colormap?: Colormap;
  selected?: string; onSelect: (cellId?: string) => void;
}) {
  const [measure, setMeasure] = useState<Measure>('mass');
  const masses = new Map(allocation.cells.map(c => [c.cell_id, c]));
  const shown = allocation.cells.map(c => value(c, measure)).filter((v): v is number => v !== null && v !== undefined);
  const maximum = measure === 'resolution' ? 1 : Math.max(...shown, 1e-12);
  const roles = showGrounding ? groundingRoles(grounding) : new Map();
  const rowMax = Math.max(...allocation.rows.map(r => r.mass ?? 0), 1e-12);
  const columnMax = Math.max(...allocation.columns.map(c => c.mass ?? 0), 1e-12);
  const imageRegions = allocation.regions.filter(r => !r.region.startsWith('non_image'));
  const nonImage = allocation.regions.filter(r => r.region.startsWith('non_image')).reduce((s, r) => s + r.mass, 0);
  const budget = [...imageRegions.map(r => ({key: r.region, label: r.label, mass: r.mass})), {key: 'non_image', label: 'Non-image positions', mass: nonImage}]
    .filter(r => r.mass > 0);
  const chosen = selected ? masses.get(selected) : undefined;
  const chosenCell = selected ? table.cells.find(c => c.native_id === selected) : undefined;
  const r = allocation.resolution;
  return <div className="docattentionlens-table-heatmap" data-testid="table-heatmap">
    <p className="docattentionlens-table-budget-label">Page budget <small>{allocation.selection}</small></p>
    <div className="docattentionlens-table-budget" role="img" aria-label={budget.map(b => `${b.label} ${format(b.mass)}`).join(', ')}>
      {budget.map(b => <span key={b.key} className={`is-${b.key}`} style={{flexGrow: b.mass}} title={`${b.label}: ${format(b.mass)}`}/>)}
    </div>
    <ul className="docattentionlens-table-budget-legend">{budget.map(b => <li key={b.key}><i className={`is-${b.key}`}/>{b.label} {format(b.mass)}</li>)}</ul>
    <div className="docattentionlens-segmented" role="radiogroup" aria-label="Cell colour">
      {MEASURES.map(m => <button key={m.id} type="button" role="radio" aria-checked={measure === m.id} title={m.title} onClick={() => setMeasure(m.id)}>{m.label}</button>)}
    </div>
    <div className="docattentionlens-table-grid" style={{gridTemplateColumns: `repeat(${table.column_count}, minmax(0, 1fr)) 44px`}} role="grid" aria-label={`${table.title}: cells coloured by ${measure}`}>
      {table.cells.map(cell => {
        const m = masses.get(cell.native_id), v = m ? value(m, measure) : null;
        const rows = span(cell.row_indices), columns = span(cell.column_indices);
        const t = v === null || v === undefined ? 0 : measure === 'resolution' ? (m!.isolable ? 1 : .25) : Math.min(1, v / maximum);
        const role = roles.get(cell.native_id);
        const {text} = cellText(cell);
        const background = v === null || v === undefined ? undefined : rgb(heatColor(t, colormap));
        return <button key={cell.native_id} type="button" role="gridcell" data-cell-id={cell.native_id} data-mass={m?.mass ?? ''}
          className={`docattentionlens-table-cell-block${m && !m.isolable ? ' is-shared' : ''}${role ? ` is-${role.kind}` : ''}${selected === cell.native_id ? ' is-selected' : ''}${v === null || v === undefined ? ' is-unmeasured' : ''}`}
          style={{gridRow: `${rows.start} / ${rows.end}`, gridColumn: `${columns.start} / ${columns.end}`, background, color: t > .55 ? '#111' : undefined}}
          aria-pressed={selected === cell.native_id} onClick={() => onSelect(selected === cell.native_id ? undefined : cell.native_id)}
          title={`${text || 'no text'} · mass ${format(m?.mass)} · enrichment ${format(m?.enrichment)}${m && !m.isolable ? ' · shared positions' : ''}${role ? ` · ${role.label}` : ''}`}>
          {role && <b>{role.label}</b>}<span>{text}</span>
        </button>;
      })}
      {allocation.rows.map(row => <span key={row.band_id} className="docattentionlens-table-margin is-row" style={{gridRow: row.index + 1, gridColumn: table.column_count + 1}}
        title={`Table row ${row.index + 1}: ${format(row.mass)}`}><i style={{width: `${(row.mass ?? 0) / rowMax * 100}%`}}/></span>)}
      {allocation.columns.map(column => <span key={column.band_id} className="docattentionlens-table-margin is-column" style={{gridRow: table.row_count + 1, gridColumn: column.index + 1}}
        title={`Table column ${column.index + 1}: ${format(column.mass)}`}><i style={{height: `${(column.mass ?? 0) / columnMax * 100}%`}}/></span>)}
    </div>
    <p className="docattentionlens-muted docattentionlens-table-scale">{measure === 'resolution' ? 'Bright: isolable · dim: shares every position with neighbours.' : `0 → ${format(maximum)} · ${measure === 'mass' ? 'cell maximum' : 'enrichment maximum'}`} Hatched cells have no image position of their own ({r.isolable_cells} of {r.cell_count} isolable). Bars at the right and bottom are table row and column masses.</p>
    {chosen && chosenCell && <dl className="docattentionlens-table-grounding" data-testid="table-cell-detail">
      <dt>Cell</dt><dd>row {chosenCell.row_indices.map(i => i + 1).join('–')}, column {chosenCell.column_indices.map(i => i + 1).join('–')} · {cellText(chosenCell).text || 'no text'}{cellText(chosenCell).source === 'pdf' && <small>PDF text layer</small>}</dd>
      <dt>Roles</dt><dd>{chosenCell.roles.map(x => `${x.role.replaceAll('_', ' ')} (${x.status.replaceAll('_', ' ')})`).join(', ') || 'none'}</dd>
      <dt>Mass</dt><dd>{format(chosen.mass)}</dd>
      <dt>Enrichment</dt><dd>{format(chosen.enrichment)} <small>share of image attention ÷ share of page area ({(chosen.area_share * 100).toFixed(2)}%)</small></dd>
      <dt>Positions</dt><dd>{chosen.touching_positions} touching · largest share {chosen.largest_position_share.toFixed(2)} · {chosen.isolable ? 'isolable' : 'shared with neighbours'}</dd>
    </dl>}
  </div>;
}
