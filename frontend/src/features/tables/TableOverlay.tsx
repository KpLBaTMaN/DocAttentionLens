import { cellText, groundingRoles, type Table, type TableGrounding } from './tableGeometry';

/** Table structure over the original page, in page pixels; orientation only, no attention values. */
export function TableOverlay({tables, grounding, showGrid, showGrounding, scale, highlight}: {
  tables: Table[]; grounding?: TableGrounding | null; showGrid: boolean; showGrounding: boolean;
  /** A cell chosen in the Table heatmap, outlined on the page. */
  highlight?: string;
  /** Screen pixels per page pixel, so labels keep a readable size at any zoom. */
  scale: number;
}) {
  const roles = showGrounding ? groundingRoles(grounding) : new Map();
  const font = 12 / scale;
  return <g className="docattentionlens-table-overlay" data-testid="table-overlay">
    {tables.map(table => {
      const b = table.bbox, w = table.wrap_bbox;
      const grounded = showGrounding && grounding?.table_id === table.table_id;
      return <g key={table.key} data-table-id={table.table_id}>
        <rect className="docattentionlens-table-wrap" x={w.x1} y={w.y1} width={w.x2 - w.x1} height={w.y2 - w.y1} vectorEffect="non-scaling-stroke">
          <title>{table.title}: table with its caption and notes</title></rect>
        <rect className="docattentionlens-table-outline" x={b.x1} y={b.y1} width={b.x2 - b.x1} height={b.y2 - b.y1} vectorEffect="non-scaling-stroke"/>
        {showGrid && table.cells.map(cell => {
          const c = cell.bbox, {text, source} = cellText(cell);
          return <rect key={cell.native_id} className="docattentionlens-table-cell" data-cell-id={cell.native_id} x={c.x1} y={c.y1} width={c.x2 - c.x1} height={c.y2 - c.y1} vectorEffect="non-scaling-stroke">
            <title>{`Row ${cell.row_indices.map(i => i + 1).join('–')} · column ${cell.column_indices.map(i => i + 1).join('–')} · ${text ? `${text}${source === 'pdf' ? ' (PDF text)' : ''}` : 'no text'}`}</title></rect>;
        })}
        {grounded && table.cells.filter(cell => roles.has(cell.native_id)).sort((a, z) => order(roles.get(a.native_id)!.kind) - order(roles.get(z.native_id)!.kind)).map(cell => {
          const c = cell.bbox, role = roles.get(cell.native_id)!;
          return <g key={`ground-${cell.native_id}`} data-testid="grounding-cell" data-role={role.kind} data-cell-id={cell.native_id}>
            <rect className={`docattentionlens-grounding is-${role.kind}`} x={c.x1} y={c.y1} width={c.x2 - c.x1} height={c.y2 - c.y1} vectorEffect="non-scaling-stroke"/>
            <text className={`docattentionlens-grounding-label is-${role.kind}`} x={c.x1 + 2 / scale} y={c.y1 - 3 / scale} fontSize={font}>{role.label}</text>
          </g>;
        })}
        {table.cells.filter(cell => cell.native_id === highlight).map(cell => <rect key={`highlight-${cell.native_id}`} className="docattentionlens-table-highlight" data-testid="table-cell-highlight"
          data-cell-id={cell.native_id} x={cell.bbox.x1} y={cell.bbox.y1} width={cell.bbox.x2 - cell.bbox.x1} height={cell.bbox.y2 - cell.bbox.y1} vectorEffect="non-scaling-stroke"/>)}
      </g>;
    })}
  </g>;
}

/** Support first and the answer last, so the answer cell's outline is on top. */
const order = (kind: string) => ({support: 0, column_path: 1, row_path: 2, answer: 3} as Record<string, number>)[kind] ?? 0;
