import { useMemo } from 'react';
import type { DisplaySettings } from '../../api/client';
import type { Generation, Attention } from './GenerationPanel';
import { ATTENTION_FILL, attentionAlpha, heatColor, patchGrid, rangeFloor, rgb, scalePosition, type AttentionMode, type Colormap } from './attentionDisplay';
import { quantityLabel } from './perspective';

export type MapPosition = Pick<Generation['positions'][number],'index'|'bbox'>;
type Props = {positions: MapPosition[]; row: Attention; maximum: number;
  mode: AttentionMode; opacity: number; smooth: boolean; showPatches: boolean;
  /** Colour map for Heatmap and Blend (Overlay stays amber), and where the scale starts. */
  colormap?: Colormap; floor?: DisplaySettings['attention_range_floor'];
  /** The row only marks a selected position; its values are placeholders, not weights. */
  outlineOnly?: boolean; patchTestId?: string};

const OVERLAY_RGB = [parseInt(ATTENTION_FILL.slice(1, 3), 16), parseInt(ATTENTION_FILL.slice(3, 5), 16), parseInt(ATTENTION_FILL.slice(5, 7), 16)];

/** Page attention in source coordinates, drawn in the chosen display mode. */
export function AttentionLayer({positions, row, maximum, mode, opacity, smooth, showPatches, colormap = 'inferno', floor: floorMode = 'zero', outlineOnly = false, patchTestId = 'attention-patch'}: Props) {
  const patches = positions.filter(p => p.bbox && p.index < row.target_count);
  const drawn = mode !== 'original' && !outlineOnly;
  const coloured = mode === 'heatmap' || mode === 'blend';
  const geometry = patches.map(p => p.bbox!.join()).join(';');
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const grid = useMemo(() => patchGrid(patches.map(p => p.bbox!)), [geometry]);
  const floor = rangeFloor(patches.map(p => row.values[p.index]), floorMode);
  const colour = (value: number) => coloured ? heatColor(scalePosition(value, maximum, floor), colormap) : OVERLAY_RGB;
  const alpha = (value: number) => mode === 'heatmap' ? 1 : mode === 'blend' ? opacity : attentionAlpha(value, maximum, opacity, floor);
  const raster = useMemo(() => {
    if (!drawn || !smooth || !grid) return undefined;
    const canvas = document.createElement('canvas');
    canvas.width = grid.columns.length; canvas.height = grid.rows.length;
    const context = canvas.getContext('2d');
    if (!context) return undefined;
    const image = context.createImageData(canvas.width, canvas.height);
    for (const p of patches) {
      const value = row.values[p.index];
      const i = (grid.rows.indexOf(p.bbox![1]) * canvas.width + grid.columns.indexOf(p.bbox![0])) * 4;
      image.data.set([...colour(value), Math.round(255 * alpha(value))], i);
    }
    context.putImageData(image, 0, 0);
    return canvas.toDataURL();
  // Patch geometry is captured in `grid`; values, scale and styling select the raster.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drawn, smooth, grid, row, maximum, mode, opacity, colormap, floor]);
  const label = quantityLabel(row.quantity).toLowerCase();
  return <g data-testid="attention-layer" data-mode={mode} data-colormap={coloured ? colormap : undefined}>
    {raster && <image data-testid="attention-raster" href={raster} x={grid!.x1} y={grid!.y1} width={grid!.x2 - grid!.x1} height={grid!.y2 - grid!.y1} preserveAspectRatio="none" style={{imageRendering: 'auto'}} pointerEvents="none"/>}
    {showPatches && <path data-testid="attention-patch-grid" className="docattentionlens-patch-grid" d={patches.map(p => {const [x1, y1, x2, y2] = p.bbox!; return `M${x1} ${y1}H${x2}V${y2}H${x1}Z`;}).join('')} vectorEffect="non-scaling-stroke" pointerEvents="none"/>}
    {(mode !== 'original' || outlineOnly) && patches.map(p => {
      const [x1, y1, x2, y2] = p.bbox!;
      const value = row.values[p.index];
      const source = p.index === row.source_index;
      return <rect key={p.index} data-testid={patchTestId} data-position-index={p.index} x={x1} y={y1} width={x2 - x1} height={y2 - y1}
        stroke={source ? '#39a9e0' : 'none'} strokeWidth={source ? 2 : 0} vectorEffect="non-scaling-stroke"
        fill={coloured ? rgb(colour(value) as [number, number, number]) : ATTENTION_FILL} fillOpacity={drawn && !raster ? alpha(value) : 0}
        shapeRendering={coloured ? 'crispEdges' : undefined} pointerEvents="none"><title>Image position {p.index}: {label} {value.toFixed(8)}</title></rect>;
    })}
  </g>;
}
