// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, render } from '@testing-library/react';
import { PageCanvas, type ElementRegions } from '../src/features/document/PageCanvas';
import type { DisplaySettings } from '../src/api/client';
import type { AttentionMode } from '../src/features/document/attentionDisplay';
import { HOT, PAGE, RETRIEVAL_ID, PAGE_ID, SIMILARITY_NEGATIVE, SIMILARITY_POSITIVE, SIMILARITY_WINNER, TABLE_ID, TEXT_ID,
  elementMasses, elementRecords, generationPositions, generatorPatchIndex, generatorRow, inferno, overlayOpacity, pageRecord,
  patchValues, retrieverPositions, similarityValues } from '../fixtures/attention';

// The canvas starts at 600 × 680 before any resize. A 400 × 300 page fits at scale 1.5,
// centred vertically: offset (0, 115). Every overlay must share that one transform.

afterEach(cleanup);

const SETTINGS: DisplaySettings = {zoom: 1, center_x: null, center_y: null, show_boxes: true, show_support: false, show_reading_order: true,
  show_geometry_flags: true, attention_opacity: .65, attention_smooth: false, attention_show_patches: false, similarity_opacity: .65} as DisplaySettings;
const row = generatorRow(patchValues(6));
const masses = elementMasses(row);
const regionMaximum = Math.max(...Object.values(masses));

type CanvasProps = Parameters<typeof PageCanvas>[0];
function canvas({settings = SETTINGS, attention = true, regions, evidence, mode = 'overlay' as AttentionMode, selected = null as string | null, loaded = true, scaleMax = HOT, pick}:
  {settings?: DisplaySettings; attention?: boolean; regions?: ElementRegions; evidence?: CanvasProps['evidence']; mode?: AttentionMode; selected?: string | null; loaded?: boolean; scaleMax?: number; pick?: CanvasProps['pick']} = {}) {
  const view = render(<PageCanvas page={pageRecord} elements={elementRecords} settings={settings} selected={selected} source={null} target={null}
    support={new Set([TABLE_ID])} onView={() => {}} scaleMax={scaleMax} attentionMode={mode} regions={regions} evidence={evidence} pick={pick}
    attention={attention ? {positions: generationPositions.filter(p => p.page_id === PAGE_ID), row} : undefined}/>);
  const image = view.container.querySelector('image')!;
  if (loaded) fireEvent.load(image);
  return {view, image, all: (testId: string) => [...view.container.querySelectorAll(`[data-testid="${testId}"]`)]};
}

describe('page overlay alignment', () => {
  it('draws nothing quantitative until the page image has loaded', () => {
    const {all, image} = canvas({loaded: false});
    expect(all('attention-patch')).toHaveLength(0);
    fireEvent.load(image);
    expect(all('attention-patch')).toHaveLength(12);
  });

  it('places the page image and its patches under one fitted transform', () => {
    const {image, all} = canvas();
    const group = image.parentElement!;
    expect(group.getAttribute('transform')).toBe('translate(0 115) scale(1.5)');
    expect([image.getAttribute('width'), image.getAttribute('height')]).toEqual([String(PAGE.width), String(PAGE.height)]);
    expect(all('attention-layer')[0].parentElement).toBe(group);
    // Patch 6 (row 1, column 2) therefore covers display pixels x 300–450, y 265–415.
    const hot = all('attention-patch').find(p => p.getAttribute('data-position-index') === String(generatorPatchIndex(6)))!;
    expect(['x', 'y', 'width', 'height'].map(a => Number(hot.getAttribute(a)))).toEqual([200, 100, 100, 100]);
  });

  it('zoom and pan move the page and its patches together', () => {
    // Zoom 2 about source point (300, 200): scale 3, and that point lands at the viewport centre (300, 340).
    const {image, all} = canvas({settings: {...SETTINGS, zoom: 2, center_x: 300, center_y: 200}});
    expect(image.parentElement!.getAttribute('transform')).toBe('translate(-600 -260) scale(3)');
    expect(all('attention-layer')[0].parentElement).toBe(image.parentElement);
  });
});

describe('document-aware element regions', () => {
  const regions = (mode: AttentionMode): ElementRegions => ({masses, maximum: regionMaximum, labelled: new Set([TABLE_ID, TEXT_ID]), mode, opacity: .65});

  it('fills each element box with its own mass, scaled to the element maximum', () => {
    const {all} = canvas({attention: false, regions: regions('overlay')});
    const drawn = all('element-region').map(r => ({id: r.getAttribute('data-element-id'), mass: Number(r.getAttribute('data-mass')),
      opacity: Number(r.getAttribute('fill-opacity')), box: ['x', 'y', 'width', 'height'].map(a => Number(r.getAttribute(a)))}));
    // Text owns patches 6, 7, 10 and 11, including the hot patch 6: 0.24 + 0.032 + 0.044 + 0.048.
    expect(masses[TEXT_ID]).toBeCloseTo(.364, 12);
    expect(masses[TABLE_ID]).toBeCloseTo(.004 + .008, 12);
    expect(drawn).toEqual([
      {id: TABLE_ID, mass: masses[TABLE_ID], opacity: expect.closeTo(overlayOpacity(masses[TABLE_ID], regionMaximum), 12), box: [0, 0, 200, 100]},
      {id: TEXT_ID, mass: masses[TEXT_ID], opacity: .65, box: [200, 100, 200, 200]},
    ]);
    expect(all('element-region-label').map(l => l.textContent)).toEqual(['table · 0.0120', 'text · 0.364']);
  });

  it('heatmap regions use the inferno colour of mass ÷ element maximum', () => {
    const {all} = canvas({attention: false, regions: regions('heatmap')});
    expect(all('element-region').map(r => r.getAttribute('fill'))).toEqual([
      `rgb(${inferno(masses[TABLE_ID] / regionMaximum).join(',')})`, 'rgb(252,255,164)']);
  });

  it('the selected element shows its patches instead of a region fill', () => {
    const {all} = canvas({attention: false, regions: regions('overlay'), selected: TEXT_ID});
    expect(all('element-region').map(r => r.getAttribute('data-element-id'))).toEqual([TABLE_ID]);
  });

  it('Original hides mass fills but keeps every boundary with its value', () => {
    const {view, all} = canvas({attention: false, regions: regions('original')});
    expect(all('element-region')).toHaveLength(0);
    const boxes = [...view.container.querySelectorAll('.docattentionlens-box')].map(b => b.getAttribute('aria-label'));
    expect(boxes).toEqual([`Element 1 · table · element attention mass ${masses[TABLE_ID].toPrecision(4)}`, `Element 2 · text · element attention mass ${masses[TEXT_ID].toPrecision(4)}`]);
  });
});

describe('MaxSim similarity', () => {
  const similarities = similarityValues(3);
  const evidence = {positions: retrieverPositions, row: {capture_id: RETRIEVAL_ID, page_id: PAGE_ID, query_index: 3, similarities, winner_index: SIMILARITY_WINNER, maximum: .9,
    units: 'dot product of captured vectors', normalization: 'fixture'}};

  it('colours each patch by sign with opacity |value| ÷ largest |value|, and outlines the winner', () => {
    // The workspace's automatic similarity scale is the row's largest absolute value.
    const {all} = canvas({attention: false, evidence, scaleMax: .9});
    const drawn = all('similarity-patch').map(r => ({index: Number(r.getAttribute('data-position-index')), fill: r.getAttribute('fill'),
      opacity: Number(r.getAttribute('fill-opacity')), stroke: r.getAttribute('stroke')}));
    expect(drawn.map(d => d.index)).toEqual(Array.from({length: 12}, (_, k) => k));
    drawn.forEach(d => {
      const value = similarities[d.index]!;
      expect(d.fill).toBe(value >= 0 ? SIMILARITY_POSITIVE : SIMILARITY_NEGATIVE);
      expect(d.opacity).toBeCloseTo(Math.min(.65, Math.abs(value) / .9 * .65), 12);
    });
    expect(drawn.filter(d => d.stroke === '#ffc400').map(d => d.index)).toEqual([SIMILARITY_WINNER]);
    expect(drawn.find(d => d.index === SIMILARITY_WINNER)!.opacity).toBe(.65);
  });
});

describe('clicking the page', () => {
  const patches = retrieverPositions.filter(p => p.bbox).map(p => ({index: p.index, bbox: p.bbox!}));
  // Patch 6 (row 1, column 2) covers display pixels x 300–450, y 265–415 and lies inside the text element.
  const click = (svg: Element, x: number, y: number, moveTo = {x, y}) => {
    fireEvent.pointerDown(svg, {button: 0, clientX: x, clientY: y, pointerId: 1});
    fireEvent.pointerMove(svg, {clientX: moveTo.x, clientY: moveTo.y, pointerId: 1});
    fireEvent.pointerUp(svg, {button: 0, clientX: moveTo.x, clientY: moveTo.y, pointerId: 1});
  };

  it('chooses the patch under the pointer, even inside an element box', () => {
    const picked: number[] = [];
    const {view} = canvas({pick: {positions: patches, onPick: index => picked.push(index)}});
    const svg = view.container.querySelector('svg')!;
    svg.setPointerCapture = () => {};
    click(svg, 375, 340);
    expect(picked).toEqual([patches.find(p => p.bbox[0] === 200 && p.bbox[1] === 100)!.index]);
    expect(view.container.querySelector('.docattentionlens-canvas')!.classList.contains('is-pickable')).toBe(true);
  });

  it('a drag pans instead of picking, and element boxes are not controls', () => {
    const picked: number[] = [];
    const {view} = canvas({pick: {positions: patches, onPick: index => picked.push(index)}});
    const svg = view.container.querySelector('svg')!;
    svg.setPointerCapture = () => {};
    click(svg, 375, 340, {x: 395, y: 360});
    expect(picked).toEqual([]);
    expect(view.container.querySelectorAll('.docattentionlens-box[role=button], .docattentionlens-box[tabindex]')).toHaveLength(0);
  });

  it('without patches to choose, the page is not pickable', () => {
    const {view} = canvas();
    expect(view.container.querySelector('.docattentionlens-canvas')!.classList.contains('is-pickable')).toBe(false);
  });
});
