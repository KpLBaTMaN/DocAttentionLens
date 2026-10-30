import { describe, it, expect } from 'vitest';
import { massBreakdown } from '../src/features/document/massBreakdown';

describe('mass breakdown by target role', () => {
  const targets = [
    {index: 0, role: 'special', piece: '<|im_start|>'},
    {index: 1, role: 'prompt', piece: 'system'},
    {index: 2, role: 'image', page_id: 'p1', bbox: [0, 0, 1, 1]},
    {index: 3, role: 'image', page_id: 'p2', bbox: [0, 0, 1, 1]},
    {index: 4, role: 'query', piece: 'profit'},
    {index: 5, role: 'generated', piece: 'True'},
    {index: 6, role: 'query', piece: 'later'},
  ];

  it('covers every target, page groups first and the rest largest first', () => {
    const groups = massBreakdown([.4, .05, .3, .1, .1, .05], targets, {pageId: 'p1', severalPages: true});
    expect(groups.map(g => g.key)).toEqual(['page', 'other_pages', 'special', 'question', 'prompt', 'answer']);
    expect(groups.reduce((sum, g) => sum + g.mass, 0)).toBeCloseTo(1);
    expect(groups.find(g => g.key === 'special')?.top?.piece).toBe('<|im_start|>');
    // Position 6 is past the row and is not counted.
    expect(groups.find(g => g.key === 'question')?.positions).toBe(1);
  });

  it('joins pages on a single page and maps retriever roles', () => {
    const groups = massBreakdown([.5, .2, .3], [
      {index: 0, role: 'visual_patch'}, {index: 1, role: 'query_text', piece: 'q'}, {index: 2, role: 'instruction', piece: 'x'}]);
    expect(groups.map(g => [g.key, g.label])).toEqual([['image', 'Page image'], ['other', 'Prompt and special tokens'], ['question', 'Question']]);
  });

  it('gives signed rows no shares', () => {
    expect(massBreakdown([.5, -.2], [{index: 0, role: 'image'}, {index: 1, role: 'query'}])).toEqual([]);
  });
});
