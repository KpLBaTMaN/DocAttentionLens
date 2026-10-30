// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { PromptTokens, type PromptToken } from '../src/features/document/PromptTokens';

afterEach(cleanup);
// happy-dom has no ResizeObserver; the prompt only reads sizes from it.
globalThis.ResizeObserver ??= class { observe() {} unobserve() {} disconnect() {} } as unknown as typeof ResizeObserver;

// A long template run, a page image, another template token, the question and the generated answer.
const template = Array.from({length: 12}, (_, i): PromptToken => ({index: i, piece: `t${i}`, role: 'special', part: 'template', value: .01}));
const image = Array.from({length: 4}, (_, i): PromptToken => ({index: 12 + i, piece: '<img>', role: 'image', part: 'image', value: .1, page: 'Page 3'}));
const tokens: PromptToken[] = [...template, ...image,
  {index: 16, piece: '<|vision_end|>', role: 'special', part: 'template', value: 0},
  {index: 17, piece: '▁What', role: 'query', part: 'question', value: .05},
  {index: 18, piece: '▁year', role: 'query', part: 'question', value: .15},
  {index: 19, piece: '2', role: 'generated', part: 'answer', value: .2},
];

describe('PromptTokens', () => {
  it('names itself and the source, and labels a row per part', () => {
    render(<PromptTokens tokens={tokens} answerLabel="generated answer" selected={t => t.index === 19} onSelect={() => {}}/>);
    expect(screen.getByText('Prompt')).toBeTruthy();
    expect(screen.getByText(/what the model read · 20 tokens/)).toBeTruthy();
    expect(screen.getByTestId('prompt-readout').textContent).toContain('“2”');
    expect(screen.getByTestId('prompt-readout').textContent).toContain('generated answer · position 19');
    const labels = [...document.querySelectorAll('.docattentionlens-prompt-row-label')].map(l => l.textContent);
    expect(labels).toEqual(['Context', 'Question', 'Generated answer']);
    expect(screen.getByTestId('prompt-image').textContent).toContain('Page 3 · 4 patches');
  });

  it('folds a long template run into one chip that opens on request', () => {
    render(<PromptTokens tokens={tokens} selected={t => t.index === 19} onSelect={() => {}}/>);
    expect(screen.queryByText('t5')).toBeNull();
    fireEvent.click(screen.getByRole('button', {name: 'Show 12 template tokens'}));
    expect(screen.getByText('t5')).toBeTruthy();
  });

  it('states each part\'s share of the source row in the bar', () => {
    render(<PromptTokens tokens={tokens} selected={t => t.index === 19} onSelect={() => {}}/>);
    // 0.12 template + 0.4 image + 0.2 question + 0.2 answer = 0.92.
    const map = screen.getByRole('img', {name: /Prompt parts in order/});
    expect(map.getAttribute('aria-label')).toContain('page 3 43%');
    expect(map.getAttribute('aria-label')).toContain('question 22%');
  });

  it('hides to its header and resizes, hiding when dragged to the top', () => {
    const onOpen = vi.fn(), onHeight = vi.fn();
    const {rerender} = render(<PromptTokens tokens={tokens} onSelect={() => {}} onOpen={onOpen} onHeight={onHeight}/>);
    fireEvent.click(screen.getByRole('button', {name: 'Hide prompt'}));
    expect(onOpen).toHaveBeenLastCalledWith(false);
    const handle = screen.getByRole('separator', {name: 'Resize prompt'});
    fireEvent.keyDown(handle, {key: 'End'});
    expect(onHeight).toHaveBeenCalled();
    fireEvent.keyDown(handle, {key: 'Home'});
    expect(onOpen).toHaveBeenLastCalledWith(false);
    rerender(<PromptTokens tokens={tokens} onSelect={() => {}} open={false} onOpen={onOpen} onHeight={onHeight}/>);
    expect(document.querySelector('.docattentionlens-prompt-tokens-text')).toBeNull();
    expect(screen.getByText(/hidden · P or drag the edge to show/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', {name: 'Show prompt'}));
    expect(onOpen).toHaveBeenLastCalledWith(true);
  });
});
