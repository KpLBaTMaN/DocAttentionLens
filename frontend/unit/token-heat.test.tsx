// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render } from '@testing-library/react';
import { TokenStrip, type Token } from '../src/features/document/TokenStrip';
import { PromptTokens, type PromptToken } from '../src/features/document/PromptTokens';

afterEach(cleanup);
globalThis.ResizeObserver ??= class { observe() {} unobserve() {} disconnect() {} } as unknown as typeof ResizeObserver;

// Amber attention, lilac positive and blue negative similarity, at the page heatmap's opacity rule: weight / maximum × opacity.
const AMBER = 'rgba(217,154,54,0.325)', LILAC = 'rgba(176,128,209,0.325)', BLUE = 'rgba(36,126,213,0.325)';
const background = (el: Element) => (el as HTMLElement).style.background.replace(/\s/g, '');
const pieces = ['<bos>', '▁What', 'ever', '▁year', '?'];
const roles = ['bos', 'query', 'query', 'query', 'query'];
const strip: Token[] = pieces.map((piece, index) => ({index, piece, role: roles[index]}));
const prompt: PromptToken[] = strip.map(t => ({...t, part: t.index ? 'question' : 'template'}));

describe('token strips and the prompt shade and join tokens the same way', () => {
  it('TokenStrip: amber weights, signed similarity colours, no fill for zero or missing values', () => {
    render(<TokenStrip label="Question" tokens={strip} values={[.5, 0, null, .5, -.5]} scaleMax={1} opacity={.65}/>);
    expect([...document.querySelectorAll('.docattentionlens-token')].map(background)).toEqual([AMBER, '', '', AMBER, '']);
    cleanup();
    render(<TokenStrip label="Question" tokens={strip} values={[.5, 0, null, .5, -.5]} scaleMax={1} opacity={.65} signed/>);
    expect([...document.querySelectorAll('.docattentionlens-token')].map(background)).toEqual([LILAC, '', '', LILAC, BLUE]);
  });

  it('TokenStrip: word pieces join, special roles never do', () => {
    render(<TokenStrip label="Question" tokens={strip}/>);
    const chips = [...document.querySelectorAll('.docattentionlens-token')];
    expect(chips.map(c => c.classList.contains('is-continuation'))).toEqual([false, false, true, false, true]);
    expect(chips.map(c => c.classList.contains('is-special'))).toEqual([true, false, false, false, false]);
  });

  it('PromptTokens: the same colours and joining', () => {
    const values = [.5, 0, null, .5, -.5];
    render(<PromptTokens tokens={prompt.map((t, i) => ({...t, value: values[i]}))} scaleMax={1} opacity={.65}/>);
    let chips = [...document.querySelectorAll('[data-testid="token-heat"]')];
    expect(chips.map(background)).toEqual([AMBER, '', '', AMBER, '']);
    // The first token has nothing before it to leave a gap from.
    expect(chips.map(c => c.classList.contains('is-joined'))).toEqual([true, false, true, false, true]);
    expect(chips.map(c => c.classList.contains('is-special'))).toEqual([true, false, false, false, false]);
    cleanup();
    render(<PromptTokens tokens={prompt.map((t, i) => ({...t, value: values[i]}))} scaleMax={1} opacity={.65} signed/>);
    chips = [...document.querySelectorAll('[data-testid="token-heat"]')];
    expect(chips.map(background)).toEqual([LILAC, '', '', LILAC, BLUE]);
  });
});
