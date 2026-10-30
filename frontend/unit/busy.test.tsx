// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { Busy, BusySweep } from '../src/app/Busy';
import { RetrieverRowDetails } from '../src/features/document/MethodResult';
import type { RetrieverAttention } from '../src/features/document/useRetrieverAttention';

afterEach(cleanup);

describe('working indicator', () => {
  it('is decorative: status text and button names read exactly as before', () => {
    render(<><p role="status"><Busy/>Calculating selected method from saved evidence…</p><button><Busy/>Run ColQwen2.5 on 1 page</button><BusySweep/></>);
    expect(screen.getByRole('status').textContent).toBe('Calculating selected method from saved evidence…');
    expect(screen.getByRole('button', {name: 'Run ColQwen2.5 on 1 page'})).toBeTruthy();
    for (const mark of [...screen.getAllByTestId('busy'), screen.getByTestId('busy-sweep')]) expect(mark.getAttribute('aria-hidden')).toBe('true');
  });

  it('shows on a placeholder only while the calculation is in flight', () => {
    const attention = (loading: boolean) => ({loading, unavailable: undefined, result: undefined, row: undefined}) as unknown as RetrieverAttention;
    const {rerender} = render(<RetrieverRowDetails attention={attention(true)}/>);
    expect(screen.getByText('Calculating selected method from saved evidence…')).toBeTruthy();
    expect(screen.queryByTestId('busy')).not.toBeNull();
    rerender(<RetrieverRowDetails attention={attention(false)}/>);
    expect(screen.getByText('Calculating selected method from saved evidence…')).toBeTruthy();
    expect(screen.queryByTestId('busy')).toBeNull();
  });
});
