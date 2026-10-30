// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { PageKey } from '../src/features/document/ScaleKey';

afterEach(cleanup);
const amber = {kind: 'amber' as const, opacity: .65};

describe('PageKey', () => {
  it('states the range and invites a hover before anything is under the pointer', () => {
    render(<PageKey ramp={amber} floor={0} maximum={.008} unit="Patch value"/>);
    expect(screen.getByTestId('page-key-floor').textContent).toBe('0');
    expect(screen.getByTestId('page-key-max').textContent).toBe('0.008000');
    expect(screen.getByTestId('page-key-readout').textContent).toBe('Hover the page');
    expect(screen.queryByTestId('page-key-marker')).toBeNull();
  });

  it('places the marker on the stretched scale and reads the value as a share of the maximum', () => {
    render(<PageKey ramp={{kind: 'blend', opacity: .45, colormap: 'viridis'}} floor={.002} maximum={.01} unit="Patch value" hover={{label: 'Patch 7', value: .006}}/>);
    expect(screen.getByTestId('page-key-floor').textContent).toBe('0.002000');
    expect(screen.getByTestId('page-key-marker').style.left).toBe('50%');
    const readout = screen.getByTestId('page-key-readout');
    expect(readout.textContent).toBe('60% · 0.006000');
    expect(readout.title).toBe('Patch 7: 0.006000, 60% of the maximum');
  });

  it('pins a value above a fixed scale to the end and says so', () => {
    render(<PageKey ramp={amber} floor={0} maximum={.004} unit="Patch value" hover={{label: 'Patch 1', value: .01}}/>);
    expect(screen.getByTestId('page-key-marker').style.left).toBe('100%');
    expect(screen.getByTestId('page-key-readout').textContent).toBe('>100% · 0.010000');
    expect(screen.getByTestId('page-key-readout').title).toContain('above the scale');
  });
});
