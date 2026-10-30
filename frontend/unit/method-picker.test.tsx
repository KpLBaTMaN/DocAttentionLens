// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { useRef, useState } from 'react';
import type { Method } from '../src/api/client';
import { MethodPicker, MethodTrigger, popoverPosition, type MethodPickerProps } from '../src/features/document/MethodPicker';
import type { MethodChoice } from '../src/features/document/methodChoices';
import { MethodBar } from '../src/features/document/MethodBar';

afterEach(cleanup);

function descriptor(method_id: string, family: string, name: string, {perspective = 'general', evidence_kind = 'attention_allocation', status = 'implemented',
  parameters = ['step', 'layer', 'head'], variant_id = 'v1', units = 'dimensionless mass', requires = [] as string[]}: {perspective?: Method['perspective']; evidence_kind?: string; requires?: string[];
  status?: Method['status']; parameters?: string[]; variant_id?: string; units?: string} = {}): Method {
  return {method_id, variant_id, family, perspective, name, summary: `${name} summary.`, evidence_kind, requires, status, quantity: 'attention_weights',
    parameters_schema: {properties: Object.fromEntries(parameters.map(p => [p, {type: 'integer'}]))},
    output: {schema: 'method_analysis', units, signed: false, support: 'valid causal targets', normalization: 'no renormalization'}} as unknown as Method;
}
const choice = (method: Method, reasons: string[] = []): MethodChoice => ({method, available: !reasons.length && method.status !== 'planned', reasons});

const raw = descriptor('attention.row', 'direct_attention', 'Raw Attention', {variant_id: 'captured_weights_v1'});
const headByHead = descriptor('attention.head_by_head', 'direct_attention', 'Head-by-Head');
const mean = descriptor('attention.mean_heads', 'aggregation', 'Mean-Head Attention', {parameters: ['step', 'layer', 'heads'], units: 'mean mass'});
const maxHeads = descriptor('attention.max_heads', 'aggregation', 'Maximum-Head Attention', {parameters: ['step', 'layer', 'heads']});
const rollout = descriptor('attention.rollout', 'connectivity', 'Attention Rollout', {evidence_kind: 'attention_summary', requires: ['full_square_attention_matrices'], parameters: ['step', 'layer', 'residual_coefficient']});
const entropy = descriptor('attention.entropy', 'attention_statistics', 'Attention Entropy', {evidence_kind: 'attention_summary'});
const elements = descriptor('attention.element_allocation', 'aggregation', 'Element Attention Mass', {perspective: 'document_aware', variant_id: 'ownership_mass_v1'});
const order = descriptor('attention.reading_order', 'document_structure', 'Reading Order Profile', {perspective: 'document_aware', status: 'planned'});
const ROLLOUT_GAP = "Unavailable for this capture: full attention matrices: run a new generator capture with 'Save full attention matrices'.";
const choices = [choice(raw), choice(headByHead), choice(mean), choice(maxHeads), choice(rollout, [ROLLOUT_GAP]), choice(entropy), choice(elements), choice(order)];

function Harness(props: Partial<MethodPickerProps> & {initiallyOpen?: boolean}) {
  const anchorRef = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(props.initiallyOpen ?? true);
  return <div className="docattentionlens-app">
    <MethodTrigger ref={anchorRef} label="Raw Attention" open={open} onClick={() => setOpen(o => !o)}/>
    <button type="button">Elsewhere</button>
    <MethodPicker choices={choices} currentKey="attention.row/captured_weights_v1" perspective="general" onSelect={() => {}}
      {...props} open={open} anchorRef={anchorRef} onClose={() => {props.onClose?.(); setOpen(false);}}/>
  </div>;
}

const dialog = () => screen.getByRole('dialog', {name: 'Choose analysis method'});
const search = () => screen.getByRole('combobox');
const optionNames = () => screen.queryAllByRole('option').map(o => o.querySelector('.docattentionlens-mp-option-name')?.textContent);
const activeOption = () => document.getElementById(search().getAttribute('aria-activedescendant') ?? '');
const key = (k: string, target: Element = search(), init: object = {}) => fireEvent.keyDown(target, {key: k, ...init});

describe('method picker', () => {
  it('opens as a labelled dialog with the search focused and collapsed families with counts', () => {
    render(<Harness/>);
    expect(dialog()).toBeTruthy();
    expect(document.activeElement).toBe(search());
    const family = screen.getByRole('button', {name: /Attention · Aggregation/});
    expect(family.getAttribute('aria-expanded')).toBe('false');
    expect(within(family).getByLabelText('2 methods')).toBeTruthy();
    // The current method's family opens so the selection is visible.
    expect(screen.getByRole('button', {name: /Attention · Direct inspection/}).getAttribute('aria-expanded')).toBe('true');
    const current = screen.getAllByRole('option', {selected: true});
    expect(current.every(o => o.textContent === 'Raw Attention')).toBe(true);
    fireEvent.click(family);
    expect(family.getAttribute('aria-expanded')).toBe('true');
    expect(within(screen.getByRole('group', {name: /Attention · Aggregation/})).getAllByRole('option')).toHaveLength(2);
  });

  it('positions the trigger as a dialog opener', () => {
    render(<Harness initiallyOpen={false}/>);
    const trigger = screen.getByRole('button', {name: /^Raw Attention\s*, choose analysis method$/});
    expect(trigger.getAttribute('aria-haspopup')).toBe('dialog');
    expect(trigger.getAttribute('aria-expanded')).toBe('false');
    expect(screen.queryByRole('dialog')).toBeNull();
    fireEvent.click(trigger);
    expect(trigger.getAttribute('aria-expanded')).toBe('true');
    expect(dialog()).toBeTruthy();
  });

  it('filters by search, expanding every matching family', () => {
    render(<Harness/>);
    fireEvent.change(search(), {target: {value: 'heads'}});
    expect(optionNames()).toEqual(['Mean-Head Attention', 'Maximum-Head Attention']);
    fireEvent.change(search(), {target: {value: 'statistics'}});
    expect(optionNames()).toEqual(['Attention Entropy']);
    fireEvent.change(search(), {target: {value: 'element'}});
    expect(screen.getByRole('status').textContent).toBe('No General methods match “element”.');
  });

  it('shows unavailable and planned methods disabled, with their reasons for screen readers', () => {
    const onSelect = vi.fn();
    render(<Harness onSelect={onSelect}/>);
    fireEvent.change(search(), {target: {value: 'rollout'}});
    const option = screen.getByRole('option', {name: /Attention Rollout/});
    expect(option.getAttribute('aria-disabled')).toBe('true');
    expect(option.textContent).toContain('Needs a full-matrix capture');
    expect(option.querySelector('.docattentionlens-mp-sr')?.textContent).toContain(ROLLOUT_GAP);
    fireEvent.click(option);
    expect(onSelect).not.toHaveBeenCalled();
    // The preview states readiness and the full reason.
    const readiness = screen.getByTestId('method-readiness');
    expect(readiness.textContent).toContain('Unavailable');
    expect(readiness.textContent).toContain(ROLLOUT_GAP);
    expect((screen.getByRole('button', {name: 'Use Attention Rollout'}) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole('radio', {name: 'Document-aware'}));
    fireEvent.change(search(), {target: {value: 'reading'}});
    expect(screen.getByRole('option', {name: /Reading Order Profile/}).getAttribute('aria-disabled')).toBe('true');
    expect(screen.getByRole('option', {name: /Reading Order Profile/}).textContent).toContain('Planned');
  });

  it('moves the highlight with the arrows, previews it and selects with Enter', () => {
    const onSelect = vi.fn(), onClose = vi.fn();
    render(<Harness onSelect={onSelect} onClose={onClose} catalogueHref={m => `/toolbox/${m.method_id}?returnTo=%2F`}
      renderSettings={m => <p>Settings for {m.name}</p>}/>);
    expect(activeOption()?.textContent).toBe('Raw Attention');
    fireEvent.click(screen.getByRole('button', {name: /Attention · Aggregation/}));
    key('ArrowDown');
    expect(activeOption()?.textContent).toBe('Mean-Head Attention');
    const preview = screen.getByTestId('method-preview');
    expect(preview.textContent).toContain('Mean-Head Attention summary.');
    expect(preview.textContent).toContain('mean mass');
    expect(preview.textContent).toContain('Layer · Head selection');
    expect(preview.textContent).toContain('Settings for Mean-Head Attention');
    expect(within(preview).getByRole('link', {name: /Full description in Methods/}).getAttribute('href')).toBe('/toolbox/attention.mean_heads?returnTo=%2F');
    key('Enter');
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect.mock.calls[0][0]).toBe(mean);
    expect(onSelect.mock.calls[0][1].asGroupingInput).toBe(false);
    expect(onClose).toHaveBeenCalled();
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(document.activeElement).toBe(screen.getByRole('button', {name: /choose analysis method/}));
  });

  it('walks family headers with the arrows and opens one with Enter', () => {
    render(<Harness/>);
    key('ArrowUp');
    expect(activeOption()?.textContent).toMatch(/Attention · Direct inspection/);
    key('ArrowDown'); key('ArrowDown'); key('ArrowDown');
    expect(activeOption()?.textContent).toMatch(/Attention · Aggregation/);
    key('Enter');
    key('ArrowDown');
    expect(activeOption()?.textContent).toBe('Mean-Head Attention');
    key('ArrowUp'); key('ArrowUp');
    expect(activeOption()?.textContent).toBe('Head-by-Head');
  });

  it('closes on Escape and returns focus to the trigger', () => {
    const onClose = vi.fn();
    render(<Harness onClose={onClose}/>);
    key('Escape');
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(document.activeElement).toBe(screen.getByRole('button', {name: /choose analysis method/}));
  });

  it('closes on an outside click without taking focus', () => {
    const onClose = vi.fn();
    render(<Harness onClose={onClose}/>);
    fireEvent.pointerDown(screen.getByRole('button', {name: 'Elsewhere'}));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('focuses the search on "/" and keeps Tab inside the dialog', () => {
    render(<Harness catalogueHref={() => '/toolbox/x'}/>);
    const radio = screen.getByRole('radio', {name: 'General'});
    radio.focus();
    key('/', radio);
    expect(document.activeElement).toBe(search());
    const link = screen.getByRole('link', {name: /Full description in Methods/});
    link.focus();
    key('Tab', link);
    expect(document.activeElement).toBe(search());
    key('Tab', search(), {shiftKey: true});
    expect(document.activeElement).toBe(link);
  });

  it('filters by perspective and reports the change', () => {
    const onPerspective = vi.fn();
    render(<Harness onPerspective={onPerspective}/>);
    fireEvent.click(screen.getByRole('radio', {name: 'Document-aware'}));
    expect(onPerspective).toHaveBeenCalledWith('document_aware');
    expect(screen.getByRole('radio', {name: 'Document-aware'}).getAttribute('aria-checked')).toBe('true');
    expect(within(screen.getByRole('button', {name: /Attention · Aggregation/})).getByLabelText('1 methods')).toBeTruthy();
    expect(screen.queryByRole('button', {name: /Direct inspection/})).toBeNull();
    expect(screen.getByRole('listbox').getAttribute('aria-label')).toBe('Document-aware analysis methods');
    key('ArrowLeft', screen.getByRole('radio', {name: 'Document-aware'}));
    expect(onPerspective).toHaveBeenLastCalledWith('general');
  });

  it('offers General inputs grouped into elements in the Document-aware view', () => {
    const onSelect = vi.fn();
    render(<Harness onSelect={onSelect} perspective="document_aware" groupingInputs currentKey="attention.element_allocation/ownership_mass_v1"
      currentInputKey="attention.row/captured_weights_v1"/>);
    const grouped = screen.getByRole('group', {name: /grouped into elements/});
    expect(within(grouped).getAllByRole('option').map(o => o.querySelector('.docattentionlens-mp-option-name')?.textContent))
      .toEqual(['Raw Attention', 'Head-by-Head', 'Mean-Head Attention', 'Maximum-Head Attention', 'Attention Rollout']);
    fireEvent.click(within(grouped).getByRole('option', {name: 'Mean-Head Attention'}));
    expect(onSelect.mock.calls[0][0]).toBe(mean);
    expect(onSelect.mock.calls[0][1].asGroupingInput).toBe(true);
  });

  it('shows loading and errors in place of the list', () => {
    const {rerender} = render(<Harness choices={[]} loading/>);
    expect(screen.getByRole('status').textContent).toBe('Loading method catalogue…');
    rerender(<Harness choices={[]} error="Method catalogue unavailable"/>);
    expect(screen.getByRole('alert').textContent).toBe('Method catalogue unavailable');
  });

  it('becomes a bottom sheet below 900 px', () => {
    render(<Harness/>);
    expect(dialog().className).not.toContain('is-sheet');
    const width = window.innerWidth;
    act(() => {(window as unknown as {innerWidth: number}).innerWidth = 375; window.dispatchEvent(new Event('resize'));});
    try {
      expect(dialog().className).toContain('is-sheet');
      expect(dialog().getAttribute('aria-modal')).toBe('true');
    } finally {
      act(() => {(window as unknown as {innerWidth: number}).innerWidth = width; window.dispatchEvent(new Event('resize'));});
    }
  });
});

describe('docked method picker (Details › Method)', () => {
  it('is an in-flow region that stays open after a choice and names the current rows', () => {
    const onSelect = vi.fn(), onClose = vi.fn();
    render(<div className="docattentionlens-app"><MethodPicker open docked choices={choices} currentKey="attention.element_allocation/ownership_mass_v1"
      currentInputKey="attention.mean_heads/v1" perspective="document_aware" groupingInputs onSelect={onSelect} onClose={onClose}/></div>);
    expect(screen.queryByRole('dialog')).toBeNull();
    const region = screen.getByRole('region', {name: 'Choose analysis method'});
    expect(region.getAttribute('data-testid')).toBe('method-browser');
    // A docked picker does not take focus on its own; the workspace focuses it when asked.
    expect(document.activeElement).not.toBe(search());
    const input = region.querySelector('[data-grouping-input][data-method-key="attention.mean_heads/v1"]');
    expect(input?.getAttribute('data-badge')).toBe('Current input');
    expect(region.querySelector('[data-method-key="attention.element_allocation/ownership_mass_v1"]:not([data-grouping-input])')?.getAttribute('data-badge')).toBe('Current');
    fireEvent.click(region.querySelector('[data-grouping-input][data-method-key="attention.max_heads/v1"]')!);
    expect(onSelect).toHaveBeenCalledWith(maxHeads, expect.objectContaining({asGroupingInput: true}));
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByRole('region', {name: 'Choose analysis method'})).toBeTruthy();
  });

  it('clears the search on Escape without closing, and ignores outside clicks', () => {
    const onClose = vi.fn();
    render(<div className="docattentionlens-app"><button type="button">Elsewhere</button>
      <MethodPicker open docked choices={choices} currentKey="attention.row/captured_weights_v1" perspective="general" onSelect={() => {}} onClose={onClose}/></div>);
    fireEvent.change(search(), {target: {value: 'heads'}});
    key('Escape');
    expect((search() as HTMLInputElement).value).toBe('');
    key('Escape');
    fireEvent.pointerDown(screen.getByRole('button', {name: 'Elsewhere'}));
    expect(onClose).not.toHaveBeenCalled();
  });
});

describe('method bar', () => {
  it('names each step, opens its choice, steps through methods and carries the shown method', () => {
    const open = vi.fn(), grouping = vi.fn(), next = vi.fn(), browse = vi.fn();
    render(<MethodBar testId="method-choice" data={{'method-key': 'attention.element_allocation/ownership_mass_v1', 'input-key': 'attention.mean_heads/v1', perspective: undefined}}
      steps={[{key: 'method', kicker: 'Attention', label: 'Mean across heads', title: 'Choose', onClick: open},
        {key: 'grouping', kicker: 'Grouping', label: 'Element totals', title: 'Group', onClick: grouping}]}
      onNext={next} nextTitle="Next method: Max-Head" onBrowse={browse} browsing/>);
    const bar = screen.getByRole('group', {name: 'Method'});
    expect(bar.getAttribute('data-method-key')).toBe('attention.element_allocation/ownership_mass_v1');
    expect(bar.getAttribute('data-input-key')).toBe('attention.mean_heads/v1');
    expect(bar.hasAttribute('data-perspective')).toBe(false);
    fireEvent.click(screen.getByRole('button', {name: 'Attention: Mean across heads'}));
    fireEvent.click(screen.getByRole('button', {name: 'Grouping: Element totals'}));
    expect(open).toHaveBeenCalledOnce();
    expect(grouping).toHaveBeenCalledOnce();
    expect((screen.getByRole('button', {name: 'Previous method'}) as HTMLButtonElement).disabled).toBe(true);
    const nextButton = screen.getByRole('button', {name: 'Next method'});
    expect(nextButton.getAttribute('title')).toBe('Next method: Max-Head');
    fireEvent.click(nextButton);
    expect(next).toHaveBeenCalledOnce();
    const browseButton = screen.getByRole('button', {name: /Browse methods/});
    expect(browseButton.getAttribute('aria-pressed')).toBe('true');
    fireEvent.click(browseButton);
    expect(browse).toHaveBeenCalledOnce();
  });

  it('offers no browsing or stepping without a method choice', () => {
    render(<MethodBar steps={[{key: 'signal', kicker: 'Signal', label: 'MaxSim similarity (not attention)', title: 'Result', onClick: () => {}}]}/>);
    expect(screen.queryByRole('button', {name: /Browse methods/})).toBeNull();
    expect(screen.queryByRole('button', {name: 'Next method'})).toBeNull();
  });

  it('opens the method panel under the bar, keeps it for presses in the scrubber, closes it on a press elsewhere, Escape or its close button, and chooses the grouping from its menu', () => {
    const totals = vi.fn(), none = vi.fn();
    function Harness() {
      const [open, setOpen] = useState(false);
      const scrubber = useRef<HTMLDivElement>(null);
      return <><button type="button">Elsewhere</button><div ref={scrubber}><button type="button">Layer 3</button></div>
        <MethodBar steps={[{key: 'method', kicker: 'General method', label: 'Raw attention', title: 'The method', onClick: () => setOpen(v => !v), expanded: open},
          {key: 'grouping', kicker: 'Grouping', label: 'None', title: 'Group', onClick: () => {}, menu: {label: 'Grouping', items: [
            {label: 'None', checked: true, onSelect: none},
            {label: 'Element totals', checked: false, title: 'Add up per element', onSelect: totals}]}}]}
          onBrowse={() => setOpen(true)} browsing={open} end={<button type="button">Details</button>}
          panel={{open, onClose: () => setOpen(false), label: 'Method panel', content: <p>Settings and list</p>, keepOpen: [scrubber]}}/></>;
    }
    render(<Harness/>);
    const step = screen.getByRole('button', {name: 'General method: Raw attention'});
    expect(step.getAttribute('aria-expanded')).toBe('false');
    expect(screen.queryByRole('dialog', {name: 'Method panel'})).toBeNull();
    fireEvent.click(step);
    expect(step.getAttribute('aria-expanded')).toBe('true');
    expect(step.getAttribute('aria-controls')).toBe('method-panel');
    expect(screen.getByRole('dialog', {name: 'Method panel'}).textContent).toContain('Settings and list');
    // Presses in the bar (the Details toggle) and in the scrubber keep it, so settings can be tuned while the result updates.
    fireEvent.pointerDown(screen.getByRole('button', {name: 'Details'}));
    fireEvent.pointerDown(screen.getByRole('button', {name: 'Layer 3'}));
    expect(screen.queryByRole('dialog', {name: 'Method panel'})).not.toBeNull();
    fireEvent.click(screen.getByRole('button', {name: 'Close the method panel'}));
    expect(screen.queryByRole('dialog', {name: 'Method panel'})).toBeNull();
    fireEvent.click(step);
    expect(screen.queryByRole('dialog', {name: 'Method panel'})).not.toBeNull();
    fireEvent.pointerDown(screen.getByRole('button', {name: 'Elsewhere'}));
    expect(screen.queryByRole('dialog', {name: 'Method panel'})).toBeNull();
    fireEvent.click(screen.getByRole('button', {name: /Browse methods/}));
    expect(screen.queryByRole('dialog', {name: 'Method panel'})).not.toBeNull();
    fireEvent.keyDown(document, {key: 'Escape'});
    expect(screen.queryByRole('dialog', {name: 'Method panel'})).toBeNull();
    const grouping = screen.getByRole('button', {name: 'Grouping: None'});
    expect(grouping.getAttribute('data-step')).toBe('grouping');
    fireEvent.click(grouping);
    const menu = screen.getByRole('menu', {name: 'Grouping'});
    expect(within(menu).getByRole('menuitemradio', {name: 'None'}).getAttribute('aria-checked')).toBe('true');
    const element = within(menu).getByRole('menuitemradio', {name: 'Element totals'});
    expect(element.getAttribute('title')).toBe('Add up per element');
    fireEvent.click(element);
    expect(totals).toHaveBeenCalledOnce();
    expect(none).not.toHaveBeenCalled();
    expect(screen.queryByRole('menu', {name: 'Grouping'})).toBeNull();
  });
});

describe('popover position', () => {
  const viewport = {width: 1440, height: 900};
  it('opens below the anchor and stays inside the viewport', () => {
    expect(popoverPosition({top: 100, bottom: 124, left: 1300}, viewport)).toEqual({top: 130, left: 1440 - 760 - 8, width: 760, height: 540});
  });
  it('flips above when there is more room there', () => {
    expect(popoverPosition({top: 800, bottom: 824, left: 20}, viewport)).toEqual({top: 800 - 6 - 540, left: 20, width: 760, height: 540});
  });
  it('clamps inside a short viewport', () => {
    const p = popoverPosition({top: 150, bottom: 170, left: 0}, {width: 700, height: 400});
    expect(p).toEqual({top: 8, left: 8, width: 684, height: 384});
  });
});
