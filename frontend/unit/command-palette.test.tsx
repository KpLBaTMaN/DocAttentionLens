// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { CommandPalette, ShortcutSheet, matchCommands, type Command } from '../src/features/document/CommandPalette';
import { SHORTCUT_GROUPS } from '../src/features/document/useWorkspaceShortcuts';

afterEach(cleanup);

const run = () => {};
const commands: Command[] = [
  {id: 'method:raw', label: 'Raw attention', group: 'Method', keywords: 'method captured weights general', run},
  {id: 'method:rollout', label: 'Attention rollout', group: 'Method', keywords: 'method propagation general', run},
  {id: 'method:planned', label: 'Gradient attention', group: 'Method', keywords: 'method gradients', disabled: 'Planned', run},
  {id: 'page:1', label: 'Page 1', group: 'Page', run},
  {id: 'page:4', label: 'Page 4', group: 'Page', run},
  ...Array.from({length: 36}, (_, i) => ({id: `layer:${i}`, label: `Layer ${i}`, group: 'Layer', searchOnly: true, run})),
  {id: 'display:heatmap', label: 'Heatmap', group: 'Display', keywords: 'display mode colour', run},
  {id: 'details:scale', label: 'Details: Scale', group: 'Details', run},
];
const ids = (query: string) => matchCommands(commands, query).map(c => c.id);

describe('matchCommands', () => {
  it('lists everything except search-only entries when nothing is typed', () => {
    expect(ids('')).toEqual(commands.filter(c => !c.searchOnly).map(c => c.id));
  });

  it('finds a typed layer number exactly, before layers that only contain it', () => {
    expect(ids('layer 20')[0]).toBe('layer:20');
    expect(ids('layer 20')).not.toContain('layer:2');
    const two = ids('layer 2');
    expect(two[0]).toBe('layer:2');
    expect(two).toContain('layer:20');
    expect(two).toContain('layer:12');
  });

  it('requires every word, ranks a label that starts with the query first, and keeps given order on ties', () => {
    expect(ids('page 4')).toEqual(['page:4']);
    expect(ids('roll')[0]).toBe('method:rollout');
    // "Attention rollout" starts with the query; the other two only contain the word (the unavailable one last).
    expect(ids('attention').slice(0, 3)).toEqual(['method:rollout', 'method:raw', 'method:planned']);
    expect(ids('heat')).toEqual(['display:heatmap']);
    expect(ids('nothing like this')).toEqual([]);
  });

  it('matches keywords and groups as well as labels, and scattered letters as a last resort', () => {
    expect(ids('propagation')).toEqual(['method:rollout']);
    expect(ids('display mode')).toContain('display:heatmap');
    expect(ids('dtls scl')).toEqual(['details:scale']);
  });

  it('keeps unavailable commands listed, below an equal available one', () => {
    const tie: Command[] = [{id: 'a', label: 'Probe', group: 'Method', disabled: 'Planned', run}, {id: 'b', label: 'Probe', group: 'Method', run}];
    expect(matchCommands(tie, 'probe').map(c => c.id)).toEqual(['b', 'a']);
  });
});

describe('CommandPalette', () => {
  function setup(list = commands) {
    const onClose = vi.fn();
    const spies = list.map(c => ({...c, run: vi.fn()}));
    const trigger = document.createElement('button');
    document.body.appendChild(trigger); trigger.focus();
    const view = render(<CommandPalette open onClose={onClose} commands={spies}/>);
    return {onClose, spies, view, trigger};
  }

  it('is a modal dialog whose combobox points at the active option', () => {
    setup();
    const dialog = screen.getByRole('dialog', {name: 'Command palette'});
    expect(dialog.getAttribute('aria-modal')).toBe('true');
    const input = screen.getByRole('combobox', {name: 'Command'});
    expect(document.activeElement).toBe(input);
    const listbox = screen.getByRole('listbox', {name: 'Commands'});
    expect(input.getAttribute('aria-controls')).toBe(listbox.id);
    const active = document.getElementById(input.getAttribute('aria-activedescendant')!)!;
    expect(active.getAttribute('aria-selected')).toBe('true');
    expect(active.textContent).toContain('Raw attention');
  });

  it('moves with the arrow keys and runs the active command on Enter', () => {
    const {spies, onClose} = setup();
    const input = screen.getByRole('combobox', {name: 'Command'});
    fireEvent.change(input, {target: {value: 'layer 20'}});
    fireEvent.keyDown(input, {key: 'Enter'});
    expect(spies.find(c => c.id === 'layer:20')!.run).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalled();
  });

  it('starts a query on its first runnable match and never runs a disabled command', () => {
    const {spies} = setup();
    const input = screen.getByRole('combobox', {name: 'Command'});
    fireEvent.change(input, {target: {value: 'gradient'}});
    const option = screen.getByRole('option', {name: /Gradient attention/});
    expect(option.getAttribute('aria-disabled')).toBe('true');
    expect(option.textContent).toContain('Planned');
    fireEvent.keyDown(input, {key: 'Enter'});
    fireEvent.click(option);
    expect(spies.find(c => c.id === 'method:planned')!.run).not.toHaveBeenCalled();
    fireEvent.change(input, {target: {value: 'page'}});
    fireEvent.keyDown(input, {key: 'ArrowDown'});
    fireEvent.keyDown(input, {key: 'Enter'});
    expect(spies.find(c => c.id === 'page:4')!.run).toHaveBeenCalledTimes(1);
  });

  it('closes on Escape and returns focus to where it was', async () => {
    const {onClose, trigger} = setup();
    fireEvent.keyDown(screen.getByRole('combobox', {name: 'Command'}), {key: 'Escape'});
    expect(onClose).toHaveBeenCalledTimes(1);
    await new Promise(resolve => requestAnimationFrame(() => resolve(undefined)));
    expect(document.activeElement).toBe(trigger);
    trigger.remove();
  });

  it('says when nothing matches', () => {
    setup();
    fireEvent.change(screen.getByRole('combobox', {name: 'Command'}), {target: {value: 'zzzz qqqq'}});
    expect(screen.getByRole('status').textContent).toContain('No command matches');
  });
});

describe('ShortcutSheet', () => {
  it('lists every workspace shortcut and closes on Escape', () => {
    const onClose = vi.fn();
    render(<ShortcutSheet open onClose={onClose} groups={SHORTCUT_GROUPS}/>);
    const sheet = screen.getByRole('dialog', {name: 'Keyboard shortcuts'});
    const keys = [...sheet.querySelectorAll('kbd')].map(k => k.textContent);
    for (const key of [',', '.', '[', ']', '{', '}', 'A', 'PgUp', 'PgDn', 'V', 'F', 'D', 'M', 'Ctrl', 'S', 'K', '?']) expect(keys).toContain(key);
    expect(document.activeElement?.textContent).toBe('Close');
    fireEvent.keyDown(sheet, {key: 'Escape'});
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
