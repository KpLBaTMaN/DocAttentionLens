// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, renderHook } from '@testing-library/react';
import { isTypingTarget, shortcutAction, useWorkspaceShortcuts, type ShortcutAction, type ShortcutHandlers } from '../src/features/document/useWorkspaceShortcuts';

afterEach(() => {cleanup(); document.body.innerHTML = '';});

describe('shortcutAction', () => {
  it('maps every workspace key', () => {
    const cases: [string, ShortcutAction][] = [[',', 'sourcePrev'], ['.', 'sourceNext'], ['[', 'layerPrev'], [']', 'layerNext'], ['{', 'headPrev'], ['}', 'headNext'],
      ['a', 'allHeads'], ['A', 'allHeads'], ['PageUp', 'pagePrev'], ['PageDown', 'pageNext'], ['v', 'display'], ['V', 'display'], ['f', 'focus'], ['p', 'prompt'], ['d', 'details'], ['m', 'picker'], ['?', 'help']];
    for (const [key, action] of cases) expect(shortcutAction({key, shiftKey: key !== key.toLowerCase() || '{}?'.includes(key)})).toBe(action);
    expect(shortcutAction({key: 's', ctrlKey: true})).toBe('save');
    expect(shortcutAction({key: 'S', metaKey: true})).toBe('save');
    expect(shortcutAction({key: 'k', ctrlKey: true})).toBe('palette');
    expect(shortcutAction({key: 'k', metaKey: true})).toBe('palette');
    expect(shortcutAction({key: 'x'})).toBeUndefined();
  });

  it('ignores plain keys while typing, but keeps Ctrl/Cmd + S and K', () => {
    const input = document.createElement('input');
    const textarea = document.createElement('textarea');
    for (const target of [input, textarea]) {
      expect(shortcutAction({key: 'd', target})).toBeUndefined();
      expect(shortcutAction({key: ']', target})).toBeUndefined();
      expect(shortcutAction({key: 's', ctrlKey: true, target})).toBe('save');
      expect(shortcutAction({key: 'k', ctrlKey: true, target})).toBe('palette');
    }
  });

  it('recognises typing targets', () => {
    for (const html of ['<input>', '<textarea></textarea>', '<select><option>1</option></select>', '<div contenteditable="true"><span>x</span></div>']) {
      const host = document.createElement('div');
      host.innerHTML = html;
      document.body.append(host);
      expect(isTypingTarget(host.querySelector('span') ?? host.firstElementChild)).toBe(true);
    }
    expect(isTypingTarget(document.createElement('button'))).toBe(false);
    expect(isTypingTarget(null)).toBe(false);
  });

  it('does nothing while something is open, with Alt held, while composing or after another handler', () => {
    expect(shortcutAction({key: 'd'}, true)).toBeUndefined();
    expect(shortcutAction({key: 's', ctrlKey: true}, true)).toBeUndefined();
    expect(shortcutAction({key: 'k', ctrlKey: true}, true)).toBeUndefined();
    expect(shortcutAction({key: 'd', altKey: true})).toBeUndefined();
    expect(shortcutAction({key: 'd', isComposing: true})).toBeUndefined();
    expect(shortcutAction({key: 'd', defaultPrevented: true})).toBeUndefined();
    expect(shortcutAction({key: 's', ctrlKey: true, shiftKey: true})).toBeUndefined();
    expect(shortcutAction({key: 'd', ctrlKey: true})).toBeUndefined();
  });

  it('acts once per press for toggles, but repeats steps', () => {
    for (const key of ['d', 'f', 'm', 'v', 'a', '?']) expect(shortcutAction({key, repeat: true})).toBeUndefined();
    for (const key of [']', '}', '.', 'PageDown']) expect(shortcutAction({key, repeat: true})).toBeDefined();
  });
});

describe('useWorkspaceShortcuts', () => {
  const press = (init: KeyboardEventInit, target: EventTarget = document.body) => {
    const event = new KeyboardEvent('keydown', {bubbles: true, cancelable: true, ...init});
    target.dispatchEvent(event);
    return event;
  };
  function setup(handlers: ShortcutHandlers, options?: {blocked?: boolean}) {
    return renderHook(({h, o}) => useWorkspaceShortcuts(h, o), {initialProps: {h: handlers, o: options}});
  }

  it('runs the handler, prevents the default and leaves unhandled keys alone', () => {
    const details = vi.fn(), layerNext = vi.fn();
    setup({details, layerNext});
    expect(press({key: 'd'}).defaultPrevented).toBe(true);
    expect(press({key: ']'}).defaultPrevented).toBe(true);
    expect(details).toHaveBeenCalledTimes(1);
    expect(layerNext).toHaveBeenCalledTimes(1);
    // No handler (e.g. a method without heads): the key stays with the browser.
    expect(press({key: '}'}).defaultPrevented).toBe(false);
  });

  it('uses the latest handlers without re-subscribing', () => {
    const first = vi.fn(), second = vi.fn();
    const hook = setup({details: first});
    hook.rerender({h: {details: second}, o: undefined});
    press({key: 'd'});
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
  });

  it('ignores keys typed into fields and keys while blocked or a dialog is open', () => {
    const details = vi.fn(), save = vi.fn();
    const hook = setup({details, save});
    const input = document.createElement('input');
    document.body.appendChild(input);
    press({key: 'd'}, input);
    expect(details).not.toHaveBeenCalled();
    press({key: 's', ctrlKey: true}, input);
    expect(save).toHaveBeenCalledTimes(1);
    hook.rerender({h: {details, save}, o: {blocked: true}});
    press({key: 'd'});
    expect(details).not.toHaveBeenCalled();
    // Ctrl+S is still kept from the browser's "Save page", without saving behind an open dialog.
    expect(press({key: 's', ctrlKey: true}).defaultPrevented).toBe(true);
    expect(save).toHaveBeenCalledTimes(1);
    hook.rerender({h: {details, save}, o: undefined});
    const dialog = document.createElement('div');
    dialog.setAttribute('role', 'dialog'); dialog.setAttribute('aria-modal', 'true');
    document.body.appendChild(dialog);
    press({key: 'd'});
    expect(details).not.toHaveBeenCalled();
    dialog.remove();
    const menu = document.createElement('div');
    menu.setAttribute('role', 'menu');
    const item = document.createElement('button');
    menu.appendChild(item); document.body.appendChild(menu);
    press({key: 'd'}, item);
    expect(details).not.toHaveBeenCalled();
    press({key: 'd'});
    expect(details).toHaveBeenCalledTimes(1);
  });
});
