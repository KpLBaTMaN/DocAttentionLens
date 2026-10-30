import { useEffect, useRef } from 'react';

/** Everything the workspace keyboard can do. */
export type ShortcutAction = 'save' | 'palette' | 'help' | 'picker' | 'details' | 'focus' | 'display' | 'prompt'
  | 'pagePrev' | 'pageNext' | 'sourcePrev' | 'sourceNext' | 'layerPrev' | 'layerNext' | 'headPrev' | 'headNext' | 'allHeads';

/** The shortcut sheet (`?`): keys as shown, grouped, in reading order. */
export const SHORTCUT_GROUPS: {label: string; items: {keys: string[]; label: string}[]}[] = [
  {label: 'Source, layer and head', items: [
    {keys: [',', '.'], label: 'Previous or next source token'},
    {keys: ['[', ']'], label: 'Previous or next layer'},
    {keys: ['{', '}'], label: 'Previous or next head'},
    {keys: ['A'], label: 'All heads (methods with a mean over heads)'},
    {keys: ['PgUp', 'PgDn'], label: 'Previous or next page'},
  ]},
  {label: 'View', items: [
    {keys: ['V'], label: 'Next display mode: Original image, Heatmap, Overlay, Blend'},
    {keys: ['F'], label: 'Image focus'},
    {keys: ['P'], label: 'Show or hide the full prompt'},
    {keys: ['D'], label: 'Open or close Details'},
    {keys: ['M'], label: 'Browse methods'},
  ]},
  {label: 'Investigation', items: [
    {keys: ['Ctrl', 'S'], label: 'Save'},
    {keys: ['Ctrl', 'K'], label: 'Command palette (Cmd K on a Mac)'},
    {keys: ['?'], label: 'This list of shortcuts'},
    {keys: ['Esc'], label: 'Close a menu, sheet or dialog'},
  ]},
];

const PLAIN: Record<string, ShortcutAction> = {
  ',': 'sourcePrev', '.': 'sourceNext', '[': 'layerPrev', ']': 'layerNext', '{': 'headPrev', '}': 'headNext',
  a: 'allHeads', pageup: 'pagePrev', pagedown: 'pageNext', v: 'display', f: 'focus', p: 'prompt', d: 'details', m: 'picker', '?': 'help',
};
/** Toggles and openers act once per press, not on key repeat. */
const ONCE = new Set<ShortcutAction>(['allHeads', 'display', 'focus', 'prompt', 'details', 'picker', 'help']);

export type KeyLike = {
  key: string; ctrlKey?: boolean; metaKey?: boolean; altKey?: boolean; shiftKey?: boolean;
  repeat?: boolean; isComposing?: boolean; defaultPrevented?: boolean; target?: EventTarget | null;
};

/** True when a key event comes from a place where the user is typing. */
export function isTypingTarget(target: EventTarget | null): boolean {
  const el = target as (Element & {isContentEditable?: boolean}) | null;
  if (!el || typeof el.closest !== 'function') return false;
  return !!el.isContentEditable || !!el.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"])');
}

/** The workspace action for a key press, or undefined when the key belongs to a field, dialog or menu. */
export function shortcutAction(e: KeyLike, blocked = false): ShortcutAction | undefined {
  if (e.defaultPrevented || e.isComposing || e.altKey) return undefined;
  const key = e.key.toLowerCase();
  if (e.ctrlKey || e.metaKey) {
    if (e.shiftKey || blocked) return undefined;
    return key === 's' ? 'save' : key === 'k' ? 'palette' : undefined;
  }
  if (blocked || isTypingTarget(e.target ?? null)) return undefined;
  const action = PLAIN[key];
  if (!action || (e.repeat && ONCE.has(action))) return undefined;
  return action;
}

/** True while something modal or menu-like is open, or focus is inside one: plain keys belong to it then. */
export function dialogOpen(target: EventTarget | null): boolean {
  if (typeof document === 'undefined') return false;
  if (document.querySelector('dialog[open], [role="dialog"][aria-modal="true"]')) return true;
  const el = target as Element | null;
  return !!el && typeof el.closest === 'function' && !!el.closest('[role="menu"], [role="dialog"], [role="listbox"]');
}

export type ShortcutHandlers = Partial<Record<ShortcutAction, () => void>>;

/** One keyboard handler for the workspace; keys without a handler are left to the browser. */
export function useWorkspaceShortcuts(handlers: ShortcutHandlers, {blocked = false, enabled = true}: {blocked?: boolean; enabled?: boolean} = {}) {
  const latest = useRef({handlers, blocked, enabled});
  latest.current = {handlers, blocked, enabled};
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const {handlers: h, blocked: b, enabled: on} = latest.current;
      if (!on) return;
      const action = shortcutAction(e, b || dialogOpen(e.target));
      if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 's') e.preventDefault();
      const run = action && h[action];
      if (!run) return;
      e.preventDefault();
      run();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}
