import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Search } from '../../app/icons';
import '../../styles/command-palette.css';

/** One palette entry. `disabled` is the reason it cannot run now; it stays listed so the reason is readable. */
export type Command = {
  id: string; label: string; group: string;
  /** Extra words that find it (family, synonyms). */
  keywords?: string;
  /** Short right-hand note: the current state or a key. */
  hint?: string;
  disabled?: string;
  /** Listed only once something is typed (for example "Layer 20" among 36 layers). */
  searchOnly?: boolean;
  run: () => void;
};

const words = (text: string) => text.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(Boolean);

/** The token's letters appear in this word in order ("dtls" in "details"). */
function scattered(token: string, word: string) {
  let at = 0;
  for (const ch of word) if (ch === token[at]) at++;
  return at === token.length;
}

/** How well one typed token matches a word list: whole word 3, word start 2, inside a word 1, letters in order within one word 0.5. */
function tokenScore(token: string, haystack: string[], joined: string): number {
  if (haystack.includes(token)) return 3;
  if (haystack.some(w => w.startsWith(token))) return 2;
  if (joined.includes(token)) return 1;
  return haystack.some(w => w[0] === token[0] && scattered(token, w)) ? .5 : 0;
}

/**
 * Commands matching a query, best first. Every typed token must match the label or keywords; ties keep
 * the given order. An empty query lists every command that is not `searchOnly`, in order.
 */
export function matchCommands(commands: Command[], query: string): Command[] {
  const tokens = words(query);
  if (!tokens.length) return commands.filter(c => !c.searchOnly);
  const label = query.trim().toLowerCase();
  const scored: {command: Command; score: number; order: number}[] = [];
  commands.forEach((command, order) => {
    const haystack = [...words(command.label), ...words(command.keywords ?? ''), ...words(command.group)];
    const joined = haystack.join(' ');
    let score = 0;
    for (const token of tokens) {
      const s = tokenScore(token, haystack, joined);
      if (!s) return;
      score += s;
    }
    const name = command.label.toLowerCase();
    if (name === label) score += 4; else if (name.startsWith(label)) score += 2;
    if (command.disabled) score -= .25;
    scored.push({command, score, order});
  });
  return scored.sort((a, b) => b.score - a.score || a.order - b.order).map(s => s.command);
}

/**
 * The command palette (Ctrl/Cmd + K): a modal dialog with a search combobox over a listbox of
 * commands. Arrow keys move, Enter runs, Escape closes; focus returns to where it was.
 */
export function CommandPalette({open, onClose, commands, placeholder = 'Type a method, page, “layer 20”, a display mode or an action'}: {
  open: boolean; onClose: () => void; commands: Command[]; placeholder?: string;
}) {
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLDivElement>(null);
  const returnTo = useRef<HTMLElement | null>(null);
  const id = useId();
  const matches = useMemo(() => matchCommands(commands, query), [commands, query]);
  // Opening remembers the focused control, so closing returns there.
  useLayoutEffect(() => {
    if (!open) return;
    returnTo.current = document.activeElement as HTMLElement | null;
    setQuery(''); setActive(0);
    input.current?.focus();
  }, [open]);
  // A new query starts on its best runnable match.
  useEffect(() => {const first = matches.findIndex(c => !c.disabled); setActive(first < 0 ? 0 : first);}, [query]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {list.current?.querySelector('[aria-selected="true"]')?.scrollIntoView({block: 'nearest'});}, [active, open]);
  if (!open) return null;
  const close = () => {onClose(); const back = returnTo.current; requestAnimationFrame(() => back?.isConnected && back.focus());};
  const run = (command?: Command) => {
    if (!command || command.disabled) return;
    onClose();
    // The command may move focus itself (opening Details, the picker); otherwise focus returns.
    const back = returnTo.current;
    command.run();
    requestAnimationFrame(() => {if (document.activeElement === document.body && back?.isConnected) back.focus();});
  };
  const optionId = (i: number) => `${id}-option-${i}`;
  const current = matches[active];
  return createPortal(<div className="docattentionlens-palette-backdrop" onPointerDown={e => {if (e.target === e.currentTarget) close();}}>
    <div className="docattentionlens-palette" role="dialog" aria-modal="true" aria-label="Command palette" data-testid="command-palette"
      onKeyDown={e => {
        if (e.key === 'Escape') {e.preventDefault(); e.stopPropagation(); close();}
        else if (e.key === 'Tab') e.preventDefault();
        else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {e.preventDefault(); close();}
      }}>
      <div className="docattentionlens-palette-search">
        <Search/>
        <input ref={input} role="combobox" aria-label="Command" aria-expanded="true" aria-controls={`${id}-list`} aria-autocomplete="list"
          aria-activedescendant={current ? optionId(active) : undefined} placeholder={placeholder} value={query} autoComplete="off" spellCheck={false}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => {
            const move = (by: number) => {e.preventDefault(); if (matches.length) setActive(a => (a + by + matches.length) % matches.length);};
            if (e.key === 'ArrowDown') move(1);
            else if (e.key === 'ArrowUp') move(-1);
            else if (e.key === 'Home' && e.ctrlKey) {e.preventDefault(); setActive(0);}
            else if (e.key === 'End' && e.ctrlKey) {e.preventDefault(); setActive(Math.max(0, matches.length - 1));}
            else if (e.key === 'Enter') {e.preventDefault(); run(current);}
          }}/>
      </div>
      <div ref={list} id={`${id}-list`} role="listbox" aria-label="Commands" className="docattentionlens-palette-list">
        {matches.map((command, i) => <div key={command.id} id={optionId(i)} role="option" aria-selected={i === active} aria-disabled={!!command.disabled || undefined}
          data-command-id={command.id} className="docattentionlens-palette-option" title={command.disabled}
          onPointerMove={() => i !== active && setActive(i)} onClick={() => run(command)}>
          <span className="docattentionlens-palette-label">{command.label}</span>
          {command.disabled ? <span className="docattentionlens-palette-reason">{command.disabled}</span> : command.hint && <span className="docattentionlens-palette-hint">{command.hint}</span>}
          <span className="docattentionlens-palette-group">{command.group}</span>
        </div>)}
        {!matches.length && <p className="docattentionlens-palette-empty" role="status">No command matches “{query}”.</p>}
      </div>
      <p className="docattentionlens-palette-foot" aria-hidden="true">↑ ↓ move · Enter choose · Esc close</p>
    </div>
  </div>, document.querySelector('.docattentionlens-app') ?? document.body);
}

/** The keyboard shortcut sheet (`?`, or ⋯ › Keyboard shortcuts): a modal list; Escape closes it and focus returns. */
export function ShortcutSheet({open, onClose, groups}: {open: boolean; onClose: () => void; groups: {label: string; items: {keys: string[]; label: string}[]}[]}) {
  const closeButton = useRef<HTMLButtonElement>(null);
  const returnTo = useRef<HTMLElement | null>(null);
  const titleId = useId();
  useLayoutEffect(() => {
    if (!open) return;
    returnTo.current = document.activeElement as HTMLElement | null;
    closeButton.current?.focus();
  }, [open]);
  if (!open) return null;
  const close = () => {onClose(); const back = returnTo.current; requestAnimationFrame(() => back?.isConnected && back.focus());};
  return createPortal(<div className="docattentionlens-palette-backdrop" onPointerDown={e => {if (e.target === e.currentTarget) close();}}>
    <div className="docattentionlens-palette docattentionlens-shortcuts" role="dialog" aria-modal="true" aria-labelledby={titleId} data-testid="shortcut-sheet"
      onKeyDown={e => {
        if (e.key === 'Escape' || e.key === '?') {e.preventDefault(); e.stopPropagation(); close();}
        else if (e.key === 'Tab') {e.preventDefault(); closeButton.current?.focus();}
      }}>
      <div className="docattentionlens-shortcuts-head"><h2 id={titleId}>Keyboard shortcuts</h2>
        <button ref={closeButton} type="button" className="docattentionlens-button" onClick={close}>Close</button></div>
      {groups.map(group => <section key={group.label} aria-label={group.label}><h3>{group.label}</h3>
        <dl>{group.items.map(item => <div key={item.label} style={{display: 'contents'}}>
          <dt>{item.keys.map(k => <kbd key={k}>{k}</kbd>)}</dt><dd>{item.label}</dd></div>)}</dl></section>)}
      <p className="docattentionlens-shortcuts-note">Shortcuts are ignored while you type in a field, and while a menu, sheet or dialog is open.</p>
    </div>
  </div>, document.querySelector('.docattentionlens-app') ?? document.body);
}
