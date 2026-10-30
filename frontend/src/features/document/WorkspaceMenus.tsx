import { useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode, type RefObject } from 'react';

/**
 * Closes a popover on Escape (returning focus to its trigger) and on a press outside it and its
 * trigger (focus stays where the press put it).
 */
export function useDismiss(open: boolean, close: (refocus: boolean) => void, refs: RefObject<HTMLElement | null>[]) {
  const latest = useRef({close, refs});
  latest.current = {close, refs};
  useEffect(() => {
    if (!open) return;
    const down = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (target && latest.current.refs.some(ref => ref.current?.contains(target))) return;
      latest.current.close(false);
    };
    const key = (event: globalThis.KeyboardEvent) => {
      if (event.key !== 'Escape' || event.defaultPrevented) return;
      event.preventDefault();
      latest.current.close(true);
    };
    document.addEventListener('pointerdown', down, true);
    document.addEventListener('keydown', key);
    return () => {document.removeEventListener('pointerdown', down, true); document.removeEventListener('keydown', key);};
  }, [open]);
}

const ITEMS = '[role^="menuitem"]:not([disabled]):not([aria-disabled="true"])';

/** Arrow keys, Home and End move between a menu's items; Tab leaves and closes it. */
function menuKeys(event: KeyboardEvent<HTMLElement>, close: (refocus: boolean) => void) {
  const items = [...event.currentTarget.querySelectorAll<HTMLElement>(ITEMS)].filter(item => !item.closest('[hidden]') && item.offsetParent !== null);
  const at = items.indexOf(document.activeElement as HTMLElement);
  const move = (index: number) => {event.preventDefault(); items[(index + items.length) % items.length]?.focus();};
  if (event.key === 'ArrowDown') move(at + 1);
  else if (event.key === 'ArrowUp') move(at < 0 ? items.length - 1 : at - 1);
  else if (event.key === 'Home') move(0);
  else if (event.key === 'End') move(items.length - 1);
  else if (event.key === 'Tab') close(false);
}

/**
 * A button that opens a menu of actions. Children receive `close`; items are `MenuItem`s (or links with
 * `role="menuitem"`). The first item takes focus on opening; Escape returns focus to the button.
 */
export function MenuButton({label, buttonContent, className = '', buttonClassName = 'docattentionlens-button', menuLabel, title, align = 'end', children, testId, buttonData}: {
  /** Accessible name of the button. */
  label: string; buttonContent: ReactNode; className?: string; buttonClassName?: string;
  menuLabel?: string; title?: string; align?: 'start' | 'end'; testId?: string;
  /** `data-*` attributes of the button, such as the method bar's `data-step`. */
  buttonData?: Record<string, string>;
  children: (close: (refocus?: boolean) => void) => ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const button = useRef<HTMLButtonElement>(null), menu = useRef<HTMLDivElement>(null);
  const id = useId();
  const close = (refocus = true) => {setOpen(false); if (refocus) button.current?.focus();};
  useDismiss(open, close, [button, menu]);
  useEffect(() => {if (open) menu.current?.querySelector<HTMLElement>(ITEMS)?.focus();}, [open]);
  return <div className={`docattentionlens-menu ${className}`} data-testid={testId}>
    <button ref={button} type="button" className={buttonClassName} aria-label={label} title={title ?? label} aria-haspopup="menu" aria-expanded={open}
      aria-controls={open ? id : undefined} {...buttonData} onClick={() => setOpen(v => !v)}
      onKeyDown={e => {if (e.key === 'ArrowDown' && !open) {e.preventDefault(); setOpen(true);}}}>{buttonContent}</button>
    {open && <div ref={menu} id={id} role="menu" aria-label={menuLabel ?? label} className={`docattentionlens-menu-list is-${align}`} onKeyDown={e => menuKeys(e, close)}>
      {children(close)}
    </div>}
  </div>;
}

export function MenuItem({children, onSelect, disabled, checked, title}: {children: ReactNode; onSelect: () => void; disabled?: boolean; checked?: boolean; title?: string}) {
  return <button type="button" role={checked === undefined ? 'menuitem' : 'menuitemradio'} aria-checked={checked} tabIndex={-1} disabled={disabled}
    title={title} className="docattentionlens-menu-item" onClick={onSelect}>{children}</button>;
}

export function MenuGroup({label, children}: {label: string; children: ReactNode}) {
  const id = useId();
  return <div role="group" aria-labelledby={id} className="docattentionlens-menu-group"><span id={id} className="docattentionlens-menu-group-label">{label}</span>{children}</div>;
}

/**
 * A button that opens a non-modal panel (role dialog) below it, for content that is read rather than
 * chosen from, such as case details. Escape and a press outside close it.
 */
export function PopoverButton({label, buttonContent, buttonClassName = 'docattentionlens-button', className = '', panelLabel, title, align = 'start', children, open: controlled, onOpenChange}: {
  label?: string; buttonContent: ReactNode; buttonClassName?: string; className?: string; panelLabel: string; title?: string;
  align?: 'start' | 'end'; children: (close: (refocus?: boolean) => void) => ReactNode;
  open?: boolean; onOpenChange?: (open: boolean) => void;
}) {
  const [own, setOwn] = useState(false);
  const open = controlled ?? own;
  const setOpen = (next: boolean) => {setOwn(next); onOpenChange?.(next);};
  const button = useRef<HTMLButtonElement>(null), panel = useRef<HTMLDivElement>(null);
  const id = useId();
  const close = (refocus = true) => {setOpen(false); if (refocus) button.current?.focus();};
  useDismiss(open, close, [button, panel]);
  return <div className={`docattentionlens-popover-anchor ${className}`}>
    <button ref={button} type="button" className={buttonClassName} aria-label={label} title={title} aria-expanded={open} aria-controls={open ? id : undefined}
      onClick={() => setOpen(!open)}>{buttonContent}</button>
    {open && <div ref={panel} id={id} role="dialog" aria-label={panelLabel} className={`docattentionlens-popover is-${align}`}>{children(close)}</div>}
  </div>;
}
