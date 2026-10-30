import { useEffect, useLayoutEffect, useRef, useState, type FocusEvent, type KeyboardEvent, type ReactNode, type RefObject } from 'react';
import { createPortal } from 'react-dom';

/** Dismissal for a <details> menu; focus changes are ignored while a press is down (a Chrome/Edge crash). */
export function useMenuDismiss(menu: RefObject<HTMLDetailsElement | null>) {
  const pressing = useRef(false);
  useEffect(() => {
    const down = (event: PointerEvent) => {
      pressing.current = true;
      const element = menu.current;
      if (element?.open && !element.contains(event.target as Node)) element.open = false;
    };
    const up = () => {pressing.current = false;};
    document.addEventListener('pointerdown', down, true);
    document.addEventListener('pointerup', up, true);
    document.addEventListener('pointercancel', up, true);
    return () => {document.removeEventListener('pointerdown', down, true); document.removeEventListener('pointerup', up, true); document.removeEventListener('pointercancel', up, true);};
  }, [menu]);
  return {
    onKeyDown: (e: KeyboardEvent<HTMLDetailsElement>) => {if (e.key === 'Escape' && e.currentTarget.open) {e.preventDefault(); e.currentTarget.open = false; e.currentTarget.querySelector('summary')?.focus();}},
    onBlur: (e: FocusEvent<HTMLDetailsElement>) => {if (!pressing.current && e.relatedTarget && !e.currentTarget.contains(e.relatedTarget as Node)) e.currentTarget.open = false;},
  };
}

/**
 * Publishes an element's rendered height as a CSS variable on the document root, so pinned bands,
 * sticky offsets, scroll padding and the fitted page height share one budget when bands wrap or collapse.
 */
export function usePinnedHeight(target: RefObject<HTMLElement | null>, name: string) {
  const publish = () => {
    const element = target.current, root = document.documentElement;
    if (!element) return;
    const value = `${element.offsetHeight}px`;
    if (root.style.getPropertyValue(name) !== value) root.style.setProperty(name, value);
  };
  // Renders publish before paint; the observer covers wrapping caused by resizing and late fonts.
  useLayoutEffect(publish);
  useEffect(() => {
    const element = target.current;
    if (!element) return;
    const observer = new ResizeObserver(publish);
    observer.observe(element);
    return () => {observer.disconnect(); document.documentElement.style.removeProperty(name);};
  // The published value always reads the current element.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, name]);
}

/** While there are unsaved changes, warns before the page unloads and asks before a same-origin link leaves the workspace. */
export function useLeaveGuard(dirty: boolean) {
  useEffect(() => {
    const warn = (event:BeforeUnloadEvent) => {if(dirty){event.preventDefault();event.returnValue='';}};
    const navigateAway=(event:MouseEvent)=>{
      if(!dirty||event.ctrlKey||event.metaKey||event.shiftKey||event.button!==0) return;
      const anchor=(event.target as HTMLElement).closest?.('a');
      if(!anchor||anchor.target==='_blank'||anchor.origin!==location.origin||anchor.href===location.href) return;
      if(!window.confirm('This investigation has unsaved changes. Leave this workspace? Save first to keep page and element selections.')) event.preventDefault();
    };
    window.addEventListener('beforeunload', warn);
    window.document.addEventListener('click',navigateAway,true);
    return () => {window.removeEventListener('beforeunload',warn);window.document.removeEventListener('click',navigateAway,true);};
  }, [dirty]);
}

/** A stored per-browser preference, read once (for example to choose a new investigation's initial layout). */
export function readBrowserPreference(key: string, initial: boolean) {
  try { const stored = localStorage.getItem(key); return stored === null ? initial : stored === 'true'; } catch { return initial; }
}

/** Per-browser presentation preference; saved investigations are unaffected and storage may be unavailable. */
export function useBrowserPreference(key: string, initial: boolean): [boolean, (next: boolean) => void] {
  const [value, setValue] = useState(() => readBrowserPreference(key, initial));
  const change = (next: boolean) => {
    setValue(next);
    try { localStorage.setItem(key, String(next)); } catch { /* The choice still applies in this session. */ }
  };
  return [value, change];
}

/** A stored per-browser number, such as a panel height; undefined until one is chosen. */
export function useBrowserNumber(key: string): [number | undefined, (next: number) => void] {
  const [value, setValue] = useState<number | undefined>(() => {
    try { const stored = Number(localStorage.getItem(key)); return stored > 0 ? stored : undefined; } catch { return undefined; }
  });
  const change = (next: number) => {
    setValue(next);
    try { localStorage.setItem(key, String(Math.round(next))); } catch { /* The choice still applies in this session. */ }
  };
  return [value, change];
}

/** A stored per-browser choice among named values; anything else stored reads as `initial`. */
export function useBrowserChoice<T extends string>(key: string, initial: T, valid: (value: unknown) => value is T): [T, (next: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try { const stored = localStorage.getItem(key); return valid(stored) ? stored : initial; } catch { return initial; }
  });
  const change = (next: T) => {
    setValue(next);
    try { localStorage.setItem(key, next); } catch { /* The choice still applies in this session. */ }
  };
  return [value, change];
}

/** Whether a media query currently matches, following changes. */
export function useMediaQuery(query: string) {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const media = window.matchMedia(query);
    const change = () => setMatches(media.matches);
    change();
    media.addEventListener('change', change);
    return () => media.removeEventListener('change', change);
  }, [query]);
  return matches;
}

/** Above 900 px the workspace shows its panels side by side; narrower layouts use bottom sheets from the bottom bar. */
export function useWideLayout() {
  return useMediaQuery('(min-width: 901px)');
}

/** At 1600 px and wider the Details panel can stay open beside the page; the choice is remembered per browser. */
export const DETAILS_WIDE_QUERY = '(min-width: 1600px)';

/**
 * At 1200 px and wider the workspace has three columns: the page's elements left of the page and the selected
 * element right of it, with the method under the method bar and Details as a drawer over the element.
 */
export const LEDGER_QUERY = '(min-width: 1200px)';

export function Drawer({ title, onClose, children, open = true, wide = false }: { title: string; onClose: () => void; children: ReactNode; open?: boolean; wide?: boolean }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { if (open) ref.current?.showModal(); else ref.current?.close(); }, [open]);
  // Keep dialogs outside locally hidden/scrolling panels, including the mobile dock.
  return createPortal(<dialog ref={ref} className={`docattentionlens-workspace-drawer${wide?' docattentionlens-wide-drawer':''}`} aria-label={title} onCancel={onClose}>
    <header><h2>{title}</h2><button className="docattentionlens-button" aria-label="Close panel" onClick={onClose}>Close</button></header>
    <div>{children}</div>
  </dialog>,document.querySelector('.docattentionlens-app')??document.body);
}

export function ResizeHandle({ label, value, min, max, horizontal, reverse, onChange }: { label: string; value: number; min: number; max: number; horizontal?: boolean; reverse?: boolean; onChange: (value: number) => void }) {
  const drag = useRef<{ start: number; value: number } | undefined>(undefined);
  const clamp = (n: number) => Math.min(max, Math.max(min, Math.round(n)));
  return <div className={`docattentionlens-resize-handle ${horizontal ? 'is-horizontal' : ''}`} role="separator" tabIndex={0} aria-label={label} aria-orientation={horizontal ? 'horizontal' : 'vertical'} aria-valuemin={min} aria-valuemax={max} aria-valuenow={value}
    onPointerDown={e => { drag.current = { start: horizontal ? e.clientY : e.clientX, value }; e.currentTarget.setPointerCapture(e.pointerId); e.preventDefault(); }}
    onPointerMove={e => { if (drag.current) onChange(clamp(drag.current.value + ((horizontal ? e.clientY : e.clientX) - drag.current.start) * (reverse ? -1 : 1))); }}
    onPointerUp={() => { drag.current = undefined; }} onPointerCancel={() => { drag.current = undefined; }}
    onKeyDown={e => { const direction = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 0; if (direction) { e.preventDefault(); onChange(clamp(value + direction * 16 * (reverse ? -1 : 1))); } if (e.key === 'Home' || e.key === 'End') { e.preventDefault(); onChange(e.key === 'Home' ? min : max); } }}/>
}
