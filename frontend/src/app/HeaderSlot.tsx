import { createContext, type ReactNode, useContext, useEffect, useState } from 'react';
import { createPortal } from 'react-dom';

/** Id of the element in the top bar that holds route-specific context (for example the workspace breadcrumb). */
export const HEADER_CONTEXT_ID = 'docattentionlens-header-context';

type SlotState = { target: HTMLElement | null; setTarget: (element: HTMLElement | null) => void };
const SlotContext = createContext<SlotState | null>(null);

/** Shares the top bar's context element with routes. App renders it around the header and routes. */
export function HeaderSlotProvider({ children }: { children: ReactNode }) {
  const [target, setTarget] = useState<HTMLElement | null>(null);
  return <SlotContext.Provider value={{ target, setTarget }}>{children}</SlotContext.Provider>;
}

/** The empty context area of the top bar, between the area navigation and the right-hand controls. */
export function HeaderSlotTarget() {
  const slot = useContext(SlotContext);
  return <div id={HEADER_CONTEXT_ID} className="docattentionlens-header-context" ref={slot?.setTarget}/>;
}

/** Renders its children in the top bar's context area, or in place with `inline` when there is no top bar. */
export function HeaderSlot({ children, inline = false }: { children: ReactNode; inline?: boolean }) {
  const slot = useContext(SlotContext);
  const [found, setFound] = useState<HTMLElement | null>(null);
  useEffect(() => {
    if (slot) return;
    const locate = () => document.getElementById(HEADER_CONTEXT_ID);
    const existing = locate();
    if (existing) { setFound(existing); return; }
    const observer = new MutationObserver(() => {
      const element = locate();
      if (element) { setFound(element); observer.disconnect(); }
    });
    observer.observe(document.body, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, [slot]);
  const target = slot ? slot.target : found;
  if (target) return createPortal(children, target);
  return inline && !slot ? <>{children}</> : null;
}
