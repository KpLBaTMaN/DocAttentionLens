import type { ReactNode } from 'react';

/** Props shared by every DocAttentionLens icon. Icons are decorative; give the control an accessible name. */
export type IconProps = { size?: number; className?: string };

function icon(name: string, paths: ReactNode) {
  function Icon({ size = 16, className }: IconProps) {
    return <svg className={className ? `docattentionlens-icon ${className}` : 'docattentionlens-icon'} width={size} height={size} viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round"
      aria-hidden="true" focusable="false">{paths}</svg>;
  }
  Icon.displayName = `${name}Icon`;
  return Icon;
}

export const ChevronDown = icon('ChevronDown', <path d="m6 9 6 6 6-6"/>);
export const ChevronLeft = icon('ChevronLeft', <path d="m15 6-6 6 6 6"/>);
export const ChevronRight = icon('ChevronRight', <path d="m9 6 6 6-6 6"/>);
export const More = icon('More', <><circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/></>);
export const PanelRight = icon('PanelRight', <><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M15 4v16"/></>);
export const Settings = icon('Settings', <>
  <path d="M12.2 2h-.4a2 2 0 0 0-2 2v.2a2 2 0 0 1-1 1.7l-.4.3a2 2 0 0 1-2 0l-.2-.1a2 2 0 0 0-2.7.7l-.2.4a2 2 0 0 0 .7 2.7l.2.1a2 2 0 0 1 1 1.7v.6a2 2 0 0 1-1 1.7l-.2.1a2 2 0 0 0-.7 2.7l.2.4a2 2 0 0 0 2.7.7l.2-.1a2 2 0 0 1 2 0l.4.3a2 2 0 0 1 1 1.7v.2a2 2 0 0 0 2 2h.4a2 2 0 0 0 2-2v-.2a2 2 0 0 1 1-1.7l.4-.3a2 2 0 0 1 2 0l.2.1a2 2 0 0 0 2.7-.7l.2-.4a2 2 0 0 0-.7-2.7l-.2-.1a2 2 0 0 1-1-1.7v-.6a2 2 0 0 1 1-1.7l.2-.1a2 2 0 0 0 .7-2.7l-.2-.4a2 2 0 0 0-2.7-.7l-.2.1a2 2 0 0 1-2 0l-.4-.3a2 2 0 0 1-1-1.7V4a2 2 0 0 0-2-2Z"/>
  <circle cx="12" cy="12" r="3"/>
</>);
export const Maximize = icon('Maximize', <path d="M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M16 21h3a2 2 0 0 0 2-2v-3"/>);
export const Minimize = icon('Minimize', <path d="M8 3v3a2 2 0 0 1-2 2H3M21 8h-3a2 2 0 0 1-2-2V3M3 16h3a2 2 0 0 1 2 2v3M16 21v-3a2 2 0 0 1 2-2h3"/>);
export const Minus = icon('Minus', <path d="M5 12h14"/>);
export const Plus = icon('Plus', <path d="M5 12h14M12 5v14"/>);
export const X = icon('X', <path d="M18 6 6 18M6 6l12 12"/>);
export const Search = icon('Search', <><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></>);
export const Info = icon('Info', <><circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/></>);
export const Sun = icon('Sun', <><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4"/></>);
export const Moon = icon('Moon', <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z"/>);
export const FileText = icon('FileText', <><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Z"/><path d="M14 3v5h5M9 13h6M9 17h6M9 9h1"/></>);
export const Files = icon('Files', <><path d="M15 2H9a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V7l-5-5Z"/><path d="M15 2v5h5M4 7v13a2 2 0 0 0 2 2h9"/></>);
export const Sliders = icon('Sliders', <path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0M14 4v4M8 10v4M16 16v4"/>);
