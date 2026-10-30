import { useEffect, useState } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import { HeaderSlotTarget } from './HeaderSlot';
import { Moon, Sun } from './icons';

export type ServiceState = 'ready' | 'connecting' | 'unavailable';

type AppHeaderProps = {
  theme: string;
  onThemeChange: (theme: string) => void;
  serviceState: ServiceState;
};

/** The persistent top bar shared by every route: area navigation, route context, service state and appearance. */
export function AppHeader({ theme, onThemeChange, serviceState }: AppHeaderProps) {
  const { pathname } = useLocation();
  const libraryActive = pathname === '/artifacts' || pathname.startsWith('/artifacts/') || pathname.startsWith('/experiments/') || pathname.startsWith('/evaluations/');
  const next = theme === 'light' ? 'dark' : 'light';
  // A first health check normally answers quickly; only a slow connection is worth announcing.
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    setSlow(false);
    if (serviceState !== 'connecting') return;
    const timer = setTimeout(() => setSlow(true), 1500);
    return () => clearTimeout(timer);
  }, [serviceState]);
  const shownState = serviceState === 'connecting' && !slow ? 'ready' : serviceState;

  return <header className="docattentionlens-header">
    <Link className="docattentionlens-brand" to="/attention-tool" aria-label="DocAttentionLens">
      <span className="docattentionlens-brand-word" aria-hidden="true">DocAttentionLens</span>
    </Link>
    <nav className="docattentionlens-primary-nav" aria-label="Main areas">
      {/* Marking a document's elements is part of setting up an investigation. */}
      <NavLink to="/attention-tool" className={({isActive}) => isActive || pathname.startsWith('/documents/') ? 'active' : ''}>Investigate</NavLink>
      <NavLink to="/toolbox">Methods</NavLink>
      <Link to="/artifacts" className={libraryActive ? 'active' : ''} aria-current={libraryActive ? 'page' : undefined}>Library</Link>
    </nav>
    <HeaderSlotTarget/>
    <span className="docattentionlens-service-status" role="status" data-state={shownState}>
      {shownState !== 'ready' && <><span className="docattentionlens-service-dot" aria-hidden="true"/><span className="docattentionlens-service-text">{shownState === 'unavailable' ? 'Service unavailable' : 'Connecting'}</span></>}
    </span>
    <button type="button" className="docattentionlens-icon-button docattentionlens-theme-toggle" onClick={() => onThemeChange(next)}
      aria-label={next === 'light' ? 'Switch to White appearance' : 'Switch to Charcoal appearance'}
      title={next === 'light' ? 'White appearance' : 'Charcoal appearance'}>
      {next === 'light' ? <Sun/> : <Moon/>}
    </button>
  </header>;
}
