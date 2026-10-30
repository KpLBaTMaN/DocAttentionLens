import { Component, createContext, type ReactNode, useCallback, useState } from 'react';
import { Link, Navigate, Route, Routes } from 'react-router-dom';
import { AppHeader } from './AppHeader';
import { HeaderSlotProvider } from './HeaderSlot';
import { api, unwrap } from '../api/client';
import { useRemote } from './remote';
import { AttentionTool } from '../features/attention-tool/AttentionTool';
import { MarkElements } from '../features/marking/MarkElements';
import { Toolbox } from '../features/toolbox/Toolbox';
import { ArtifactArea, JobView, ReceiptView } from '../features/artifacts/Artifacts';
import '../styles/document.css';
import '../styles/experiments.css';
import { ExperimentView } from '../features/experiments/Experiments';
import { EvaluationView } from '../features/evaluation/Evaluations';
import '../styles/evaluation.css';
import '../styles/retrieval.css';

export function Heading({ kicker, title, children }: { kicker: string; title: string; children?: ReactNode }) {
  return <section className="docattentionlens-heading"><div><span className="docattentionlens-kicker">{kicker}</span><h1 className="docattentionlens-title">{title}</h1></div>{children}</section>;
}

export function Message({ error, children }: { error?: boolean; children: ReactNode }) {
  return <div className={`docattentionlens-notice ${error ? 'docattentionlens-failure' : ''}`} role={error ? 'alert' : 'status'}>{children}</div>;
}

class Boundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? <Message error>The view could not open. Reload to recover saved work.</Message> : this.props.children; }
}

/** The app's appearance (Charcoal = dark, White = light), for menus outside the header that offer it. */
export const ThemeContext = createContext<{theme: string; setTheme: (theme: string) => void}>({theme: 'dark', setTheme: () => {}});

export default function App() {
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('docattentionlens-theme') === 'light' ? 'light' : 'dark'; } catch { return 'dark'; }
  });
  const health = useRemote(useCallback(async () => unwrap(await api.GET('/health/ready')), []), 5000);
  function chooseTheme(next: string) {
    setTheme(next);
    try { localStorage.setItem('docattentionlens-theme', next); } catch { /* Preference remains usable in this session. */ }
  }
  return <div className="docattentionlens-app" data-theme={theme}><ThemeContext.Provider value={{theme, setTheme: chooseTheme}}><HeaderSlotProvider>
    <a className="docattentionlens-skip" href="#main">Skip to main content</a>
    <AppHeader theme={theme} onThemeChange={chooseTheme} serviceState={health.error ? 'unavailable' : health.data?.status === 'ready' ? 'ready' : 'connecting'}/>
    <main id="main" tabIndex={-1}><Boundary><Routes>
      <Route path="/" element={<Navigate to="/attention-tool" replace/>}/>
      <Route path="/attention-tool" element={<AttentionTool/>}/>
      <Route path="/attention-tool/:investigationId" element={<AttentionTool/>}/>
      <Route path="/documents/:documentId/elements" element={<MarkElements/>}/>
      <Route path="/toolbox/:methodId?" element={<Toolbox/>}/>
      <Route path="/experiments/:experimentId" element={<ExperimentView/>}/>
      <Route path="/evaluations/:studyId" element={<EvaluationView/>}/>
      <Route path="/artifacts/jobs/:jobId" element={<JobView/>}/>
      <Route path="/artifacts/operational_receipt/:recordId" element={<ReceiptView/>}/>
      <Route path="/artifacts/:section?" element={<ArtifactArea/>}/>
      <Route path="*" element={<><Heading kicker="DocAttentionLens" title="Page not found"/><Message><Link to="/attention-tool">Return to Investigate</Link></Message></>}/>
    </Routes></Boundary></main>
  </HeaderSlotProvider></ThemeContext.Provider></div>;
}
