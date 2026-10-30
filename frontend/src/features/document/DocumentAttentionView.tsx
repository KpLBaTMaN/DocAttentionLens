import { useId } from 'react';
import type { DisplaySettings } from '../../api/client';

type View = NonNullable<DisplaySettings['attention_element_view']>;

/** Page colouring is independent of the attention calculation and element totals. */
export function DocumentAttentionView({view, selected, onChange}: {view: View; selected: boolean; onChange: (view: View) => void}) {
  const id = useId();
  return <fieldset className="docattentionlens-document-page-values" aria-label="Page colouring">
    <legend className="docattentionlens-sr-only">Page colouring</legend>
    <span>Page colouring</span>
    <div className="docattentionlens-segmented" role="radiogroup" aria-label="Page attention values" onKeyDown={event=>{
      if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key))return;
      event.preventDefault();
      const choices=Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>('button:not(:disabled)'));
      const index=choices.indexOf(event.target as HTMLButtonElement);
      const next=choices[(index+(['ArrowRight','ArrowDown'].includes(event.key)?1:-1)+choices.length)%choices.length];
      next?.click();next?.focus();
    }}>
      <button type="button" role="radio" aria-checked={view === 'elements'} onClick={()=>onChange('elements')}>Element totals</button>
      <button type="button" role="radio" aria-checked={view === 'patches'} aria-describedby={`${id}-background`} onClick={()=>onChange('patches')}>Attention patches</button>
      {(selected || view === 'selected_element') && <button type="button" role="radio" disabled={!selected} aria-checked={view === 'selected_element'} onClick={()=>onChange('selected_element')}>Selected patches</button>}
    </div>
    <span id={`${id}-background`} className="docattentionlens-sr-only">Attention patches show the selected method’s whole page, including background.</span>
  </fieldset>;
}
