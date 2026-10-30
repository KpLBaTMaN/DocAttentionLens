import { useState } from 'react';
import { downloadBlob,exportPageFigure,type ResearchReceipt as Receipt } from './receiptExport';

export function ResearchReceipt({receipt,caption,canExportFigure}:{receipt:Receipt;caption:string;canExportFigure:boolean}) {
  const [error,setError]=useState<string>();
  const [busy,setBusy]=useState(false);
  const prefix=`attention-${receipt.evidence.capture_id??'unsaved'}-${receipt.analysis.method_id.replaceAll('.','-')}`;
  return <section className="docattentionlens-research-receipt" aria-label="Research receipt"><h3>Research receipt</h3>
    <dl><dt>Evidence</dt><dd>{receipt.evidence.role} · {receipt.evidence.signal} · {receipt.evidence.capture_id??'No capture'}</dd>
      <dt>Method</dt><dd>{receipt.analysis.method_name} · {receipt.analysis.variant_id??'similarity'}</dd><dt>Quantity</dt><dd>{receipt.analysis.quantity??'Awaiting selected result'} · {receipt.analysis.units}</dd>
      <dt>Normalization</dt><dd>{receipt.analysis.normalization??'Awaiting selected result'}</dd><dt>Display</dt><dd>Page scale {receipt.presentation.page_scale_min.toPrecision(6)} → {receipt.presentation.page_scale_max.toPrecision(6)} · {receipt.presentation.scale_mode} · {receipt.presentation.display_mode} · smoothing {receipt.presentation.settings.attention_smooth&&receipt.evidence.signal==='attention'?'on (display only)':'off'}</dd>
      <dt>Coverage</dt><dd>Only saved source rows and target positions are available. The exact input, model revision, capture validation, array hashes and result are included in the receipt.</dd></dl>
    <p>{receipt.interpretation.attention}</p><div className="docattentionlens-actions"><button className="docattentionlens-button" disabled={!receipt.evidence.capture_id} onClick={()=>downloadBlob(new Blob([JSON.stringify(receipt,null,2)],{type:'application/json'}),`${prefix}-receipt.json`)}>Export receipt (JSON)</button>
      <button className="docattentionlens-button" disabled={busy||!canExportFigure} onClick={async()=>{setError(undefined);setBusy(true);try{const svgs=[...document.querySelectorAll<SVGSVGElement>('[data-testid="page-canvas"] svg')];downloadBlob(await exportPageFigure(receipt,svgs,caption),`${prefix}-page.svg`);}catch(reason){setError(reason instanceof Error?reason.message:'Figure export failed.');}finally{setBusy(false);}}}>{busy?'Preparing figure…':'Export page figure (SVG)'}</button></div>
    {!canExportFigure&&<p>A page figure is available when the selected signal has a loaded page map. Statistical results remain available in the receipt.</p>}{error&&<p role="alert">{error}</p>}
    <details><summary>Exact receipt</summary><pre>{JSON.stringify(receipt,null,2)}</pre></details>
  </section>;
}
