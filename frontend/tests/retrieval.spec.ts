import { currentPage, openTool, openDetailsTab, showEvidence, showRetrievalScores, setFit, choosePage, showPageList, showElements, saveInvestigation } from './workspaceHelpers';
import { test, expect } from '@playwright/test';
import { findRetrievalCapture, requireSaved } from './savedEvidence';

test('real saved retrieval links query rows, patches, ledger and investigation recovery',async({page})=>{
  // Recovery turns to the third ranked page, so the capture must rank at least three pages.
  // Preferred: a capture whose question starts after prefix tokens, so opening on the first question word is visible.
  const firstWord=(c:{query_positions:{eligible:boolean;role:string;index:number}[]})=>c.query_positions.find(p=>p.eligible&&p.role==='query_text')?.index??0;
  const found=await findRetrievalCapture(page.request,c=>c.pages.length>=3,c=>firstWord(c)>0);
  requireSaved(!found,'Requires a saved real retrieval capture ranking at least three pages; run scripts/retrieval_smoke.py with three or more candidate pages');
  const capture=found!;
  // MaxSim opens on the first eligible question word (position 0 is <bos>/prefix, which always matches the page's own);
  // a capture with no eligible question word stays on position 0. The second reading is the next eligible position.
  const opened=firstWord(capture);
  const next=capture.query_positions.find(p=>p.eligible&&p.index>opened)!.index;
  await page.setViewportSize({width:1600,height:1000});
  await page.goto('/attention-tool?capture='+capture.capture_id);
  await showRetrievalScores(page);
  // The workspace opens on the page with retriever attention when the capture has one, otherwise the first ranked page.
  const shown=capture.pages.find(p=>p.page_id===(capture.attention_encodings?.find(e=>e.pass_kind==='document')?.page_id??capture.pages[0].page_id))!;
  expect(await currentPage(page)).toBe(shown.page_id);
  const patchCount=shown.positions.filter(p=>p.bbox).length;
  await expect(page.getByTestId('similarity-patch')).toHaveCount(patchCount);
  await showPageList(page);
  await expect(page.getByRole('heading',{name:'Candidate comparison',exact:true})).toBeVisible();
  // The signal note is part of the element inspector in Details › Element.
  await showElements(page);
  await page.getByText('Signal interpretation',{exact:true}).click();
  await expect(page.getByText('Retrieval similarities are signed scores, not attention weights.',{exact:false})).toBeVisible();
  const first=await page.request.get(`/api/v1/retrieval/captures/${capture.capture_id}/rows?page_id=${shown.page_id}&query_index=${opened}`).then(r=>r.json());
  // The query position is MaxSim's source in Details › Method; its ledgers are in Result.
  await openDetailsTab(page,'Method');
  await expect(page.getByLabel('Overlay query position',{exact:true})).toHaveValue(String(opened));
  await openDetailsTab(page,'Result');
  await expect(page.getByTestId('retrieval-winner')).toContainText(first.maximum.toFixed(6));
  await openDetailsTab(page,'Method');
  await page.getByLabel('Overlay query position',{exact:true}).selectOption(String(next));
  await openDetailsTab(page,'Result');
  const second=await page.request.get(`/api/v1/retrieval/captures/${capture.capture_id}/rows?page_id=${shown.page_id}&query_index=${next}`).then(r=>r.json());
  expect(first.similarities).not.toEqual(second.similarities);
  await expect(page.getByTestId('retrieval-winner')).toContainText(second.maximum.toFixed(6));
  await showEvidence(page,'Element contributions');
  await expect(page.getByTestId('allocated-total')).toHaveText(shown.score.toFixed(6));
  const patch=shown.positions.find(p=>p.bbox)!;
  const assertAlignment=async()=>{
    const bounds=await page.evaluate(index=>{
      const image=document.querySelector('.docattentionlens-canvas image')!.getBoundingClientRect();
      const patch=document.querySelector(`[data-position-index="${index}"]`)!.getBoundingClientRect();
      return {image:{left:image.left,top:image.top,width:image.width},patch:{left:patch.left,top:patch.top,width:patch.width}};
    },patch.index);
    const scale=bounds.image.width/shown.width;
    expect(Math.abs(bounds.patch.left-bounds.image.left-patch.bbox![0]*scale)).toBeLessThan(1);
    expect(Math.abs(bounds.patch.top-bounds.image.top-patch.bbox![1]*scale)).toBeLessThan(1);
    expect(Math.abs(bounds.patch.width-(patch.bbox![2]-patch.bbox![0])*scale)).toBeLessThan(1);
  };
  await assertAlignment();
  await page.getByRole('button',{name:'Zoom in',exact:true}).click();await assertAlignment();
  await setFit(page,'page');
  await showEvidence(page,'Query winner ledger');
  await page.getByRole('button',{name:'Rescore without this winner vector',exact:true}).click();
  await expect(page.getByText('Frozen-vector score',{exact:false})).toBeVisible();
  await expect(page.getByText('inputs were not re-encoded.',{exact:false})).toBeVisible();
  await openTool(page,'Notes & name');
  await page.getByLabel('Investigation name').fill('Browser: real retrieval recovery');
  await saveInvestigation(page);
  await expect(page).toHaveURL(/\/attention-tool\/investigation_/);
  await page.reload();
  await openDetailsTab(page,'Method');
  await expect(page.getByLabel('Overlay query position',{exact:true})).toHaveValue(String(next));
  await expect(page.getByTestId('similarity-patch')).toHaveCount(patchCount);
  // Another ranked page (the third, or the first when the third is the one shown) carries its own score.
  const other=[capture.pages[2],capture.pages[0]].find(p=>p.page_id!==shown.page_id)!;
  await choosePage(page,other.page_id);
  await showEvidence(page,'Element contributions');
  await expect(page.getByTestId('allocated-total')).toHaveText(other.score.toFixed(6));
  await page.screenshot({path:'../var/screenshots/retrieval-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await expect(page.getByTestId('page-canvas').locator('image')).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)).toBe(false);
  await page.screenshot({path:'../var/screenshots/retrieval-mobile.png',fullPage:true});
});

test('GPU launch from a case creates a real capture and exposes the originating job',async({page})=>{
  test.setTimeout(180000);
  const plan=await page.request.get('/api/v1/retrieval/plan').then(r=>r.json());
  test.skip(!plan.available,'GPU execution is unavailable; saved evidence test still runs');
  const records=await page.request.get('/api/v1/retrieval/captures').then(r=>r.json());
  requireSaved(!records.items?.length,'Real capture supplies an existing frozen case for this test');
  await page.goto('/attention-tool?case='+records.items[0].case_id+'&configure=1');
  await page.locator('.docattentionlens-setup-steps').getByRole('button',{name:/Model/}).click();
  await page.getByRole('checkbox',{name:/Let a retriever choose the pages first/}).check();
  await page.getByRole('button',{name:/^(Continue|Review setup)$/}).click();
  const candidates=page.locator('.docattentionlens-setup-page-checks input');
  await expect(candidates.first()).toBeVisible();
  // Keep one default candidate (the evidence page when the case has one).
  const checked=page.locator('.docattentionlens-setup-page-checks input:checked');
  while(await checked.count()>1)await checked.last().uncheck();
  await page.getByRole('button',{name:'Retrieve selected pages',exact:true}).click();
  const ranking=page.getByLabel('Saved context ranking',{exact:true});
  await expect(ranking).not.toHaveValue('',{timeout:120000});
  const id=await ranking.inputValue();
  await page.goto('/attention-tool?capture='+id);
  await expect(page.getByTestId('similarity-patch')).toHaveCount(1024,{timeout:120000});
  await expect(page.getByTestId('retrieval-winner')).toBeVisible();
  await showPageList(page);
  await expect(page.getByRole('heading',{name:'Candidate comparison',exact:true})).toBeVisible();
});
