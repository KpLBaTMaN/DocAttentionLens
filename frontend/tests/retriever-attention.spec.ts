import { expectRowTotals, withPanelHidden, detailsPanel, chooseMethod, openMethodPicker, openDetailsTab, closeMethodPicker } from './workspaceHelpers';
import { test, expect } from '@playwright/test';

test('retriever attention explores its own encodings, axes and methods without generation',async({page})=>{
  test.setTimeout(90_000);
  const records=await page.request.get('/api/v1/retrieval/captures?limit=25').then(r=>r.json());
  const capture=records.items.find((c:{attention_encodings?:unknown[]})=>c.attention_encodings?.some((e:any)=>e.pass_kind==='document'&&e.layer_count===18));
  expect(capture).toBeTruthy();
  let runs=0;
  await page.route(/\/api\/v1\/(retrieval|generation)\/runs$/,route=>{runs++;return route.abort();});
  await page.goto(`/attention-tool?capture=${capture.capture_id}&role=retrieval`);
  const panel=await detailsPanel(page,'Retrieval panel');
  // Layer and head are on the scrubber under the page.
  await expect(page.getByLabel('Attention layer',{exact:true})).toHaveValue('0');
  await expect(page.getByLabel('Attention layer',{exact:true}).locator('option')).toHaveCount(18);
  await expect(page.getByLabel('Attention head',{exact:true}).locator('option')).toHaveCount(8);
  // General lists the general encoding variants plus planned general attention descriptors.
  const encodingMethods=await page.request.get('/api/v1/retrieval/attention-methods').then(r=>r.json());
  const planned=(await page.request.get('/api/v1/methods?status=planned').then(r=>r.json())).items
    .filter((m:{evidence_kind:string;family:string;perspective:string})=>m.perspective==='general'&&(m.evidence_kind.startsWith('attention_')||m.family==='head_and_representation'));
  // The method picker lists them by family; every family opened.
  const picker=await openMethodPicker(page);
  const closed=picker.locator('.docattentionlens-mp-family[aria-expanded="false"]');
  while(await closed.count())await closed.first().click();
  const listed=picker.locator('.docattentionlens-mp-section [role="option"]');
  await expect(listed).toHaveCount(encodingMethods.filter((m:{perspective:string})=>m.perspective==='general').length+planned.length);
  // Source 0 is an image patch: textual-source blocks require another source.
  await expect(picker.locator('.docattentionlens-mp-section [role="option"]:not([aria-disabled="true"])')).toHaveCount(encodingMethods.filter((m:{perspective:string;requires:string[]})=>m.perspective==='general'&&!m.requires.includes('captured_text_sources')).length);
  await closeMethodPicker(page);
  await expect(page.getByLabel('Attention layer slider',{exact:true})).toHaveAttribute('aria-valuemax','17');
  await expect(page.getByRole('radiogroup',{name:'Attention head slider'}).getByRole('radio')).toHaveCount(8);
  await expect(panel.locator('optgroup').first()).toBeAttached();
  await expect(page.getByLabel('Prediction step',{exact:true})).toHaveCount(0);
  await expect(page.getByRole('tab',{name:'Generation',exact:true})).toHaveCount(0);
  await expect(page.getByLabel('Saved attention capture',{exact:true})).toHaveCount(0);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout:20_000});
  await panel.getByLabel('Attention source position').selectOption('10');
  await page.getByLabel('Attention layer',{exact:true}).selectOption('17');
  await page.getByLabel('Attention head',{exact:true}).selectOption('7');
  await withPanelHidden(page,()=>expect(page.getByTestId('retriever-attention-context')).toContainText('Source 10 · layer 17 · head 7'));
  await expectRowTotals(page,'','retriever-attention-mass');
  await page.screenshot({path:'../var/screenshots/retriever-document-attention.png'});
  await chooseMethod(page,'attention.head_by_head/retriever_encoding_v1');
  // The method's output, with its comparison rows, is in Details › Result; readings and the source are in Method.
  await openDetailsTab(page,'Result');
  await expect(page.getByLabel('Comparison row',{exact:true})).toBeVisible({timeout:20_000});
  await page.getByLabel('Comparison row',{exact:true}).selectOption('7');
  await openDetailsTab(page,'Method');
  await panel.getByRole('button',{name:'Question',exact:true}).click();
  await chooseMethod(page,'attention.entropy/retriever_encoding_v1');
  await openDetailsTab(page,'Result');
  await expect(page.getByTestId('method-result').getByText('Normalized entropy',{exact:true})).toBeVisible({timeout:20_000});
  await expect(page.getByTestId('attention-patch')).toHaveCount(0);
  await openDetailsTab(page,'Method');
  await panel.getByLabel('Attention source position').selectOption('1');
  await page.getByRole('button',{name:'Save investigation',exact:true}).click();
  await expect(page).toHaveURL(/investigation_/);await page.reload();
  await openDetailsTab(page,'Method');
  await expect(panel.getByRole('button',{name:'Question',exact:true})).toHaveAttribute('aria-pressed','true');
  await expect(panel.getByLabel('Attention source position')).toHaveValue('1');
  await expect(page.getByLabel('Attention layer',{exact:true})).toHaveValue('17');
  await expect(page.getByLabel('Attention head',{exact:true})).toHaveValue('7');
  await openDetailsTab(page,'Result');
  await expect(page.getByTestId('method-result').getByText('Normalized entropy',{exact:true})).toBeVisible();
  // The signal switch heads Details › Method; MaxSim's query position replaces the attention source there.
  await openDetailsTab(page,'Method');
  await panel.getByLabel('Retrieval evidence type').selectOption('similarity');
  await expect(page.getByLabel('Overlay query position')).toBeVisible();
  await expect(panel.getByLabel('Attention source position')).toHaveCount(0);
  await openDetailsTab(page,'Result');
  await expect(page.getByTestId('retrieval-winner')).toBeVisible();
  await page.setViewportSize({width:390,height:844});
  // On phones the panel is a sheet: open it to switch the signal back.
  await openDetailsTab(page,'Method');
  await panel.getByLabel('Retrieval evidence type').selectOption('attention');
  await expect(panel.getByRole('group',{name:'What the model read'})).toBeInViewport();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await page.screenshot({path:'../var/screenshots/retriever-query-attention-mobile.png'});
  expect(runs).toBe(0);
});
