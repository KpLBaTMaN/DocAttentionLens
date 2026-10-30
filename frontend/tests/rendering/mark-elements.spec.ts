/**
 * Mark elements end to end in the browser against an in-memory marking API: draw, type, reorder, undo,
 * autosave, review and Done back to setup. The backend's own behaviour is tested in tests/integration.
 */
import { expect, test, type Page } from '@playwright/test';
import { whitePng } from './mockApi';

const UPLOAD = 'document_upload', MARKED = 'document_marked';
const PAGES = [1, 2].map(n => ({page_id:`page_${n}`, page_number:n, width:400, height:300, image_url:`/api/v1/pages/page_${n}/image`, text_layer:n === 1}));
type Saved = {expected_revision:number; elements:{element_type:string; reading_order:number; page_id:string}[]; reviewed_page_ids:string[]; manual_order_page_ids:string[]};

async function mockMarking(page:Page) {
  const calls = {saves:[] as Saved[], fits:[] as {page_id:string; tighten:boolean; bbox:Record<string, number>}[], freezes:[] as number[], unexpected:[] as string[]};
  let draft = {document_id:UPLOAD, filename:'report.pdf', revision:0, pages:PAGES, elements:[] as unknown[], reviewed_page_ids:[] as string[], manual_order_page_ids:[] as string[], frozen:null as unknown};
  await page.route(url => url.pathname.startsWith('/api/') || url.pathname.startsWith('/health/'), async route => {
    const request = route.request(), path = new URL(request.url()).pathname, method = request.method();
    const json = (body:unknown, status = 200) => route.fulfill({status, contentType:'application/json', body:JSON.stringify(body)});
    if (path === '/health/ready') return json({status:'ready', dependencies:{}});
    if (path.endsWith('/image')) return route.fulfill({contentType:'image/png', body:whitePng(400, 300)});
    if (path === `/api/v1/documents/${UPLOAD}/marking` && method === 'GET') return json(draft);
    if (path === `/api/v1/documents/${UPLOAD}/marking` && method === 'PUT') {
      const body = request.postDataJSON() as Saved;
      if (body.expected_revision !== draft.revision) return json({error:{code:'revision_conflict', message:'Changed elsewhere'}}, 409);
      calls.saves.push(body);
      draft = {...draft, revision:draft.revision + 1, elements:body.elements, reviewed_page_ids:body.reviewed_page_ids, manual_order_page_ids:body.manual_order_page_ids};
      return json(draft);
    }
    if (path === `/api/v1/documents/${UPLOAD}/marking/fit`) {
      const body = request.postDataJSON();
      calls.fits.push(body);
      const b = body.bbox, text = body.page_id === 'page_1';
      return json({bbox:body.tighten ? {x1:b.x1 + 4, y1:b.y1 + 4, x2:b.x2 - 4, y2:b.y2 - 4} : b, tightened:body.tighten,
        text:text ? 'Fixture text' : null, text_source:text ? 'pdf_text_layer' : 'none'});
    }
    if (path === `/api/v1/documents/${UPLOAD}/marking/freeze`) {
      const body = request.postDataJSON();
      calls.freezes.push(body.expected_revision);
      if (body.expected_revision !== draft.revision) return json({error:{code:'revision_conflict', message:'Changed elsewhere'}}, 409);
      draft = {...draft, revision:draft.revision + 1, frozen:{document_id:MARKED, annotation_version:'a'.repeat(64), element_count:draft.elements.length}};
      const ref = {dataset_id:'dataset_marked', dataset_version:'v', domain:'uploaded', document_id:MARKED};
      return json({draft, document:{key:MARKED, document_id:MARKED, ref, native_id:MARKED, filename:'report.pdf', total_pages:2, page_ids:['m1', 'm2'],
        question_count:0, element_count:draft.elements.length, annotation_version:'a'.repeat(64), source_pdf_sha256:'', ocr_model:'pdf_text_layer', ocr_status:'pdf_text_layer'}}, 201);
    }
    calls.unexpected.push(`${method} ${path}`);
    return json({error:{code:'not_found', message:'Not part of this test'}}, 404);
  });
  return calls;
}

/** Drag across the shown page between two fractions of its width and height. */
async function draw(page:Page, from:[number, number], to:[number, number]) {
  const shown = (await page.locator('.docattentionlens-marking-canvas image').boundingBox())!;
  await page.mouse.move(shown.x + shown.width * from[0], shown.y + shown.height * from[1]);
  await page.mouse.down();
  await page.mouse.move(shown.x + shown.width * to[0], shown.y + shown.height * to[1], {steps:4});
  await page.mouse.up();
}

test('marks, orders and saves elements, then returns to setup with the marked version', async ({page}) => {
  await page.setViewportSize({width:1600, height:1000});
  const calls = await mockMarking(page);
  await page.goto(`/documents/${UPLOAD}/elements?return=setup`);
  await expect(page.getByText('report.pdf · page 1 of 2')).toBeVisible();
  await expect(page.getByRole('heading', {name:'How to mark this page'})).toBeVisible();

  // Done needs at least one element.
  await page.getByRole('button', {name:'Done'}).click();
  await expect(page.getByText('Mark at least one element before choosing Done.')).toBeVisible();
  expect(calls.freezes).toEqual([]);

  await page.keyboard.press('2');
  await expect(page.getByLabel('New boxes')).toHaveValue('heading');
  await draw(page, [.1, .1], [.5, .3]);
  const order = page.locator('.docattentionlens-marking-order li');
  await expect(order).toHaveCount(1);
  await expect(page.getByRole('heading', {name:'Element 1 · Heading'})).toBeVisible();
  await expect(page.getByLabel(/^Text/)).toHaveValue('Fixture text');
  const fitted = calls.fits[0];
  expect(fitted.tighten).toBe(true);
  for (const [key, value] of Object.entries({x1:40, y1:30, x2:200, y2:90})) expect(fitted.bbox[key]).toBeCloseTo(value, -0.5);

  await page.keyboard.press('3');
  await draw(page, [.1, .5], [.9, .7]);
  await expect(order).toHaveText([/1\s*Heading/, /2\s*Text/]);
  await expect(page.locator('.docattentionlens-mark-chip text')).toHaveText(['1 heading', '2 text']);

  // Alt+↑ moves the selected text first; the page's order is then set by hand until automatic order is chosen again.
  await page.keyboard.press('Alt+ArrowUp');
  await expect(order).toHaveText([/1\s*Text/, /2\s*Heading/]);
  await page.getByRole('button', {name:'Use automatic order'}).click();
  await expect(order).toHaveText([/1\s*Heading/, /2\s*Text/]);
  await page.keyboard.press('Control+z');
  await expect(order).toHaveText([/1\s*Text/, /2\s*Heading/]);
  await page.keyboard.press('Control+Shift+z');
  await expect(order).toHaveText([/1\s*Heading/, /2\s*Text/]);

  await expect(page.getByRole('status').filter({hasText:'Draft saved'})).toBeVisible();
  expect(calls.saves.at(-1)!.elements.map(e => [e.element_type, e.reading_order])).toEqual([['heading', 1], ['text', 2]]);

  // Enter marks the page reviewed and opens the next page to review.
  await page.locator('.docattentionlens-marking-canvas').focus();
  await page.keyboard.press('Enter');
  await expect(page.getByText('report.pdf · page 2 of 2')).toBeVisible();
  await expect(page.locator('.docattentionlens-marking-pages li').first()).toContainText('✓ Reviewed · 2 elements');
  expect(calls.unexpected).toEqual([]);

  await page.getByRole('button', {name:'Done'}).click();
  await page.waitForURL('**/attention-tool');
  expect(calls.saves.at(-1)!.reviewed_page_ids).toEqual(['page_1']);
  expect(calls.freezes).toEqual([calls.saves.length]);
  const setup = await page.evaluate(() => ({draft:JSON.parse(localStorage.getItem('docattentionlens-investigation-setup-v2')!), open:sessionStorage.getItem('docattentionlens-investigation-setup-open')}));
  expect(setup.open).toBe('1');
  expect([setup.draft.documentId, setup.draft.step, setup.draft.mode, setup.draft.caseId]).toEqual([MARKED, 1, 'upload', undefined]);
});
