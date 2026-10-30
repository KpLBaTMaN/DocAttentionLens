import { test, expect, type Page } from '@playwright/test';
import type { components } from '../../src/api/generated/schema';
import * as F from '../../fixtures/attention';
import { mockAttentionApi, tableRecord, tableCaseRecord } from './mockApi';

type S = components['schemas'];
const model: S['ModelRegistration'] = {registration_id: 'builtin', name: 'Fixture generator', model_id: 'fixture', revision: 'fixture', role: 'generation', architecture: {model_type: 'fixture'}, validation: 'UI fixture only',
  capabilities: {can_run: true, attention_available: true, signal_description: 'Captured attention', input_description: 'Question and pages', summary: 'Fixture coverage', reason: '', limits: {page_limit: 2}}};
const dataset = (id: string, name: string, version: string, table = false): S['DatasetSummary'] => ({dataset_id: id, name, dataset_version: version, adapter_version: 'fixture',
  counts: {documents: table ? 1 : 2, tables: table ? 1 : 0, questions: 2, elements: table ? 0 : 2}, domain_counts: {}, domain_aliases: {}, element_types: [], element_categories: [], question_types: [], quality_counts: {}, quality_issues: [], source_manifest: {}, source_file_count: 1, json_snapshot_sha256: 'fixture',
  coordinate_convention: 'page_image_pixels', geometry_validation: 'fixture', structure_kind: table ? 'table_grounding' : 'document_layout'});
const datasets = [dataset('dataset_fixture', 'Layout benchmark', 'version-one'), dataset('dataset_old', 'Layout benchmark', 'version-two'), dataset('dataset_table', 'Table benchmark', 'table-version', true)];
const tableDoc = {...F.documentRecord, document_id: 'doc_table', ref: {...F.documentRecord.ref, dataset_id: 'dataset_table'}, filename: 'table.pdf'};
const question: S['QuestionRecord'] = {key: 'q1', question_id: 'question_fixture', document_id: F.DOCUMENT_ID, ref: F.documentRecord.ref, native_id: 1,
  question: 'Original question?', document_scoped_question: 'Document question?', collection_scoped_question: 'Collection question?', answer: 'Top left', question_type: 'lookup',
  native_element_ids: [], support_element_ids: [], support_page_ids: [F.PAGE_ID], support_element_refs: [], support_page_refs: [], support_role: 'evaluation_annotation'};

async function mockPicker(page: Page) {
  const api = await mockAttentionApi(page, {table: true});
  const cases: Record<string, unknown>[] = [];
  let selectedCase = F.caseRecord;
  await page.route('**/api/v1/setup/models', route => route.fulfill({json: [model]}));
  await page.route(url => url.pathname === '/api/v1/datasets', route => route.fulfill({json: {items: datasets, total: datasets.length}}));
  await page.route(/\/api\/v1\/datasets\/[^/]+\/summary$/, route => route.fulfill({json: datasets.find(d => route.request().url().includes(d.dataset_id))!}));
  await page.route(url => url.pathname === '/api/v1/documents', route => {
    const url = new URL(route.request().url());
    const id = url.searchParams.get('dataset_id')!;
    const doc = id === 'dataset_table' ? tableDoc : {...F.documentRecord, ref: {...F.documentRecord.ref, dataset_id: id}, filename: url.searchParams.has('before') ? 'second.pdf' : F.documentRecord.filename};
    return route.fulfill({json: {items: url.searchParams.get('search') === 'missing' ? [] : [doc], total: id === 'dataset_table' ? 1 : 2, next_cursor: url.searchParams.has('before') ? null : 'next'}});
  });
  await page.route('**/api/v1/documents/doc_table', route => route.fulfill({json: tableDoc}));
  await page.route(url => url.pathname === '/api/v1/tables', route => route.fulfill({json: {items: [{...tableRecord, document_id: tableDoc.document_id}], total: 1}}));
  await page.route(url => url.pathname === '/api/v1/questions', route => {
    const table = new URL(route.request().url()).searchParams.get('document_id') === tableDoc.document_id;
    return route.fulfill({json: {items: [table ? {...question, question_id: 'question_table', question: tableCaseRecord.query, table_grounding: tableCaseRecord.table_grounding} : question], total: 1}});
  });
  await page.route(`**/api/v1/cases/${F.CASE_ID}`, route => route.fulfill({json: selectedCase}));
  await page.route(url => url.pathname === '/api/v1/cases', route => {
    const body = route.request().postDataJSON(); cases.push(body);
    selectedCase = body.task_scope === 'table' ? {...tableCaseRecord, dataset_id: 'dataset_table', document_id: tableDoc.document_id, query_variant: 'original', task_scope: 'table'}
      : {...F.caseRecord, query: body.query_variant === 'original' ? question.question : body.query_variant === 'document_scoped' ? question.document_scoped_question : question.collection_scoped_question, query_variant: body.query_variant, task_scope: body.task_scope};
    return route.fulfill({json: selectedCase});
  });
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/, route => {throw new Error(`Unexpected inference request: ${route.request().url()}`);});
  await page.goto('/attention-tool');
  await page.getByRole('button', {name: 'New investigation', exact: true}).click();
  await page.getByRole('button', {name: 'Continue', exact: true}).click();
  return {api, cases};
}

test('simplified setup preserves wording, page selection and draft recovery in both themes and on phones', async ({page}, info) => {
  const {api, cases} = await mockPicker(page);
  await expect(page.getByRole('heading', {level: 1})).toHaveText('Choose a question and pages');
  const picker = page.locator('.docattentionlens-question-picker');
  await expect(picker.getByLabel('Dataset', {exact: true}).locator('option')).toHaveText(['Layout benchmark · version-one', 'Layout benchmark · version-two', 'Table benchmark']);
  await expect(page.locator('.docattentionlens-setup-summary')).not.toContainText('supplied page');
  await expect(page.getByRole('button', {name: 'Review setup'})).toBeDisabled();
  await expect(page.locator('#docattentionlens-review-blocker')).toHaveText('Choose a question to continue');
  await expect(picker).not.toContainText(/Grounding:|annual_reports/);
  for (const theme of ['dark', 'light']) {
    if (theme === 'light') await page.getByRole('button', {name: 'Switch to White appearance'}).click();
    for (const width of [1440, 375]) {
      await page.setViewportSize({width, height: 900});
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.screenshot({path: info.outputPath(`picker-${theme}-${width}.png`), fullPage: true});
    }
  }
  await picker.getByLabel('Search documents', {exact: true}).fill('missing');
  await expect(picker).toContainText('No matching documents.');
  await picker.getByLabel('Search documents', {exact: true}).fill('');
  await picker.getByRole('button', {name: 'Next documents'}).click();
  await expect(picker.getByRole('button', {name: /second.pdf/})).toBeVisible();
  await picker.getByRole('button', {name: 'First documents'}).click();
  await picker.getByRole('button', {name: new RegExp(F.documentRecord.filename)}).click();
  await expect(picker.getByRole('button', {name: /Collection question/})).toBeVisible();
  await picker.locator('.docattentionlens-picker-wording summary').click();
  await picker.getByLabel('Question wording', {exact: true}).selectOption('original');
  await picker.getByLabel('Task scope', {exact: true}).selectOption('collection');
  await expect(picker.locator('.docattentionlens-picker-wording summary')).toContainText('Original · collection');
  await picker.getByRole('button', {name: /Original question/}).click();
  expect(cases.at(-1)).toMatchObject({question_id: question.question_id, query_variant: 'original', task_scope: 'collection'});
  await expect(page.locator('.docattentionlens-frozen-prompt')).toContainText(question.question);
  await expect(page.locator('.docattentionlens-setup-summary')).toContainText('1 supplied page');
  await expect(page.locator('.docattentionlens-setup-pages input:checked')).toHaveCount(1);
  await expect(page.getByRole('button', {name: 'Review setup'})).toBeEnabled();
  await page.reload();
  await page.getByText('Choose a different dataset question', {exact: true}).click();
  await expect(picker.locator('.docattentionlens-picker-wording summary')).toContainText('Original · collection');
  await page.getByRole('button', {name: 'Review setup'}).click();
  await expect(page.locator('.docattentionlens-setup-review')).toContainText(question.question);
  expect(api.unexpected).toEqual([]);
});

test('changing a dataset clears accepted inputs and retains table-specific selection and scope', async ({page}) => {
  const {api, cases} = await mockPicker(page);
  const picker = page.locator('.docattentionlens-question-picker');
  await picker.getByRole('button', {name: new RegExp(F.documentRecord.filename)}).click();
  await picker.getByRole('button', {name: /Collection question/}).click();
  await page.getByText('Choose a different dataset question', {exact: true}).click();
  await picker.getByLabel('Dataset', {exact: true}).selectOption('dataset_old');
  await expect(picker.getByLabel('Dataset', {exact: true})).toHaveValue('dataset_old');
  await expect(page.getByRole('button', {name: 'Review setup'})).toBeDisabled();
  await expect(page.locator('.docattentionlens-frozen-prompt')).toHaveCount(0);
  const draft = await page.evaluate(() => JSON.parse(localStorage.getItem('docattentionlens-investigation-setup-v2')!));
  expect(draft.caseId).toBeUndefined(); expect(draft.supplied).toEqual([]);
  await picker.getByLabel('Dataset', {exact: true}).selectOption('dataset_table');
  await expect(picker.getByRole('region', {name: 'Tables'})).toBeVisible();
  await picker.getByTestId('table-choice').click();
  await expect(picker.locator('.docattentionlens-picker-wording')).toHaveCount(0);
  await expect(picker.getByTestId('table-question')).toContainText('header path');
  await picker.getByTestId('table-question').click();
  expect(cases.at(-1)).toMatchObject({question_id: 'question_table', query_variant: 'original', task_scope: 'table'});
  await expect(page.getByRole('button', {name: 'Review setup'})).toBeEnabled();
  await page.getByRole('button', {name: 'Review setup'}).click();
  await expect(page.locator('.docattentionlens-setup-review')).toContainText('Table grounding');
  expect(api.unexpected).toEqual([]);
});
