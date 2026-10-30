import { test, expect, type Page } from '@playwright/test';
import type { components } from '../../src/api/generated/schema';
import * as F from '../../fixtures/attention';
import { mockAttentionApi } from './mockApi';

const key = 'docattentionlens-investigation-setup-v2';
const model: components['schemas']['ModelRegistration'] = {
  registration_id: 'builtin', name: 'Fixture generator', model_id: 'fixture', revision: 'fixture', role: 'generation',
  architecture: {model_type: 'fixture'}, validation: 'UI fixture only',
  capabilities: {can_run: true, attention_available: true, signal_description: 'Captured attention', input_description: 'Question and page images', summary: 'Fixture coverage', reason: '', limits: {page_limit: 2}},
};

async function mockEntry(page: Page) {
  const api = await mockAttentionApi(page, {generationCapture: {...F.generationCapture, validation: {status: 'passed_for_this_capture'}}});
  await page.route('**/api/v1/setup/models', route => route.fulfill({json: [model]}));
  await page.route(url => url.pathname === '/api/v1/datasets', route => route.fulfill({json: {items: [], total: 0}}));
  await page.route('**/api/v1/datasets/dataset_fixture/summary', route => route.fulfill({json: {source_manifest: {page_sources: []}}}));
  return api;
}

test('compact entry fits desktop and phone in both themes; examples reopen without inference', async ({page}, info) => {
  const api = await mockEntry(page);
  const launches: string[] = [];
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/, route => {launches.push(route.request().url()); return route.abort();});
  await page.goto('/attention-tool');
  const entry = page.locator('.docattentionlens-setup-home');
  await expect(entry.getByRole('heading', {level: 1})).toHaveText('Investigate document attention');
  await expect(entry.getByRole('button', {name: 'New investigation', exact: true})).toBeVisible();
  await expect(entry.locator('.docattentionlens-primary')).toHaveCount(1);
  await expect(entry.getByRole('region', {name: 'Unfinished setup'})).toHaveCount(0);
  await expect(entry.getByRole('link', {name: 'Saved investigations'})).toHaveAttribute('href', '/artifacts/investigations');
  await expect(entry.getByRole('status')).toContainText('Model runs are currently unavailable');
  await expect(entry).not.toContainText(/DocAttentionLens|Run your own|Explore an example|Looking for earlier work/);
  for (const theme of ['dark', 'light']) {
    if (theme === 'light') await page.getByRole('button', {name: 'Switch to White appearance'}).click();
    for (const width of [1280, 375]) {
      await page.setViewportSize({width, height: 820});
      await expect(entry.getByRole('button', {name: 'New investigation', exact: true})).toBeInViewport();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.screenshot({path: info.outputPath(`entry-${theme}-${width}.png`), fullPage: true});
    }
  }
  await page.getByRole('button', {name: 'Open an example'}).click();
  const examples = page.getByRole('region', {name: 'Saved examples'});
  await expect(examples).toContainText('answer correctness has not been evaluated');
  await examples.getByRole('button', {name: new RegExp(F.caseRecord.query.replace('?', '\\?'))}).click();
  await expect(page).toHaveURL(new RegExp(`generation=${F.GENERATION_ID}`));
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  expect(launches).toEqual([]);
  expect(api.unexpected).toEqual([]);
});

test('cancelling replacement preserves an unfinished setup; resuming and confirmed replacement work', async ({page}, info) => {
  const api = await mockEntry(page);
  await page.goto('/attention-tool');
  // A fresh setup opens directly, with no discard confirmation.
  await page.getByRole('button', {name: 'New investigation', exact: true}).click();
  const full = page.getByRole('checkbox', {name: /Save full attention matrices/});
  await full.check();
  await page.evaluate(() => sessionStorage.clear());
  await page.reload();
  const resume = page.getByRole('region', {name: 'Unfinished setup'});
  await expect(resume).toContainText(model.name);
  const saved = await page.evaluate(key => localStorage.getItem(key), key);
  page.once('dialog', async dialog => {
    expect(dialog.message()).toContain('replace your unfinished setup');
    await dialog.dismiss();
  });
  await page.getByRole('button', {name: 'New investigation', exact: true}).click();
  await expect(resume).toBeVisible();
  expect(await page.evaluate(key => localStorage.getItem(key), key)).toBe(saved);
  await resume.getByRole('button', {name: 'Resume', exact: true}).click();
  await expect(full).toBeChecked();
  await page.reload();
  await expect(full).toBeChecked();
  await page.evaluate(() => sessionStorage.clear());
  await page.reload();
  page.once('dialog', dialog => dialog.accept());
  await page.getByRole('button', {name: 'New investigation', exact: true}).click();
  await expect(full).not.toBeChecked();
  await expect.poll(() => page.evaluate(key => JSON.parse(localStorage.getItem(key)!).fullAttention, key)).toBe(false);
  // A real case supplies the filename and model in the compact recovery row.
  await page.evaluate(({key, caseId}) => {
    const draft = JSON.parse(localStorage.getItem(key)!);
    localStorage.setItem(key, JSON.stringify({...draft, caseId, step: 1}));
    sessionStorage.clear();
  }, {key, caseId: F.CASE_ID});
  await page.reload();
  await expect(resume).toContainText(F.documentRecord.filename);
  await expect(resume).toContainText(model.name);
  await page.setViewportSize({width: 375, height: 820});
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({path: info.outputPath('entry-unfinished-phone.png'), fullPage: true});
  await resume.getByRole('button', {name: 'Resume', exact: true}).click();
  await expect(page.getByRole('heading', {name: 'Choose a question and pages'})).toBeVisible();
  expect(api.unexpected).toEqual([]);
});

test('an empty examples list retains a route to saved runs', async ({page}) => {
  const api = await mockEntry(page);
  await page.route('**/api/v1/generation/captures?*', route => route.fulfill({json: {items: [], total: 0}}));
  await page.goto('/attention-tool');
  const toggle = page.getByRole('button', {name: 'Open an example'});
  await toggle.click();
  await expect(toggle).toHaveAttribute('aria-expanded', 'true');
  const examples = page.getByRole('region', {name: 'Saved examples'});
  await expect(examples.getByRole('link', {name: 'Browse saved runs'})).toHaveAttribute('href', '/artifacts/runs');
  await toggle.click();
  await expect(examples).toHaveCount(0);
  await expect(toggle).toHaveAttribute('aria-expanded', 'false');
  expect(api.unexpected).toEqual([]);
});
