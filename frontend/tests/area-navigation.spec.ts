import { test, expect } from '@playwright/test';

test('area navigation, theme persistence and truthful planned method details', async ({ page }) => {
  await page.goto('/attention-tool');
  await expect(page.getByRole('heading', { name: 'Investigate document attention', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'New investigation', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Switch to White appearance', exact: true }).click();
  await expect(page.locator('.docattentionlens-app')).toHaveAttribute('data-theme', 'light');
  await page.getByRole('link', { name: 'Methods', exact: true }).click();
  const questions = await page.request.get('/api/v1/methods').then(r => r.json());
  type Descriptor = {method_id: string; variant_id: string; family: string; status: string; perspective: string};
  const items: Descriptor[] = questions.items;
  // The page lists the API's own catalogue: every descriptor once (method and variant unique), grouped by family.
  expect(items.length).toBeGreaterThan(0);
  expect(new Set(items.map(m => `${m.method_id}/${m.variant_id}`)).size).toBe(items.length);
  await expect(page.locator('.docattentionlens-method-row')).toHaveCount(items.length);
  await expect(page.getByText(`${items.length} method variants`, { exact: true })).toBeVisible();
  await expect(page.locator('.docattentionlens-methods-tally')).toHaveText(`${items.length} methods`);
  const families = [...new Set(items.map(m => m.family))];
  await expect(page.locator('.docattentionlens-methods-family')).toHaveCount(families.length);
  // Each family section holds exactly that family's descriptors (section sizes match the catalogue's family sizes).
  const familySizes = families.map(f => items.filter(m => m.family === f).length).sort((a, b) => a - b);
  const sectionSizes = await page.locator('.docattentionlens-methods-family').evaluateAll(sections => sections.map(s =>
    [Number(s.querySelector('h2 .docattentionlens-methods-count')?.textContent), s.querySelectorAll('.docattentionlens-method-row').length]));
  for (const [stated, rows] of sectionSizes) expect(stated).toBe(rows);
  expect(sectionSizes.map(([, rows]) => rows).sort((a, b) => a - b)).toEqual(familySizes);
  // Statuses are shown as the catalogue states them: planned variants carry their badge and planned styling.
  const planned = items.filter(m => m.status === 'planned').length;
  await expect(page.locator('.docattentionlens-method-row.is-planned')).toHaveCount(planned);
  await expect(page.locator('.docattentionlens-method-row .docattentionlens-badge').filter({hasText: /^planned$/})).toHaveCount(planned);
  await expect(page.locator('.docattentionlens-method-row h3').first()).toHaveText('Raw attention');
  await expect(page.locator('.docattentionlens-method-row .docattentionlens-kicker').first()).toContainText('General');
  await page.getByRole('textbox', { name: 'Filter methods' }).fill('Document-aware');
  await expect(page.locator('.docattentionlens-method-row')).toHaveCount(questions.items.filter((m: {perspective: string}) => m.perspective === 'document_aware').length);
  await page.getByRole('textbox', { name: 'Filter methods' }).fill('');
  for (const method of questions.items) {
    await expect(page.getByText(method.question, { exact: true })).toHaveCount(0);
  }
  await page.getByRole('textbox', { name: 'Filter methods' }).fill('Retrieval score decomposition');
  await expect(page.locator('.docattentionlens-method-row')).toHaveCount(1);
  await page.locator('.docattentionlens-method-row').click();
  await expect(page.getByRole('link', { name: 'Open document workspace' })).toBeVisible();
  await expect(page.getByText('retrieval_scoring', { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Retrieval score decomposition' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'What it does' })).toBeVisible();
  await expect(page.locator('.docattentionlens-app')).toHaveAttribute('data-theme', 'light');
  await page.getByRole('button', { name: 'Switch to Charcoal appearance', exact: true }).click();
  await page.screenshot({ path: '../var/screenshots/toolbox.png', fullPage: true });
});

test('CPU job completes, receipt opens and both records survive refresh', async ({ page }) => {
  await page.goto('/artifacts/jobs');
  await expect(page.getByText('CPU worker connected', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Run CPU check' }).click();
  await expect(page).toHaveURL(/\/artifacts\/jobs\/job_/);
  await expect(page.getByRole('status').filter({ hasText: /^completed$/ })).toBeVisible({ timeout: 20000 });
  const jobURL = page.url();
  await page.reload();
  await expect(page.getByRole('link', { name: 'Open receipt' })).toBeVisible();
  await page.getByRole('link', { name: 'Open receipt' }).click();
  await expect(page.getByRole('heading', { name: 'CPU check completed' })).toBeVisible();
  await expect(page.getByText('Infrastructure verification', { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Operational receipt' })).toBeVisible();
  await expect(page.locator('pre').first()).toContainText('utf8_sha256_v1');
  await page.getByRole('link', { name: 'Open originating job' }).click();
  await expect(page).toHaveURL(jobURL);
  await page.screenshot({ path: '../var/screenshots/completed-job.png', fullPage: true });
});

test('narrow viewport preserves navigation and readable empty states', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/attention-tool');
  await expect(page.getByRole('link', { name: 'Methods', exact: true })).toBeVisible();
  await expect(page.getByRole('heading', {name:'Investigate document attention'})).toBeVisible();
  const overflowing = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  expect(overflowing).toBe(false);
  await page.screenshot({ path: '../var/screenshots/attention-mobile.png', fullPage: true });
});
