import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';

const read = (name: string) => JSON.parse(readFileSync(`../var/verification/${name}.json`, 'utf8'));
const generation = read('generation-smoke');
const experiment = read('experiment-smoke');
const evaluation = read('evaluation-smoke');
const infrastructure = read('smoke');
const routes = [
  ['/attention-tool', 'Investigate'],
  ['/attention-tool/' + generation.investigation_id, 'Investigate'],
  ['/toolbox', 'Methods'],
  ['/toolbox/retrieval_scoring', 'Methods'],
  ...['inputs', 'cases', 'runs', 'results', 'investigations', 'experiments', 'evaluations', 'jobs', 'exports'].map(section => ['/artifacts/' + section, 'Library']),
  ['/experiments/' + experiment.result_id, 'Library'],
  ['/evaluations/' + evaluation.study_id, 'Library'],
  ['/artifacts/jobs/' + infrastructure.job_id, 'Library'],
  ['/artifacts/operational_receipt/' + infrastructure.receipt_id, 'Library'],
];

for (const width of [1440, 390]) {
  test(`shared header contains the same navigation on every area and detail route at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    let expectedGeometry: unknown;
    for (const [route, activeLink] of routes) {
      await page.goto(route);
      const header = page.getByRole('banner');
      const nav = header.getByRole('navigation', { name: 'Main areas' });
      await expect(header).toHaveCount(1);
      await expect(nav.getByRole('link')).toHaveCount(3);
      await expect(nav.getByRole('link', { name: activeLink })).toHaveAttribute('aria-current', 'page');
      for (const name of ['Investigate', 'Methods', 'Library']) {
        await expect(nav.getByRole('link', { name })).toBeInViewport();
      }
      const geometry = await header.evaluate(element => {
        const box = (node: Element) => {
          const { x, y, width, height } = node.getBoundingClientRect();
          return { x, y, width, height };
        };
        return { header: box(element), links: Array.from(element.querySelectorAll('nav a'), box) };
      });
      // On phones an open investigation adds its own second row (question, Save, ⋯) under the shared one;
      // the shared row, its position and its links keep the same geometry everywhere.
      if (width < 900 && route.startsWith('/attention-tool/')) geometry.header.height = (expectedGeometry as typeof geometry | undefined)?.header.height ?? geometry.header.height;
      if (!expectedGeometry) expectedGeometry = geometry;
      expect(geometry).toEqual(expectedGeometry);
      if (route === '/toolbox') {
        await page.evaluate(() => window.scrollTo(0, 300));
        expect((await header.boundingBox())!.y).toBe(0);
        await page.screenshot({ path: `../var/screenshots/shared-header-${width}.png` });
      }
    }
    await page.goto('/attention-tool');
    const header = page.getByRole('banner');
    await header.evaluate(e => e.setAttribute('data-persistence-check', 'same-header'));
    await header.getByRole('button', { name: 'Switch to White appearance', exact: true }).click();
    for (const name of ['Methods', 'Library', 'Investigate']) {
      await header.getByRole('link', { name }).click();
      await expect(header).toHaveAttribute('data-persistence-check', 'same-header');
      await expect(page.locator('.docattentionlens-app')).toHaveAttribute('data-theme', 'light');
    }
    await page.reload();
    await expect(page.locator('.docattentionlens-app')).toHaveAttribute('data-theme', 'light');
    await expect(header.getByRole('button', { name: 'Switch to Charcoal appearance', exact: true })).toBeVisible();
  });
}
