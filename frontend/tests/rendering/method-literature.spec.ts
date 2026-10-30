import { expect, test } from '@playwright/test';
import { catalogue, mockAttentionApi } from './mockApi';

// Method detail pages state how each calculation relates to its cited literature,
// using the generated catalogue rather than hand-written expectations.
for (const [method_id, label] of [['attention.sinks', 'Standard summary'], ['attention.sink_detection', 'Adaptation'],
                                  ['attention.rollout_evolution', 'Published method']] as const) {
  test(`${method_id} shows its literature relation`, async ({page}) => {
    await mockAttentionApi(page);
    const relation = catalogue.find(m => m.method_id === method_id)!.literature_relation!;
    await page.goto(`/toolbox/${encodeURIComponent(method_id)}`);
    const shown = page.getByTestId('literature-relation');
    await expect(shown.locator('.docattentionlens-badge')).toHaveText(label);
    await expect(shown).toContainText(relation.note);
    if (method_id === 'attention.sinks') await page.screenshot({path: '../var/screenshots/method-literature-relation.png'});
  });
}

test('document-aware methods without a recorded relation show none', async ({page}) => {
  await mockAttentionApi(page);
  await page.goto('/toolbox/attention.element_allocation');
  await expect(page.getByRole('heading', {name: 'What it does'})).toBeVisible();
  await expect(page.getByTestId('literature-relation')).toHaveCount(0);
});
