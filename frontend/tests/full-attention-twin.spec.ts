import { test, expect, type APIRequestContext } from '@playwright/test';
import { openMethodPicker, methodChoice } from './workspaceHelpers';
import { requireSaved, type SavedGeneration } from './savedEvidence';

/** Every saved generator capture, newest first. */
async function allCaptures(request: APIRequestContext) {
  const items: SavedGeneration[] = [];
  let before: string | null | undefined;
  do {
    const page = await request.get(`/api/v1/generation/captures?limit=25${before ? `&before=${before}` : ''}`).then(r => r.json());
    items.push(...page.items);
    before = page.next_cursor;
  } while (before);
  return items;
}

const sameRun = (a: SavedGeneration, b: SavedGeneration) => a.case_id === b.case_id && a.retrieval_capture_id === b.retrieval_capture_id
  && a.context.system_prompt === b.context.system_prompt && a.provenance.model_registration_id === b.provenance.model_registration_id
  && a.context.selected_page_ids.join('|') === b.context.selected_page_ids.join('|');

/**
 * A capture saved before full attention existed offers its full-attention twin (the same run captured again) in the
 * method picker, where the matrix methods are unavailable; switching makes them available without a new run.
 */
test('an older capture offers its full-attention version of the same run', async ({page}) => {
  const captures = await allCaptures(page.request);
  const older = captures.find(c => !c.full_attention && c.validation.status === 'passed_for_this_capture'
    && captures.some(t => t.full_attention && sameRun(t, c)));
  requireSaved(!older, 'Requires a capture without full attention and a full-attention capture of the same run');
  const twin = captures.find(t => t.full_attention && sameRun(t, older!))!;
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/, route => route.abort());
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto('/attention-tool?generation=' + older!.capture_id);
  const picker = await openMethodPicker(page);
  await picker.getByRole('combobox', {name: 'Find a method'}).fill('rollout');
  await expect(picker.locator('[role="option"][data-method-key="attention.rollout/residual_mean_square_v1"]')).toHaveAttribute('aria-disabled', 'true');
  await expect(picker.getByTestId('full-attention-twin')).toBeVisible({timeout: 20_000});
  await picker.getByRole('button', {name: 'Switch to it', exact: true}).click();
  // The workspace now shows the twin, and rollout can be chosen.
  // The model chip names the shown capture by its last eight characters.
  await expect(page.getByText(twin.capture_id.slice(-8)).first()).toBeVisible({timeout: 20_000});
  const again = await openMethodPicker(page);
  await again.getByRole('combobox', {name: 'Find a method'}).fill('rollout');
  await expect(again.locator('[role="option"][data-method-key="attention.rollout/residual_mean_square_v1"]')).not.toHaveAttribute('aria-disabled', 'true', {timeout: 20_000});
  await expect(again.getByTestId('full-attention-twin')).toHaveCount(0);
  await again.locator('[role="option"][data-method-key="attention.rollout/residual_mean_square_v1"]').click();
  await expect(methodChoice(page)).toHaveAttribute('data-method-key', 'attention.rollout/residual_mean_square_v1');
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  expect(twin.full_attention!.square_limit).toBeGreaterThan(0);
});
