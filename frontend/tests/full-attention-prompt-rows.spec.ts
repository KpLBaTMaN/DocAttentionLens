import { test, expect } from '@playwright/test';
import { findGenerationCapture, requireSaved } from './savedEvidence';
import { chooseTokenView, openTemplateRuns } from './workspaceHelpers';

/**
 * A full-attention capture keeps every prefill row, so template tokens in the full prompt become sources for the
 * ordinary row methods (read exactly from the saved triangle). Captures made before generation schema 0.8.0 kept the
 * first eight prediction steps only: their later answer tokens stay unclickable and say why. A 0.8.0 capture keeps
 * every answer token's row, so it has none (tests/rendering/all-answer-tokens.spec.ts covers both).
 */
test('template tokens are sources on a full-attention capture; uncaptured answer steps explain themselves', async ({page}) => {
  const capture = await findGenerationCapture(page.request, c => !!c.full_attention);
  requireSaved(!capture, 'Requires a saved full-attention generator capture');
  // A template word after the first page image: causal attention lets it reach the image (earlier tokens cannot).
  const firstImage = capture!.positions.find(p => p.role === 'image')!.index;
  const template = capture!.positions.find(p => p.role !== 'query' && p.role !== 'image' && p.role !== 'generated'
    && p.index > firstImage && p.index < capture!.full_attention!.position_count && /^[A-Za-z]/.test(p.piece.replace(/^[▁Ġ]/, '')));
  requireSaved(!template, 'Requires a template word after the page image and before the answer');
  // The row the page should draw, read through the same endpoint the workspace uses.
  const expected = await page.request.get(`/api/v1/generation/captures/${capture!.capture_id}/rows?layer=0&head=0&source_index=${template!.index}`).then(r => r.json());
  expect(expected.source_index).toBe(template!.index);
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/, route => route.abort());
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto('/attention-tool?generation=' + capture!.capture_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  const prompt = page.locator('.docattentionlens-prompt-tokens');
  await chooseTokenView(page, 'Prompt');
  await expect(prompt).toBeVisible();
  await openTemplateRuns(page);
  const token = prompt.locator(`button.docattentionlens-prompt-token.is-template`).filter({hasText: template!.piece.replace(/^[▁Ġ]/, '')}).first();
  await expect(token).toBeEnabled();
  await token.click();
  await expect(page.getByTestId('prediction-alignment').first()).toContainText(`(position ${template!.index}) · full attention row`);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  if (capture!.output_token_ids.length > capture!.steps.length) {
    const later = prompt.locator('.docattentionlens-prompt-token.is-answer:not(button)').first();
    await expect(later).toHaveAttribute('title', new RegExp(`not captured: attention rows are kept for the first ${capture!.steps.length} prediction steps`));
  }
});
