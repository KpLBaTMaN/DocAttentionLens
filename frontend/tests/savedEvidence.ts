import { test, type APIRequestContext } from '@playwright/test';
import type { components } from '../src/api/generated/schema';

export type SavedGeneration = components['schemas']['GenerationCapture'];

/**
 * Skips a test whose saved model evidence is missing. With DOCATTENTIONLENS_REQUIRE_SAVED_CAPTURES=1 the
 * test fails instead, so a verification run cannot pass green without exercising real captures.
 * Call it inside the test body, after looking up the evidence.
 */
export function requireSaved(missing: boolean, reason: string) {
  if (!missing) return;
  if (process.env.DOCATTENTIONLENS_REQUIRE_SAVED_CAPTURES === '1') throw new Error(`Saved evidence required (DOCATTENTIONLENS_REQUIRE_SAVED_CAPTURES=1): ${reason}`);
  test.skip(true, reason);
}

/**
 * Pages newest-first through every saved generator capture and returns the newest one meeting
 * `required`, choosing one that also meets `preferred` when any does. Tests pick evidence by their
 * own preconditions this way rather than by whichever capture was saved last. Undefined when none
 * qualifies; pass that to requireSaved.
 */
export async function findGenerationCapture(request: APIRequestContext, required: (capture: SavedGeneration) => boolean, preferred?: (capture: SavedGeneration) => boolean) {
  let fallback: SavedGeneration | undefined;
  let before: string | null | undefined;
  do {
    const query = new URLSearchParams({limit: '25'});
    if (before) query.set('before', before);
    const response = await request.get(`/api/v1/generation/captures?${query}`);
    if (!response.ok()) throw new Error(`Listing saved generator captures failed: ${response.status()} ${await response.text()}`);
    const page: components['schemas']['GenerationPage'] = await response.json();
    for (const capture of page.items) {
      if (!required(capture)) continue;
      if (!preferred || preferred(capture)) return capture;
      fallback ??= capture;
    }
    before = page.next_cursor;
  } while (before);
  return fallback;
}

export type SavedRetrieval = components['schemas']['RetrievalCapture'];

/**
 * Pages newest-first through every saved retrieval capture and returns the newest one meeting `required`
 * (preferring one that also meets `preferred`), so tests choose evidence by their own preconditions rather
 * than by whichever capture a GPU launch saved last. Undefined when none qualifies; pass that to requireSaved.
 */
export async function findRetrievalCapture(request: APIRequestContext, required: (capture: SavedRetrieval) => boolean, preferred?: (capture: SavedRetrieval) => boolean) {
  let fallback: SavedRetrieval | undefined;
  let before: string | null | undefined;
  do {
    const query = new URLSearchParams({limit: '25'});
    if (before) query.set('before', before);
    const response = await request.get(`/api/v1/retrieval/captures?${query}`);
    if (!response.ok()) throw new Error(`Listing saved retrieval captures failed: ${response.status()} ${await response.text()}`);
    const page: components['schemas']['CapturePage'] = await response.json();
    for (const capture of page.items) {
      if (!required(capture)) continue;
      if (!preferred || preferred(capture)) return capture;
      fallback ??= capture;
    }
    before = page.next_cursor;
  } while (before);
  return fallback;
}
