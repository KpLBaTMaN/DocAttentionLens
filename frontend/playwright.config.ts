import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  // Known-answer rendering tests fake the API and have their own config (playwright.rendering.config.ts).
  testIgnore: ['rendering/**'],
  outputDir: '../var/browser-tests',
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: process.env.DOCATTENTIONLENS_TEST_BASE_URL ?? 'http://127.0.0.1:8080',
    ...devices['Desktop Chrome'],
    channel: process.env.DOCATTENTIONLENS_TEST_BROWSER_CHANNEL ?? (process.platform === 'win32' ? 'msedge' : undefined),
    trace: 'retain-on-failure',
  },
});
