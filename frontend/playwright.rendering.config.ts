import { defineConfig, devices } from '@playwright/test';

// Known-answer rendering tests. Every API call is answered by tests/rendering/mockApi.ts,
// so these need no database, GPU or saved capture. By default they start their own Vite
// server on the current source; set DOCATTENTIONLENS_TEST_BASE_URL to check a built deployment instead.
const port = Number(process.env.DOCATTENTIONLENS_RENDER_PORT ?? 4317);
const external = process.env.DOCATTENTIONLENS_TEST_BASE_URL;

export default defineConfig({
  testDir: './tests/rendering',
  outputDir: '../var/rendering-tests',
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: external ?? `http://127.0.0.1:${port}`,
    ...devices['Desktop Chrome'],
    channel: process.env.DOCATTENTIONLENS_TEST_BROWSER_CHANNEL ?? (process.platform === 'win32' ? 'msedge' : undefined),
    trace: 'retain-on-failure',
  },
  webServer: external ? undefined : {
    command: `npx vite --host 127.0.0.1 --port ${port} --strictPort`,
    url: `http://127.0.0.1:${port}/attention-tool`,
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
