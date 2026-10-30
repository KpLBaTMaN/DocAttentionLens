import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// DOCATTENTIONLENS_API_URL points the dev server at another API, such as one on a disposable schema.
const api = process.env.DOCATTENTIONLENS_API_URL ?? 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': api, '/health': api } },
});
