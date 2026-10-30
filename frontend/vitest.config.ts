import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// Pure logic runs in Node. Component tests opt in to a DOM with `// @vitest-environment happy-dom`.
export default defineConfig({
  plugins: [react()],
  test: { include: ['unit/**/*.test.{ts,tsx}'], environment: 'node' },
})
