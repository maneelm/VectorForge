import { defineConfig } from "@playwright/test";
import { resolve } from "node:path";
const root = resolve(import.meta.dirname, "..");
export default defineConfig({
  testDir: "./e2e",
  workers: 1,
  timeout: 150000,
  expect: { timeout: 15000 },
  use: {
    baseURL: "http://127.0.0.1:5173",
    channel: process.env.PLAYWRIGHT_CHANNEL || "msedge",
    headless: true,
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command:
        "uv run --extra web --extra benchmark uvicorn backend.main:app --host 127.0.0.1 --port 8000",
      cwd: root,
      url: "http://127.0.0.1:8000/api/v1/health",
      timeout: 30000,
      reuseExistingServer: false,
      env: {
        VECTORFORGE_DATA_DIR: resolve(root, ".vectorforge/e2e/archives"),
        VECTORFORGE_RESULTS_DIR: resolve(root, ".vectorforge/e2e/results"),
      },
    },
    {
      command: "npm run dev",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: false,
      timeout: 30000,
    },
  ],
});
