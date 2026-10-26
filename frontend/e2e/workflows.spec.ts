import { test, expect } from "@playwright/test";
test("Exact UI: create, insert, search, delete IDs, save, load", async ({
  page,
}) => {
  await page.goto("/indexes");
  await page.getByLabel("Index name").fill("Browser Exact");
  await page.getByRole("button", { name: "Create index", exact: true }).click();
  await expect(page.getByText("Index created.", { exact: true })).toBeVisible();
  const card = page.locator(".index-card").filter({ hasText: "Browser Exact" });
  await card.getByRole("link", { name: "Workspace" }).click();
  await page
    .getByRole("button", { name: "Insert vectors", exact: true })
    .click();
  await expect(
    page.getByText("Vectors inserted.", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Search vectors", exact: true })
    .click();
  await expect(
    page.getByRole("cell", { name: "101", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("cell", { name: "1.00000000", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Delete IDs", exact: true }).click();
  await expect(page.getByText("2 vectors", { exact: true })).toBeVisible();
  await page.goto("/indexes");
  await page
    .locator(".index-card")
    .filter({ hasText: "Browser Exact" })
    .getByRole("button", { name: "Save", exact: true })
    .click();
  await expect(page.getByText("Archive saved.", { exact: true })).toBeVisible();
  await page
    .getByRole("button", { name: "Load index", exact: true })
    .last()
    .click();
  await expect(
    page.getByText("Archive restored.", { exact: true }),
  ).toBeVisible();
});
test("IVF UI: train, insert and compare probe overrides", async ({ page }) => {
  await page.goto("/indexes");
  await page.getByLabel("Index name").fill("Browser IVF");
  await page.getByLabel("Index type").selectOption("ivf");
  await page.getByLabel("Distance metric").selectOption("euclidean");
  await page.getByLabel("Clusters (nlist)").fill("2");
  await page.getByRole("button", { name: "Create index", exact: true }).click();
  await page
    .getByLabel("Training vectors", { exact: true })
    .fill("[[-10,0,0],[-9,0,0],[9,0,0],[10,0,0]]");
  await page.getByRole("button", { name: "Train IVF index" }).click();
  await expect(page.getByText(/Training completed/)).toBeVisible();
  await page
    .locator(".index-card")
    .filter({ hasText: "Browser IVF" })
    .getByRole("link", { name: "Workspace" })
    .click();
  await page
    .getByLabel("Vectors and IDs (JSON)")
    .fill('{"vectors":[[-10,0,0],[10,0,0]],"ids":["501","502"]}');
  await page
    .getByRole("button", { name: "Insert vectors", exact: true })
    .click();
  await expect(
    page.getByText("Vectors inserted.", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("Query vector (JSON)").fill("[10,0,0]");
  await page.getByLabel("Probes (nprobe)").fill("1");
  await page
    .getByRole("button", { name: "Search vectors", exact: true })
    .click();
  await expect(page.locator("tbody tr")).toHaveCount(1);
  await page.getByLabel("Probes (nprobe)").fill("2");
  await page
    .getByRole("button", { name: "Search vectors", exact: true })
    .click();
  await expect(page.locator("tbody tr")).toHaveCount(2);
  await expect(
    page.getByRole("cell", { name: "20.0000000", exact: true }),
  ).toBeVisible();
});
test("published reports power all six dashboard charts", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const catalogue = await (
    await page.request.get("/api/v1/benchmarks/runs")
  ).json();
  const published = catalogue.runs.find(
    (run: { source: string }) => run.source === "published",
  );
  expect(published).toBeTruthy();
  await page.goto(`/benchmarks?run=${published.id}`);
  await expect(
    page.getByRole("heading", { name: "Benchmark dashboard", exact: true }),
  ).toBeVisible();
  await expect(
    page.locator(".chart .recharts-wrapper > svg.recharts-surface"),
  ).toHaveCount(6);
  await expect(page.locator("tbody tr")).toHaveCount(12);
  await page.screenshot({
    path: "test-results/dashboard-desktop.png",
    fullPage: true,
  });
  await page.getByLabel("Engine", { exact: true }).selectOption("faiss_flat");
  await expect(page.locator("tbody tr")).toHaveCount(1);
  await page.setViewportSize({ width: 768, height: 1024 });
  await expect(
    page.getByRole("heading", { name: "Search latency · mean ms" }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});
test("runner executes the real bounded CLI and opens its results", async ({
  page,
}) => {
  await page.goto("/runner");
  const response = page.waitForResponse(
    (r) =>
      r.url().endsWith("/benchmarks/runs") && r.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Start quick benchmark" }).click();
  const created = await response;
  expect(created.status()).toBe(202);
  const job = await created.json();
  const activity = page.locator(".job").filter({ hasText: job.id });
  await expect(activity.getByText("completed", { exact: true })).toBeVisible({
    timeout: 120000,
  });
  await activity.getByRole("link", { name: "View results" }).click();
  await expect(
    page.locator(".chart .recharts-wrapper > svg.recharts-surface"),
  ).toHaveCount(6);
  await expect(page.locator("tbody tr")).toHaveCount(8);
});
