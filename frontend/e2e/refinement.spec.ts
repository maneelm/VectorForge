import { test, expect } from "@playwright/test";

test("empty workspace offers a compact setup path without redundant navigation", async ({
  page,
}) => {
  await page.route("**/api/v1/indexes", (route) => route.fulfill({ json: [] }));
  for (const viewport of [
    { width: 1366, height: 768 },
    { width: 768, height: 1024 },
  ]) {
    await page.setViewportSize(viewport);
    await page.goto("/workspace");
    const setup = page.getByRole("region", { name: "Workspace setup" });
    await expect(
      setup.getByRole("heading", { name: "No active indexes" }),
    ).toBeVisible();
    await expect(setup).toContainText("load a saved archive");
    await expect(page.getByLabel("Active index", { exact: true })).toHaveCount(
      0,
    );
    await expect(
      page.getByRole("banner").getByRole("link", { name: "Vector workspace" }),
    ).toHaveCount(0);
    await expect(
      page
        .getByRole("navigation")
        .getByRole("link", { name: "Vector workspace" }),
    ).toBeVisible();
    expect((await setup.boundingBox())!.height).toBeLessThan(180);
    await page.screenshot({
      path: test.info().outputPath(`empty-workspace-${viewport.width}.png`),
      fullPage: true,
    });
    await setup.getByRole("link", { name: "Open index manager" }).click();
    await expect(
      page.getByRole("heading", { name: "Index manager", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("banner").getByRole("link", { name: "Vector workspace" }),
    ).toBeVisible();
  }
});

const views = [
  { width: 1366, height: 768 },
  { width: 1920, height: 1080 },
  { width: 768, height: 1024 },
];

for (const viewport of views) {
  test(`all pages remain usable at ${viewport.width}x${viewport.height}`, async ({
    page,
  }) => {
    await page.setViewportSize(viewport);
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const created = await page.request.post("/api/v1/indexes", {
      data: {
        name: `Layout ${viewport.width}`,
        type: "exact",
        dimension: 3,
        metric: "cosine",
      },
    });
    expect(created.ok()).toBeTruthy();
    const index = await created.json();
    try {
      const inserted = await page.request.post(
        `/api/v1/indexes/${index.id}/vectors`,
        {
          data: {
            vectors: [[1, 0, 0]],
            ids: ["9223372036854775807"],
          },
        },
      );
      expect(inserted.ok()).toBeTruthy();
      const reports = await (
        await page.request.get("/api/v1/benchmarks/runs")
      ).json();
      const published = reports.runs.find(
        (run: { source: string }) => run.source === "published",
      );
      expect(published).toBeTruthy();
      for (const [path, heading] of [
        ["/", "Overview"],
        ["/indexes", "Index manager"],
        [`/workspace?index=${index.id}`, "Vector workspace"],
        [`/benchmarks?run=${published.id}`, "Benchmark dashboard"],
        ["/runner", "Benchmark runner"],
      ]) {
        await page.goto(path);
        await expect(
          page.getByRole("heading", { name: heading, exact: true }),
        ).toBeVisible();
        await expect(
          page.getByText("API connected", { exact: true }),
        ).toBeVisible();
        await expect(page.locator("nav a.active")).toHaveCount(1);
        if (path.startsWith("/workspace")) {
          await page
            .getByRole("button", { name: "Search vectors", exact: true })
            .click();
          await expect(
            page.getByRole("cell", {
              name: "9223372036854775807",
              exact: true,
            }),
          ).toBeVisible();
        }
        if (path.startsWith("/benchmarks")) {
          await expect(
            page.locator(".chart .recharts-wrapper > svg.recharts-surface"),
          ).toHaveCount(6);
          await expect(page.locator("tbody tr")).toHaveCount(12);
          // Every chart occupies real width and retains its category/axis labels.
          for (const chart of await page.locator(".chart").all()) {
            expect((await chart.boundingBox())!.width).toBeGreaterThan(300);
          }
          const bars = page
            .locator(".chart-panel")
            .first()
            .locator(".recharts-bar-rectangle path");
          await bars.first().hover();
          await expect(
            page.locator(".recharts-tooltip-wrapper").first(),
          ).toContainText("Mean latency (ms)");
        }
        expect(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
        ).toBeTruthy();
        await page.mouse.move(0, 0);
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.screenshot({
          path: test
            .info()
            .outputPath(`${heading.toLowerCase().replaceAll(" ", "-")}.png`),
          fullPage: true,
        });
      }
      await page.goto("/indexes");
      await page.keyboard.press("Tab");
      await expect(
        page.getByRole("link", { name: "Skip to content" }),
      ).toBeFocused();
      await page.keyboard.press("Enter");
      await expect(page.locator("main")).toBeFocused();
      await page.keyboard.press("Tab");
      const focused = page.locator(":focus");
      await expect(focused).toBeVisible();
      expect(
        await focused.evaluate(
          (element) => getComputedStyle(element).outlineStyle,
        ),
      ).not.toBe("none");
      await page
        .getByRole("navigation")
        .getByRole("link", { name: "Vector workspace" })
        .click();
      await expect(
        page.getByRole("heading", { name: "Vector workspace", exact: true }),
      ).toBeVisible();
      expect(errors).toEqual([]);
    } finally {
      expect(
        (await page.request.delete(`/api/v1/indexes/${index.id}`)).ok(),
      ).toBeTruthy();
    }
  });
}

test("workspace explains invalid JSON and cancelled deletions preserve vectors", async ({
  page,
}) => {
  const created = await page.request.post("/api/v1/indexes", {
    data: {
      name: "Validation review",
      type: "exact",
      dimension: 3,
      metric: "cosine",
    },
  });
  expect(created.ok()).toBeTruthy();
  const index = await created.json();
  try {
    await page.goto(`/workspace?index=${index.id}`);
    await page.getByLabel("Vectors and IDs (JSON)").fill("[broken json");
    await page
      .getByRole("button", { name: "Insert vectors", exact: true })
      .click();
    await expect(page.getByRole("alert")).toContainText("Invalid JSON");
    await page
      .getByLabel("Vectors and IDs (JSON)")
      .fill('{"vectors":[[1,0,0]],"ids":["101"]}');
    await page
      .getByRole("button", { name: "Insert vectors", exact: true })
      .click();
    await expect(
      page.getByText("Vectors inserted.", { exact: true }),
    ).toBeVisible();
    page.once("dialog", async (dialog) => {
      expect(dialog.message()).toContain("This cannot be undone");
      await dialog.dismiss();
    });
    await page.getByRole("button", { name: "Delete IDs", exact: true }).click();
    await page
      .getByRole("button", { name: "Search vectors", exact: true })
      .click();
    await expect(
      page.getByRole("cell", { name: "101", exact: true }),
    ).toBeVisible();
  } finally {
    expect(
      (await page.request.delete(`/api/v1/indexes/${index.id}`)).ok(),
    ).toBeTruthy();
  }
});

test("loading and connection failures do not appear as empty real data", async ({
  page,
}) => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/api/v1/indexes", async (route) => {
    await gate;
    await route.continue();
  });
  await page.goto("/indexes");
  await expect(
    page.getByRole("status").filter({ hasText: "Loading workspace" }).first(),
  ).toBeVisible();
  release();
  await expect(page.getByRole("status")).toHaveCount(0);
  await page.unroute("**/api/v1/indexes");
  await page.route("**/api/v1/**", (route) => route.abort());
  await page.goto("/");
  await expect(
    page.getByText("API unavailable", { exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("alert")).toContainText("Cannot reach the API");
  await expect(
    page.getByText("Active indexes could not be loaded.", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("No active indexes", { exact: true }),
  ).toHaveCount(0);
});
