# Phase 6: UI refinement

The existing five-page application now uses a compact, dark engineering interface.
The sidebar groups Workspace and Performance tasks, the header shows the current
page, and API connection state comes from the health endpoint. A skip link,
visible focus outlines, labeled table regions, and reduced-motion support improve
keyboard use. No dependencies were added.

## Design decisions

- Replace slogans, gradients, and oversized headings with concise page titles,
  flat surfaces, a blue primary accent, and shared spacing/control tokens.
- Keep overview counts grounded in API responses. Exact/IVF counts share one
  compact metric; report activity is ordered by its recorded timestamp. Loading
  and unavailable states remain distinct from an empty workspace.
- Retain selectable index cards while presenting type, metric, dimensions, and
  vector count as aligned properties. Training state uses labeled status colors.
  IVF parameters explain clusters and probes; saved archives remain separate.
- Explain batch insertion and a single query beside their JSON inputs. Preserve
  quoted int64 IDs and raw search-score formatting. Add confirmation before
  vector deletion and concise invalid-JSON feedback.
- Use horizontal latency/throughput bars to avoid angled or clipped engine and
  probe labels. A shared palette identifies each engine in charts and the table.
  Build stages and memory measurements use separately labeled neutral colors.
  Recall axes stay at 0–1; latency/throughput bars start at zero. RSS nulls remain
  missing values, and negative retained deltas remain representable.
- Show report distribution, timestamp, and configuration counts. Keep metric
  units in headings/tooltips and retain explanations of float precision, QPS,
  full probing, and RSS limitations. No measurements are fabricated.
- Keep the runner's fixed quick preset and existing job lifecycle. Queued/running
  states explain what is happening without inventing percentage progress;
  completed jobs show report links.

The chart presentation was extracted into `components/BenchmarkCharts.tsx` and
its shared color/number tokens into `components/chartTheme.ts`. Other changes
reuse existing components. API request structures, backend code, algorithms,
NPZ formats, authentication, Compose, and CI architecture are unchanged.

## Preview

From the repository root, run `uv sync --locked --extra web --extra benchmark`
and `uv run --extra web --extra benchmark python -m backend.serve`. In another
terminal, run `cd frontend`, `npm ci`, and `npm run dev`. Open
<http://127.0.0.1:5173>.

For the production frontend, run `docker compose up --build -d` and open
<http://127.0.0.1:8080>. Normal `docker compose down` retains saved data.
The [deployment guide](phase5_deployment.md) covers configuration and access
control. No images are published and no public deployment is configured.

## Validation

| Check | Result |
| --- | --- |
| TypeScript and production build | Passed locally and in the frontend Docker build |
| Vitest | All 8 existing tests passed |
| Existing Playwright workflows | All 4 passed, including the real bounded quick benchmark |
| New Playwright refinement checks | All 5 passed: all pages at 1366×768, 1920×1080, and 768×1024; JSON/deletion validation; loading/offline feedback |
| Python API integration tests | 64 passed; one existing Starlette/httpx deprecation warning |
| Ruff lint/format and frontend Prettier | Passed |
| Docker Compose builds and health | Both images built; isolated review services healthy |
| Browser against production Nginx | Deep routes and health proxy passed at all three viewport sizes; six published charts and 12 data rows rendered without browser errors |
| Real container HTTP smoke | Exact/IVF training, insert/search with extreme int64 IDs, save, published reports, and bounded four-engine benchmark passed |
| Container recreation | Saved Exact/IVF results and generated/published reports persisted; active catalogs reset as expected |

The original workflows passed in the initial browser run; the refinement checks
passed on their focused rerun after fixing their selectors. Screenshots were inspected directly,
including tablet chart labels and search results. Plot colors, axis units,
horizontal overflow, keyboard focus, and positive/error feedback were checked.
Existing tests retain their assertions; the only changes are the extracted-chart
import and accepting the intentional vector-deletion confirmation.

Browser screenshots are actual local test artifacts under ignored
`frontend/test-results/`; they are not published benchmark measurements.
Browser validation used Windows Edge; Firefox, Safari, and hosted CI were not run.
The complete unchanged Python engine suite was not rerun for these UI-only changes.
No commit, push, image publication, or public deployment was performed.
