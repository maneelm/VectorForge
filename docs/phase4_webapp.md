# Phase 4: local search laboratory

VectorForge now has a FastAPI adapter and a React/TypeScript dashboard. The web
application calls the existing indexes and launches the existing Phase 3 CLI;
engine algorithms, ranking, score units, and NPZ schemas remain unchanged.
This is a single-user, single-process application, with no authentication or
multi-worker shared state. No Phase 5 deployment infrastructure is included.

## Install and start

From the repository root, using Python 3.12+ and uv:

```powershell
uv sync --locked --extra web --extra benchmark
uv run --extra web --extra benchmark uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open <http://127.0.0.1:5173>. Swagger is at <http://127.0.0.1:8000/docs> and
OpenAPI JSON at <http://127.0.0.1:8000/openapi.json>. Keep one Uvicorn worker.
Restarting/reloading the backend discards active indexes and job history; save an
index explicitly before restarting. No process is started merely by importing
the backend, constructing the app, or collecting tests.

The frontend lockfile is committed. Vite 6.4 supports the workspace's Node 20.17;
a compatible maintained Node LTS can also be used. Tailwind 4, React Router,
Lucide, and Recharts are installed through npm. The dashboard chart route is
loaded lazily. No external services, API keys, or database are needed.

For index operations and reading saved reports without FAISS, install only web:

```powershell
uv sync --locked --extra web
uv run --extra web uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

New four-engine jobs return 503 with the install command if optional benchmark
dependencies are missing. Existing reports remain readable without them.

## Configuration and state

| Setting | Default | Meaning |
| --- | --- | --- |
| VECTORFORGE_DATA_DIR | `<repo>/.vectorforge` | Application-owned saved index archives |
| VECTORFORGE_RESULTS_DIR | `<repo>/results` | Local reports and web job outputs |
| VITE_API_BASE | `/api/v1` | Frontend API URL; Vite proxies `/api` to port 8000 |

Backend variables are read from the process environment, not automatically from
a .env file. For example, before starting Uvicorn:

```powershell
$env:VECTORFORGE_DATA_DIR = 'C:\VectorForgeData'
$env:VECTORFORGE_RESULTS_DIR = 'C:\VectorForgeResults'
```

Frontend variables can be set in `frontend/.env.local`; see `.env.example`.
Only localhost/127.0.0.1 port 5173 origins are allowed by default. Trusted host
checks accept localhost and 127.0.0.1 (`testserver` is also allowed for tests).
Bind the unauthenticated backend to loopback. CORS is not an authentication system.

Active indexes and application UUIDs live in memory. Every index has its own lock;
the catalog lock is held briefly to add/remove or snapshot entries. Operations on
one index cannot corrupt it, while independent indexes can be used concurrently.
Deletion marks entries as deleted so queued operations cannot use stale entries.
List requests may wait for an index operation when collecting its metadata.

Saving creates a UUID-named NPZ archive and JSON application manifest under the
data directory. The manifest stores the name/type/configuration without changing
the engine NPZ format. Saved archives survive restarts. Loading validates both,
uses the existing engine loader, and creates a fresh application index UUID.
Deleting an active index does not delete saved archives. Invalid manifests are
omitted from the saved listing; attempts to load a known invalid archive return
a useful error. No arbitrary uploads, filesystem paths, or overwrite controls
are accepted from browsers.

## API summary

All routes use `/api/v1`. Request models forbid unexpected fields. Errors use
`{"error":{"code":"...","message":"...","details":[...]}}`, with details
present for validation errors. Common statuses are 422 invalid input, 404 missing
resource, 409 duplicate IDs/state/capacity, 413 size limits, 400 invalid archive or
report, and 503 unavailable benchmark dependencies.

| Method | Route | Behavior |
| --- | --- | --- |
| GET | `/health` | Health, dependency capability, bounded benchmark settings |
| POST / GET | `/indexes` | Create / list active indexes |
| GET / DELETE | `/indexes/{id}` | Details / delete active index |
| POST | `/indexes/{id}/train` | Train an empty IVF index |
| POST / DELETE | `/indexes/{id}/vectors` | Insert / delete vector IDs |
| POST | `/indexes/{id}/search` | Top-k with raw scores and API engine-call latency |
| POST | `/indexes/{id}/save` | Save a new archive, never overwrite |
| GET | `/indexes/saved` | List saved archive IDs |
| POST | `/indexes/load` | Load by archive UUID |
| GET / POST | `/benchmarks/runs` | Discover reports / queue bounded quick job |
| GET | `/benchmarks/runs/{id}` | Run metadata |
| GET | `/benchmarks/runs/{id}/results` | Validated normalized measurement rows |
| GET | `/benchmarks/jobs` | Recent in-memory jobs |
| GET | `/benchmarks/jobs/{id}` | queued/running/completed/failed state and result links |

### Example index and search workflow

In PowerShell with the backend running:

```powershell
$base = 'http://127.0.0.1:8000/api/v1'
$index = Invoke-RestMethod "$base/indexes" -Method Post -ContentType 'application/json' -Body '{"name":"demo","type":"exact","dimension":3,"metric":"cosine"}'
Invoke-RestMethod "$base/indexes/$($index.id)/vectors" -Method Post -ContentType 'application/json' -Body '{"vectors":[[1,0,0],[0.9,0.1,0],[0,1,0]],"ids":["101","102","103"]}'
Invoke-RestMethod "$base/indexes/$($index.id)/search" -Method Post -ContentType 'application/json' -Body '{"query":[1,0,0],"k":2}'
$saved = Invoke-RestMethod "$base/indexes/$($index.id)/save" -Method Post
Invoke-RestMethod "$base/indexes/load" -Method Post -ContentType 'application/json' -Body (@{archive_id=$saved.archive_id} | ConvertTo-Json)
```

Search matches have `id`, `rank` starting at 1, and raw `score`. For this query,
ID 101 scores 1; ID 102 scores approximately 0.993884. `latency_ms` is measured
at runtime with perf_counter_ns around the engine call; it is not a Phase 3
benchmark measurement. Cosine/dot favor higher scores; Euclidean favors lower
unsquared distance. The API never transforms engine scores.

Vector IDs accept signed int64 JSON integers or quoted decimal strings. Match IDs
are always returned as decimal strings so JavaScript can preserve all 64 bits.
The frontend rejects unsafe numeric IDs and asks for strings. Application index,
archive, job, and report identifiers are separate UUIDs.

For IVF, create with `type:"ivf"`, `nlist`, `nprobe`, `random_state`, `max_iter`,
and `tol`. POST a `{"vectors":[...]}` matrix to `/train` with at least nlist rows,
then explicitly insert vectors. Training inserts no data. Searching/inserting
before training returns 409; populated retraining is rejected. Per-search nprobe
overrides do not mutate the default. Exact indexes reject IVF parameters. Delete
uses `{"ids":["101"]}` and preserves the engine's missing/repeated-ID semantics.

## Frontend pages

- **Overview:** live active-index/vector counts and recent saved runs; a useful
  first-index empty state with navigation to creation/search.
- **Index manager:** Exact/IVF creation, details, training, saved archive list,
  save/load, and confirmed active-index deletion.
- **Vector workspace:** index selection, single/batch JSON insertion, explicit
  example filling, query/k/nprobe, ranked raw scores, and ID deletion.
- **Benchmarks:** real saved reports, workload/run/engine/probe filters, six
  interactive Recharts visualizations and a numeric results table. A selected
  report defines one workload; unrelated runs are never aggregated.
- **Runner:** bounded quick preset, seed/metric controls, queued/running/completed/
  failed states, logs, and links to completed results. No standard/full controls.

Forms use labels, keyboard-accessible controls, inline errors, busy states, and
success notices. Prefilled examples are input only; no user data is inserted or
overwritten automatically. The layout adapts from a desktop sidebar to compact
navigation on smaller screens.

## Benchmark data and jobs

Report discovery reads published `docs/benchmark_results/` and configured local
results. Opaque deterministic report UUIDs encode source/relative location; browser
input is never interpreted as a filesystem path. The reader validates Phase 3
schema versions, finite measurements, complete configuration sets, metadata, and
one consistent workload. Invalid reports are reported in `issues`, not silently
shown as valid measurements. JSON is authoritative; its scalar values correspond
to the existing Phase 3 CSV schema. No numbers are hardcoded in the application.

The charts show mean latency, measured QPS, recall versus latency, recall versus
nprobe, construction/training/insertion time, and baseline/retained-delta RSS.
Build/RSS rows are deduplicated by engine because one build is reused across probes.
Missing values stay absent or N/A, including training for Exact/Flat. Failed rows
retain their error details and do not create fabricated chart values. The page
explains float64/float32 differences, canonical Recall@k, native boundary ties,
RSS estimation, and the VectorForge IVF full-probe direct ranking optimization.

POST `/benchmarks/runs` accepts only `{preset:"quick",metric:"cosine",seed:42}`
(metric/seed can vary). A lazy, single-thread job executor launches:

```text
python -m vectorforge.benchmarks.cli --preset quick --size 1000 --dimension 32
  --nprobe 1,4,16 --metric <metric> --seed <seed> --threads 1 --output <owned job directory>
```

This is the existing CLI with a bounded size/dimension override: training_size=2000,
query_count=50, repetitions=3, nlist=16, max_iter=25. It does not change Phase 3
methodology. The HTTP request returns 202 immediately; job polling reports its
state. One job runs and one can wait. Native failures retain captured error/log
details, rather than masquerading as completed four-engine runs. The application
owns the process tree, terminates it on timeout/shutdown, and validates completed
reports before returning result IDs. Job history is in memory (last 50); reports
stay on disk. Avoid interactive indexing or other CPU-heavy work while measuring.

## Resource and security limits

| Resource | Limit |
| --- | --- |
| Active indexes | 16 |
| Dimensions | 1–2048 |
| Insert/delete batch | 1–2000 vectors or IDs |
| Training batch | 1–10000 rows |
| Coordinates per insert/train request | 200,000 |
| Indexed size | 50,000 vectors and 2,000,000 coordinates per index |
| IVF nlist / iterations | Up to 128 / 100 |
| Training work budget | N × dimension × nlist × max_iter <= 100 million |
| Search k | 1–1000 |
| HTTP body | 4 MiB, checked before JSON parsing |
| NPZ compressed / total uncompressed size | 64 MiB; up to 20 ZIP members |
| Application manifest | 16 KiB |
| Saved archive discovery | First 1000 manifest files |
| Report file / report rows | 5 MiB / 512 |
| Report discovery | Up to 200 reports and 2000 directories per root |
| Web benchmark concurrency / timeout | One active + one queued / 180 seconds |
| Existing web output disk budget | 128 MiB; manually archive/remove old runs |

This is not a sandbox for hostile local files, a public multi-user API, or a
production scheduler. Native libraries/NumPy allocate temporary memory beyond
stored arrays. Independent operations still compete for RAM and CPU. Do not use
multiple Uvicorn workers, expose this service publicly, or assume automatic index
restoration. No automatic deletion of saved archives or user benchmark outputs
occurs. UUID paths, root-confinement checks, and archive preflight checks protect
the browser-facing filesystem boundary; they do not replace OS permissions.

## Validation and troubleshooting

```powershell
uv sync --locked --extra web --extra benchmark
uv run --extra web --extra benchmark pytest -v
uv run ruff check .
uv run ruff format --check .
cd frontend
npm ci
npm run typecheck
npm test
npm run build
npm run test:e2e
```

The Playwright smoke suite starts the actual API and Vite servers, uses an
isolated `.vectorforge/e2e/` data/results directory, and shuts down its servers.
It drives Exact insertion/search/deletion/save/load, IVF training and different
probe results, six charts from the published report, and a real bounded four-engine
CLI job with result navigation. Ordinary Python job tests mock subprocesses;
these should not be confused with the real browser integration run.

Playwright defaults to installed Microsoft Edge, used for validation on Windows.
For bundled Chromium instead:

```powershell
npx playwright install chromium
$env:PLAYWRIGHT_CHANNEL = 'chromium'
npm run test:e2e
```

Stop existing servers on ports 8000/5173 before the smoke suite; it intentionally
refuses to reuse an unrelated running application. For API connection errors,
check the backend is listening on port 8000 and Vite's proxy configuration.
For training errors, check row dimensions, at least nlist samples, and that the
index is empty. For FAISS job errors, install both extras and inspect the execution
log. For an invalid report, fix/regenerate its source with the Phase 3 tools;
the dashboard does not reinterpret malformed or mixed workload data.

## Verified implementation results

Final validation in the Windows workspace:

| Check | Result |
| --- | --- |
| `uv sync --locked --extra benchmark --extra web` | Locked environment synchronized |
| `uv run --extra benchmark --extra web pytest -v` | 551 passed; no skips |
| Existing Phase 1–3 tests | All 487 passed within the complete suite; original files unchanged |
| New backend tests | 64 passed |
| `uv run ruff check .` | All checks passed |
| `uv run ruff format --check .` | Passed |
| `npm run typecheck` | Passed |
| `npm test` | 8 passed in 2 files |
| `npm run build` | Passed; lazy dashboard bundle generated |
| `npm run test:e2e` | 4 passed in installed Microsoft Edge |
| `npm audit` / installation audit | No vulnerabilities reported |

Python emitted one upstream Starlette TestClient deprecation warning about its
httpx adapter. It did not cause a failure; HTTP tests still use the requested
FastAPI TestClient/httpx stack. Initial browser assertions were corrected to count
plot surfaces instead of legend SVGs, explicitly choose the published report, and
use an unambiguous accessible name for the engine filter. Final checks have no
failures or skipped tests.

The browser smoke run verified an actual four-engine CLI job, eight successful
engine/probe rows, completed job polling, and navigation to all six result charts.
Its generated data stays under ignored `.vectorforge/e2e/`; screenshot/trace
artifacts are under ignored `frontend/test-results/`. No Phase 3 published reports
were overwritten. The smoke servers were shut down after validation.

The main additions are:

```text
backend/
  main.py
  schemas.py
  api/{indexes,benchmarks}.py
  core/{config,errors}.py
  services/{index_manager,index_persistence,reports,benchmark_manager}.py
frontend/
  package.json / package-lock.json / tsconfig.json / vite.config.ts
  playwright.config.ts / .env.example / index.html
  src/api/client.ts
  src/hooks/useResource.ts
  src/components/UI.tsx
  src/pages/{Overview,Indexes,Workspace,Benchmarks,Runner}.tsx
  src/{App,main}.tsx / types.ts / styles.css
  src/test/{setup,client.test,interactions.test}.{ts,tsx}
  e2e/workflows.spec.ts
tests/
  conftest.py
  test_api_indexes.py / test_api_vectors.py
  test_api_persistence.py / test_api_benchmarks.py
docs/phase4_webapp.md
```

Root `README.md`, `docs/architecture.md`, `.gitignore`, `pyproject.toml`, and
`uv.lock` were updated for usage, optional web dependencies, development tests,
and generated files. Existing `src/vectorforge/` implementation and original test
logic were not modified. No Git commit or push was created.
