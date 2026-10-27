import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Link } from "react-router-dom";
import { Search, Upload, Trash2 } from "lucide-react";
import { parseVectorInput, request } from "../api/client";
import { useResource } from "../hooks/useResource";
import type { IndexInfo, SearchResult } from "../types";
import {
  Badge,
  Empty,
  ErrorBox,
  Feedback,
  Field,
  Loading,
  useAction,
} from "../components/UI";
export function Workspace() {
  const indexes = useResource<IndexInfo[]>("/indexes"),
    [params, setParams] = useSearchParams(),
    action = useAction();
  const id = params.get("index") || indexes.data?.[0]?.id || "",
    index = indexes.data?.find((i) => i.id === id);
  const [payload, setPayload] = useState(
      '{"vectors":[[1,0,0],[0.9,0.1,0],[0,1,0]],"ids":["101","102","103"]}',
    ),
    [query, setQuery] = useState("[1,0,0]"),
    [k, setK] = useState(5),
    [probe, setProbe] = useState(1),
    [deleted, setDeleted] = useState('["101"]'),
    [result, setResult] = useState<SearchResult | null>(null);
  useEffect(() => {
    setResult(null);
    setProbe(index?.configuration.nprobe ?? 1);
  }, [id, index?.configuration.nprobe]);
  return (
    <>
      <div className="page-title">
        <div>
          <h1>Vector workspace</h1>
          <p>Insert vector batches and search with a single query vector.</p>
        </div>
        <Badge
          tone={index ? (index.is_trained ? "success" : "warning") : "neutral"}
        >
          {index?.status ?? "No index selected"}
        </Badge>
      </div>
      <ErrorBox message={indexes.error} />
      <Feedback action={action} />
      {!!indexes.data?.length && (
        <div className="panel selector">
          <Field label="Active index">
            <select
              value={id}
              onChange={(e) => setParams({ index: e.target.value })}
            >
              <option value="">Select an index</option>
              {indexes.data?.map((i) => (
                <option key={i.id} value={i.id}>
                  {i.name} · {i.dimension}d · {i.metric}
                </option>
              ))}
            </select>
          </Field>
          {index && (
            <div className="selector-info">
              <strong>{index.size.toLocaleString()} vectors</strong>
              <span>
                {index.type.toUpperCase()} · {index.dimension} dimensions
              </span>
            </div>
          )}
        </div>
      )}
      {indexes.loading && !indexes.data ? (
        <Loading />
      ) : !index ? (
        <section className="panel workspace-empty" aria-label="Workspace setup">
          <div>
            <h2>
              {indexes.error
                ? "Indexes unavailable"
                : indexes.data?.length
                  ? "Select an active index"
                  : "No active indexes"}
            </h2>
            <p>
              {indexes.error
                ? "Check the API connection, then reload this page."
                : indexes.data?.length
                  ? "Choose an index above to insert vectors or run a search."
                  : "Create an index or load a saved archive to insert vectors and run searches."}
            </p>
          </div>
          <Link className="button primary" to="/indexes">
            Open index manager
          </Link>
        </section>
      ) : (
        <>
          <p className="hint">
            Inputs below are examples only. Nothing is inserted automatically.
            Use quoted decimal IDs to preserve int64 precision.
          </p>
          {!index.is_trained && (
            <ErrorBox message="This IVF index is untrained. Train it in the index manager first." />
          )}
          <div className="grid-two">
            <section className="panel">
              <div className="panel-heading">
                <h2>Insert vectors</h2>
                <Upload size={18} />
              </div>
              <form
                className="form"
                onSubmit={(e) => {
                  e.preventDefault();
                  void action.run(async () => {
                    await request(
                      `/indexes/${id}/vectors`,
                      "POST",
                      parseVectorInput(payload),
                    );
                    indexes.reload();
                  }, "Vectors inserted.");
                }}
              >
                <p className="hint" id="insert-help">
                  Batch input: a matrix with {index.dimension} numbers per
                  vector and one unique ID per row. Quote decimal IDs to retain
                  int64 precision.
                </p>
                <Field label="Vectors and IDs (JSON)">
                  <textarea
                    rows={8}
                    value={payload}
                    onChange={(e) => setPayload(e.target.value)}
                    required
                    spellCheck={false}
                    aria-describedby="insert-help"
                  />
                </Field>
                <button
                  type="button"
                  className="text-button"
                  onClick={() => {
                    const vector = Array.from(
                      { length: index.dimension },
                      (_, d) => (d === 0 ? 1 : 0),
                    );
                    setPayload(
                      JSON.stringify(
                        { vectors: [vector], ids: ["101"] },
                        null,
                        2,
                      ),
                    );
                    setQuery(JSON.stringify(vector));
                  }}
                >
                  Fill single-vector example for {index.dimension}d
                </button>
                <button
                  className="button"
                  disabled={action.busy || !index.is_trained}
                >
                  <Upload size={15} />
                  Insert vectors
                </button>
              </form>
            </section>
            <section className="panel">
              <div className="panel-heading">
                <h2>Nearest neighbors</h2>
                <Search size={18} />
              </div>
              <form
                className="form"
                onSubmit={(e) => {
                  e.preventDefault();
                  void action.run(async () => {
                    setResult(
                      await request<SearchResult>(
                        `/indexes/${id}/search`,
                        "POST",
                        {
                          query: JSON.parse(query),
                          k,
                          ...(index.type === "ivf" ? { nprobe: probe } : {}),
                        },
                      ),
                    );
                  }, "Search completed.");
                }}
              >
                <Field label="Query vector (JSON)">
                  <textarea
                    rows={3}
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    required
                    spellCheck={false}
                    aria-describedby="query-help"
                  />
                </Field>
                <p className="hint" id="query-help">
                  Single query: one JSON array containing {index.dimension}{" "}
                  numbers. k sets the maximum number of neighbors returned.
                </p>
                <div className="form-row">
                  <Field label="Neighbors (k)">
                    <input
                      type="number"
                      min={1}
                      max={1000}
                      value={k}
                      onChange={(e) => setK(Number(e.target.value))}
                      required
                    />
                  </Field>
                  {index.type === "ivf" && (
                    <Field label="Probes (nprobe)">
                      <input
                        type="number"
                        min={1}
                        max={index.configuration.nlist}
                        value={probe}
                        onChange={(e) => setProbe(Number(e.target.value))}
                        required
                      />
                    </Field>
                  )}
                </div>
                <p className="hint">
                  {index.metric === "euclidean"
                    ? "Euclidean distance · lower is better"
                    : "Raw " + index.metric + " score · higher is better"}
                </p>
                <button
                  className="button primary"
                  disabled={action.busy || !index.is_trained}
                >
                  <Search size={16} />
                  Search vectors
                </button>
              </form>
            </section>
          </div>
          {result && (
            <section className="panel detail">
              <div className="panel-heading">
                <h2>Search results</h2>
                <Badge good>{result.latency_ms.toFixed(3)} ms</Badge>
              </div>
              <p className="hint">
                {result.latency_scope} · {result.metric}
                {result.nprobe !== null ? ` · nprobe=${result.nprobe}` : ""}
              </p>
              {!result.matches.length ? (
                <Empty>
                  No candidates returned. An empty index or selected inverted
                  lists can produce no matches.
                </Empty>
              ) : (
                <div
                  className="table-scroll"
                  role="region"
                  aria-label="Search results table"
                  tabIndex={0}
                >
                  <table aria-label="Nearest neighbor search results">
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>Vector ID</th>
                        <th>
                          {result.metric === "euclidean"
                            ? "Raw distance ↓"
                            : "Raw score ↑"}
                        </th>
                        <th>Metric</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.matches.map((m) => (
                        <tr key={m.id}>
                          <td className="muted">
                            {String(m.rank).padStart(2, "0")}
                          </td>
                          <td className="mono">{m.id}</td>
                          <td className="score">{m.score.toPrecision(9)}</td>
                          <td>{result.metric}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          )}
          <section className="panel detail">
            <div className="panel-heading">
              <h2>Delete vector IDs</h2>
              <Trash2 size={17} />
            </div>
            <form
              className="form inline-form"
              onSubmit={(e) => {
                e.preventDefault();
                if (
                  !confirm(
                    `Delete these vector IDs from “${index.name}”? This cannot be undone for the active index. Saved archives are unchanged.`,
                  )
                )
                  return;
                void action.run(async () => {
                  await request(
                    `/indexes/${id}/vectors`,
                    "DELETE",
                    parseVectorInput(
                      JSON.stringify({ ids: JSON.parse(deleted) }),
                    ),
                  );
                  indexes.reload();
                  setResult(null);
                }, "Vector IDs deleted; missing IDs were ignored.");
              }}
            >
              <Field label="IDs to delete (JSON array)">
                <input
                  value={deleted}
                  onChange={(e) => setDeleted(e.target.value)}
                  required
                />
              </Field>
              <button className="button destructive" disabled={action.busy}>
                Delete IDs
              </button>
            </form>
          </section>
        </>
      )}
    </>
  );
}
