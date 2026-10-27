import { useState } from "react";
import { Link } from "react-router-dom";
import { Database, Plus, Save, Trash2, ArrowUpRight } from "lucide-react";
import { request } from "../api/client";
import { useResource } from "../hooks/useResource";
import type { IndexInfo, SavedIndex } from "../types";
import {
  Badge,
  Empty,
  ErrorBox,
  Feedback,
  Field,
  Loading,
  useAction,
} from "../components/UI";
export function Indexes() {
  const indexes = useResource<IndexInfo[]>("/indexes"),
    saved = useResource<SavedIndex[]>("/indexes/saved"),
    action = useAction();
  const [type, setType] = useState("exact"),
    [selected, setSelected] = useState(""),
    [training, setTraining] = useState("");
  const current = indexes.data?.find((i) => i.id === selected);
  const refresh = () => {
    indexes.reload();
    saved.reload();
  };
  return (
    <>
      <div className="page-title">
        <div>
          <h1>Index manager</h1>
          <p>Create, train, and save Exact or IVF indexes.</p>
        </div>
        <Badge>{indexes.data?.length ?? "—"} active</Badge>
      </div>
      <Feedback action={action} />
      <ErrorBox message={indexes.error || saved.error} />
      <div className="grid-two manager-grid">
        <section className="panel">
          <div className="panel-heading">
            <h2>Create an index</h2>
            <Plus size={18} />
          </div>
          <form
            className="form"
            onSubmit={(e) => {
              e.preventDefault();
              const f = new FormData(e.currentTarget);
              void action.run(async () => {
                const body: Record<string, unknown> = {
                  name: f.get("name"),
                  type,
                  dimension: Number(f.get("dimension")),
                  metric: f.get("metric"),
                };
                if (type === "ivf")
                  for (const key of [
                    "nlist",
                    "nprobe",
                    "random_state",
                    "max_iter",
                    "tol",
                  ])
                    body[key] = Number(f.get(key));
                const created = await request<IndexInfo>(
                  "/indexes",
                  "POST",
                  body,
                );
                setSelected(created.id);
                refresh();
              }, "Index created.");
            }}
          >
            <Field label="Index name">
              <input
                name="name"
                required
                maxLength={80}
                placeholder="e.g. research-embeddings"
              />
            </Field>
            <div className="form-row">
              <Field label="Index type">
                <select
                  aria-label="Index type"
                  value={type}
                  onChange={(e) => setType(e.target.value)}
                >
                  <option value="exact">Exact · exhaustive</option>
                  <option value="ivf">IVF · approximate</option>
                </select>
              </Field>
              <Field label="Dimensions">
                <input
                  name="dimension"
                  type="number"
                  min={1}
                  max={2048}
                  defaultValue={3}
                  required
                />
              </Field>
            </div>
            <Field label="Distance metric">
              <select name="metric">
                <option value="cosine">Cosine similarity</option>
                <option value="euclidean">Euclidean distance</option>
                <option value="dot">Dot product</option>
              </select>
            </Field>
            {type === "ivf" && (
              <>
                <div className="form-section">
                  <h3>IVF configuration</h3>
                  <p className="hint">
                    nlist sets the number of clusters. nprobe controls how many
                    are searched and must not exceed nlist.
                  </p>
                </div>
                <div className="form-row">
                  <Field label="Clusters (nlist)">
                    <input
                      name="nlist"
                      type="number"
                      min={1}
                      max={128}
                      defaultValue={4}
                      required
                    />
                  </Field>
                  <Field label="Probes (nprobe)">
                    <input
                      name="nprobe"
                      type="number"
                      min={1}
                      defaultValue={1}
                      required
                    />
                  </Field>
                </div>
                <div className="form-row">
                  <Field label="Random seed">
                    <input
                      name="random_state"
                      type="number"
                      min={0}
                      max={2147483647}
                      defaultValue={42}
                      required
                    />
                  </Field>
                  <Field label="Maximum iterations">
                    <input
                      name="max_iter"
                      type="number"
                      min={1}
                      max={100}
                      defaultValue={100}
                      required
                    />
                  </Field>
                </div>
                <Field label="Convergence tolerance">
                  <input
                    name="tol"
                    type="number"
                    min={0}
                    max={1e6}
                    step="any"
                    defaultValue={0.0001}
                    required
                  />
                </Field>
                <p className="hint">
                  IVF needs training before vectors can be inserted or searched.
                </p>
              </>
            )}
            <button className="button primary" disabled={action.busy}>
              <Plus size={16} />
              {action.busy ? "Working…" : "Create index"}
            </button>
          </form>
        </section>
        <section className="panel">
          <div className="panel-heading">
            <h2>Active indexes</h2>
            <Database size={18} />
          </div>
          {indexes.loading ? (
            <Loading />
          ) : indexes.error ? (
            <Empty>Active indexes could not be loaded.</Empty>
          ) : !indexes.data?.length ? (
            <Empty>No active indexes. Create one to begin.</Empty>
          ) : (
            indexes.data.map((i) => (
              <div
                className={`index-card ${selected === i.id ? "selected" : ""}`}
                key={i.id}
              >
                <button
                  className="index-select"
                  aria-pressed={selected === i.id}
                  onClick={() => setSelected(i.id)}
                >
                  <strong>{i.name}</strong>
                  <Badge tone={i.is_trained ? "success" : "warning"}>
                    {i.type} / {i.status}
                  </Badge>
                </button>
                <dl
                  className="index-metadata"
                  aria-label={`${i.name} properties`}
                >
                  <div>
                    <dt>Type</dt>
                    <dd>{i.type.toUpperCase()}</dd>
                  </div>
                  <div>
                    <dt>Metric</dt>
                    <dd>{i.metric}</dd>
                  </div>
                  <div>
                    <dt>Dimensions</dt>
                    <dd>{i.dimension}</dd>
                  </div>
                  <div>
                    <dt>Vectors</dt>
                    <dd>{i.size.toLocaleString()}</dd>
                  </div>
                </dl>
                <div className="card-actions">
                  <Link className="text-button" to={`/workspace?index=${i.id}`}>
                    <ArrowUpRight size={15} />
                    Workspace
                  </Link>
                  <button
                    className="text-button"
                    disabled={action.busy || !i.is_trained}
                    onClick={() =>
                      void action.run(async () => {
                        await request(`/indexes/${i.id}/save`, "POST");
                        saved.reload();
                      }, "Archive saved.")
                    }
                  >
                    <Save size={15} />
                    Save
                  </button>
                  <button
                    className="text-button danger"
                    disabled={action.busy}
                    onClick={() => {
                      if (
                        confirm(
                          `Delete active index “${i.name}”? Saved archives remain available.`,
                        )
                      )
                        void action.run(async () => {
                          await request(`/indexes/${i.id}`, "DELETE");
                          refresh();
                        }, "Index deleted.");
                    }}
                    aria-label={`Delete ${i.name}`}
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              </div>
            ))
          )}
        </section>
      </div>
      {current && (
        <section className="panel detail">
          <div className="panel-heading">
            <h2>{current.name} / details</h2>
            <Badge tone={current.is_trained ? "success" : "warning"}>
              {current.status}
            </Badge>
          </div>
          <p className="mono muted">{current.id}</p>
          <pre className="config">
            {JSON.stringify(current.configuration, null, 2)}
          </pre>
          {current.type === "ivf" && (
            <form
              className="form"
              onSubmit={(e) => {
                e.preventDefault();
                void action.run(async () => {
                  await request(`/indexes/${current.id}/train`, "POST", {
                    vectors: JSON.parse(training),
                  });
                  indexes.reload();
                }, "Training completed; call insert to index your vectors.");
              }}
            >
              <Field label="Training vectors (JSON matrix)">
                <textarea
                  aria-label="Training vectors"
                  rows={5}
                  value={training}
                  onChange={(e) => setTraining(e.target.value)}
                  placeholder="[[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]]"
                  spellCheck={false}
                  required
                />
              </Field>
              <div className="button-row">
                <button
                  type="button"
                  className="button"
                  onClick={() => {
                    const count = current.configuration.nlist ?? 4;
                    setTraining(
                      JSON.stringify(
                        Array.from({ length: Math.max(8, count * 2) }, (_, i) =>
                          Array.from({ length: current.dimension }, (_, d) =>
                            d === 0
                              ? Number(
                                  Math.cos(
                                    (i * 2 * Math.PI) / Math.max(8, count * 2),
                                  ).toFixed(4),
                                )
                              : d === 1
                                ? Number(
                                    Math.sin(
                                      (i * 2 * Math.PI) /
                                        Math.max(8, count * 2),
                                    ).toFixed(4),
                                  )
                                : 0,
                          ),
                        ),
                      ),
                    );
                  }}
                >
                  Fill example only
                </button>
                <button
                  className="button primary"
                  disabled={action.busy || current.size > 0}
                >
                  Train IVF index
                </button>
              </div>
              <p className="hint">
                Training learns centroids without inserting samples. Delete all
                vectors before retraining.
              </p>
            </form>
          )}
        </section>
      )}
      <section className="panel detail">
        <div className="panel-heading">
          <h2>Saved archives</h2>
          <span className="muted">Persistent storage</span>
        </div>
        <p className="hint">
          Loading creates a new active index. Archives survive application
          restarts.
        </p>
        {saved.loading ? (
          <Loading />
        ) : saved.error ? (
          <Empty>Saved archives could not be loaded.</Empty>
        ) : !saved.data?.length ? (
          <Empty>No saved archives yet.</Empty>
        ) : (
          saved.data.map((s) => (
            <div className="list-row" key={s.archive_id}>
              <Save size={18} />
              <div>
                <strong>{s.name}</strong>
                <small>
                  {s.type} · {(s.size_bytes / 1024).toFixed(1)} KiB ·{" "}
                  {s.archive_id}
                </small>
              </div>
              <button
                className="button"
                disabled={action.busy}
                onClick={() =>
                  void action.run(async () => {
                    const i = await request<IndexInfo>(
                      "/indexes/load",
                      "POST",
                      { archive_id: s.archive_id },
                    );
                    setSelected(i.id);
                    indexes.reload();
                  }, "Archive restored.")
                }
              >
                Load index
              </button>
            </div>
          ))
        )}
      </section>
    </>
  );
}
