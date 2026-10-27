import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useResource } from "../hooks/useResource";
import type { RunList, RunResults } from "../types";
import {
  Badge,
  Empty,
  ErrorBox,
  Field,
  Loading,
  engineName,
} from "../components/UI";
import { BenchmarkCharts } from "../components/BenchmarkCharts";
import { engineColor } from "../components/chartTheme";
import type { CSSProperties } from "react";
export function Benchmarks() {
  const runs = useResource<RunList>("/benchmarks/runs"),
    [params, setParams] = useSearchParams(),
    [size, setSize] = useState(""),
    [dimension, setDimension] = useState(""),
    [metric, setMetric] = useState(""),
    [engine, setEngine] = useState(""),
    [probe, setProbe] = useState("");
  const filtered =
    runs.data?.runs.filter(
      (r) =>
        (!size || r.size === Number(size)) &&
        (!dimension || r.dimension === Number(dimension)) &&
        (!metric || r.metric === metric),
    ) ?? [];
  const id = params.get("run") || filtered[0]?.id || "";
  const details = useResource<RunResults>(
    id && filtered.some((r) => r.id === id)
      ? `/benchmarks/runs/${id}/results`
      : null,
  );
  useEffect(() => {
    setEngine("");
    setProbe("");
  }, [id]);
  const rows =
    details.data?.results.filter(
      (r) =>
        (!engine || r.engine === engine) &&
        (!probe ||
          (probe === "none" ? r.nprobe === null : r.nprobe === Number(probe))),
    ) ?? [];
  const filterChange = (set: (s: string) => void, value: string) => {
    set(value);
    setParams({});
  };
  return (
    <>
      <div className="page-title">
        <div>
          <h1>Benchmark dashboard</h1>
          <p>
            Compare latency, throughput, and retrieval quality for a measured
            workload.
          </p>
        </div>
      </div>
      <ErrorBox message={runs.error || details.error} />
      {!!runs.data?.issues.length && (
        <ErrorBox
          message={`${runs.data.issues.length} invalid reports excluded: ${runs.data.issues[0].error}`}
        />
      )}
      <section className="panel filters">
        <Field label="Dataset size">
          <select
            value={size}
            onChange={(e) => filterChange(setSize, e.target.value)}
          >
            <option value="">All sizes</option>
            {[...new Set(runs.data?.runs.map((r) => r.size))].map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </Field>
        <Field label="Dimensions">
          <select
            value={dimension}
            onChange={(e) => filterChange(setDimension, e.target.value)}
          >
            <option value="">All dimensions</option>
            {[...new Set(runs.data?.runs.map((r) => r.dimension))].map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </Field>
        <Field label="Metric">
          <select
            value={metric}
            onChange={(e) => filterChange(setMetric, e.target.value)}
          >
            <option value="">All metrics</option>
            <option>cosine</option>
            <option>euclidean</option>
            <option>dot</option>
          </select>
        </Field>
        <Field label="Benchmark run">
          <select
            value={id}
            onChange={(e) => setParams({ run: e.target.value })}
          >
            <option value="">Select a run</option>
            {filtered.map((r) => (
              <option value={r.id} key={r.id}>
                {r.source} · {r.size} × {r.dimension} · {r.name.slice(-8)}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Engine">
          <select
            aria-label="Engine"
            value={engine}
            onChange={(e) => setEngine(e.target.value)}
          >
            <option value="">All engines</option>
            {[...new Set(details.data?.results.map((r) => r.engine))].map(
              (e) => (
                <option value={e} key={e}>
                  {engineName(e)}
                </option>
              ),
            )}
          </select>
        </Field>
        <Field label="nprobe">
          <select value={probe} onChange={(e) => setProbe(e.target.value)}>
            <option value="">All probes</option>
            <option value="none">Not applicable</option>
            {[
              ...new Set(
                details.data?.results
                  .map((r) => r.nprobe)
                  .filter((p) => p !== null),
              ),
            ]
              .sort((a, b) => a! - b!)
              .map((p) => (
                <option key={p!}>{p}</option>
              ))}
          </select>
        </Field>
      </section>
      {runs.loading || details.loading ? (
        <Loading />
      ) : !details.data ? (
        <Empty>
          No matching report selected. Published reports and completed local
          runs appear here.
        </Empty>
      ) : (
        <>
          <div className="report-heading">
            <span className="mono">
              {details.data.run.size.toLocaleString()} vectors /{" "}
              {details.data.run.dimension} dimensions /{" "}
              {details.data.run.metric}
            </span>
            <Badge good={details.data.run.complete}>
              {details.data.run.source} ·{" "}
              {details.data.run.complete ? "Complete" : "Contains failures"}
            </Badge>
          </div>
          <dl className="report-config" aria-label="Report configuration">
            <div>
              <dt>Distribution</dt>
              <dd>{details.data.run.distribution}</dd>
            </div>
            <div>
              <dt>Recorded</dt>
              <dd>{new Date(details.data.run.timestamp).toLocaleString()}</dd>
            </div>
            <div>
              <dt>Configurations shown</dt>
              <dd>
                {rows.length} / {details.data.results.length}
              </dd>
            </div>
            <div>
              <dt>Report</dt>
              <dd className="mono">{details.data.run.name}</dd>
            </div>
          </dl>
          {rows.some((r) => r.status === "ok") ? (
            <BenchmarkCharts rows={rows} />
          ) : (
            <Empty>No successful measurements match the filters.</Empty>
          )}
          <section className="panel detail">
            <div className="panel-heading">
              <h2>Measured results</h2>
              <Badge>{rows.length} configurations</Badge>
            </div>
            <div
              className="table-scroll"
              role="region"
              aria-label="Benchmark results table"
              tabIndex={0}
            >
              <table aria-label="Measured benchmark results">
                <thead>
                  <tr>
                    <th>Engine / probes</th>
                    <th className="numeric">Mean (ms)</th>
                    <th className="numeric">p50 (ms)</th>
                    <th className="numeric">p95 (ms)</th>
                    <th className="numeric">p99 (ms)</th>
                    <th className="numeric">QPS</th>
                    <th className="numeric">Recall@k</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.engine + ":" + r.nprobe}>
                      <td>
                        <span
                          className="engine-label"
                          style={
                            {
                              "--engine-color": engineColor(r.engine),
                            } as CSSProperties
                          }
                        >
                          <i aria-hidden="true" />
                          {engineName(r.engine)}
                        </span>{" "}
                        {r.nprobe !== null &&
                          `/ ${r.nprobe}${r.full_probe_direct_path ? "*" : ""}`}
                        <small>{r.precision}</small>
                        {r.error && <span className="failure">{r.error}</span>}
                      </td>
                      {[
                        r.latency_mean_ms,
                        r.latency_p50_ms,
                        r.latency_p95_ms,
                        r.latency_p99_ms,
                        r.qps,
                        r.recall_at_k,
                      ].map((v, i) => (
                        <td key={i} className="numeric">
                          {v === null ? "N/A" : v.toFixed(i === 4 ? 1 : 3)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <section className="explanations">
            <h2>Read the measurements</h2>
            <div className="grid-two">
              <p>
                <strong>Exact versus approximate.</strong> Exact scores every
                vector. IVF searches selected clusters; lower nprobe can reduce
                work and recall. Recall@k measures overlap with canonical Exact
                top-k IDs.
              </p>
              <p>
                <strong>Precision and scope.</strong> VectorForge uses float64;
                FAISS uses float32. These are single-query adapter measurements.
                QPS comes from timed workload throughput, not inverse mean
                latency.
              </p>
              <p>
                <strong>Full probing.</strong> * VectorForge IVF uses direct
                full-dataset ranking when nprobe=nlist. This is not ordinary
                list-scanning performance.
              </p>
              <p>
                <strong>Memory is an estimate.</strong> RSS includes native
                allocator effects and retained temporary memory. The delta is
                not peak memory or an exact index size; null means not
                applicable.
              </p>
            </div>
          </section>
        </>
      )}
    </>
  );
}
