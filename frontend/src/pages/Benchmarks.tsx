import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useResource } from "../hooks/useResource";
import type { Result, RunList, RunResults } from "../types";
import {
  Badge,
  Empty,
  ErrorBox,
  Field,
  Loading,
  engineName,
} from "../components/UI";
const colors = ["#a9e787", "#70b6ff", "#d2a4f4", "#f6c775"];
const tick = { fill: "#94a3b8", fontSize: 10 };
const tooltipStyle = {
  background: "#161e27",
  border: "1px solid #354354",
  borderRadius: 8,
  color: "#f3f6fa",
};
function Chart({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="panel chart-panel">
      <h2>{title}</h2>
      <div className="chart">{children}</div>
    </section>
  );
}
export function BenchmarkCharts({ rows }: { rows: Result[] }) {
  const good = rows.filter((r) => r.status === "ok"),
    data = good.map((r) => ({
      ...r,
      label:
        engineName(r.engine) +
        (r.nprobe
          ? ` / ${r.nprobe}${r.full_probe_direct_path ? "*" : ""}`
          : ""),
    })),
    build = Object.values(
      Object.fromEntries(
        data.map((r) => [
          r.engine,
          {
            ...r,
            label: engineName(r.engine),
            baseline: (r.rss_baseline_bytes ?? 0) / 2 ** 20,
            retained:
              r.retained_rss_delta_bytes === null
                ? null
                : r.retained_rss_delta_bytes / 2 ** 20,
          },
        ]),
      ),
    );
  const engines = [...new Set(good.map((r) => r.engine))];
  const bars = (field: string, label: string, source = data) => (
    <ResponsiveContainer width="100%" height="100%" minWidth={0}>
      <BarChart
        data={source}
        margin={{ top: 8, right: 12, bottom: 35, left: 0 }}
      >
        <CartesianGrid stroke="#26313e" vertical={false} />
        <XAxis
          dataKey="label"
          tick={tick}
          interval={0}
          angle={-24}
          textAnchor="end"
          height={65}
        />
        <YAxis tick={tick} />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend />
        <Bar
          dataKey={field}
          name={label}
          fill="#a9e787"
          radius={[4, 4, 0, 0]}
        />
      </BarChart>
    </ResponsiveContainer>
  );
  return (
    <div className="charts-grid">
      <Chart title="Search latency · mean ms">
        {bars("latency_mean_ms", "Mean latency (ms)")}
      </Chart>
      <Chart title="Throughput · queries / second">
        {bars("qps", "Measured QPS")}
      </Chart>
      <Chart title="Recall versus latency">
        <ResponsiveContainer width="100%" height="100%" minWidth={0}>
          <ScatterChart margin={{ bottom: 20, left: 5, right: 20 }}>
            <CartesianGrid stroke="#26313e" />
            <XAxis
              type="number"
              dataKey="latency_mean_ms"
              tick={tick}
              name="Mean latency"
              unit=" ms"
            />
            <YAxis
              type="number"
              dataKey="recall_at_k"
              domain={[0, 1]}
              tick={tick}
              name="Recall@k"
            />
            <Tooltip
              contentStyle={tooltipStyle}
              cursor={{ strokeDasharray: "3 3" }}
            />
            <Legend />
            {engines.map((e, i) => (
              <Scatter
                key={e}
                name={engineName(e)}
                data={data.filter((r) => r.engine === e)}
                fill={colors[i % 4]}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </Chart>
      <Chart title="Recall versus nprobe">
        {data.some((row) => row.nprobe !== null) ? (
          <ResponsiveContainer width="100%" height="100%" minWidth={0}>
            <LineChart margin={{ bottom: 20, left: 5, right: 20 }}>
              <CartesianGrid stroke="#26313e" />
              <XAxis
                type="number"
                dataKey="nprobe"
                allowDuplicatedCategory={false}
                tick={tick}
                name="nprobe"
              />
              <YAxis domain={[0, 1]} tick={tick} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend />
              {engines
                .filter((e) => e.endsWith("ivf"))
                .map((e, i) => (
                  <Line
                    key={e}
                    data={data
                      .filter((r) => r.engine === e)
                      .sort((a, b) => a.nprobe! - b.nprobe!)}
                    dataKey="recall_at_k"
                    name={engineName(e)}
                    stroke={colors[i]}
                    strokeWidth={2}
                    connectNulls={false}
                  />
                ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <Empty>Probe comparisons apply to IVF indexes.</Empty>
        )}
      </Chart>
      <Chart title="Index build · milliseconds">
        <ResponsiveContainer width="100%" height="100%" minWidth={0}>
          <BarChart data={build} margin={{ bottom: 30 }}>
            <CartesianGrid stroke="#26313e" vertical={false} />
            <XAxis
              dataKey="label"
              tick={tick}
              angle={-20}
              textAnchor="end"
              height={60}
            />
            <YAxis tick={tick} />
            <Tooltip contentStyle={tooltipStyle} />
            <Legend />
            {[
              ["construction_ms", "Construction"],
              ["training_ms", "IVF training"],
              ["insertion_ms", "Insertion"],
            ].map(([field, label], i) => (
              <Bar
                key={field}
                dataKey={field}
                stackId="build"
                name={label}
                fill={colors[i]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </Chart>
      <Chart title="Memory · process RSS MiB">
        <ResponsiveContainer width="100%" height="100%" minWidth={0}>
          <BarChart data={build} margin={{ bottom: 30 }}>
            <CartesianGrid stroke="#26313e" vertical={false} />
            <XAxis
              dataKey="label"
              tick={tick}
              angle={-20}
              textAnchor="end"
              height={60}
            />
            <YAxis tick={tick} />
            <Tooltip contentStyle={tooltipStyle} />
            <Legend />
            <Bar dataKey="baseline" name="Baseline RSS" fill="#70b6ff" />
            <Bar
              dataKey="retained"
              name="Retained RSS delta (estimate)"
              fill="#a9e787"
            />
          </BarChart>
        </ResponsiveContainer>
      </Chart>
    </div>
  );
}
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
      <div className="eyebrow">EVIDENCE, NOT ASSUMPTIONS / 04</div>
      <div className="page-title">
        <div>
          <h1>Benchmark dashboard</h1>
          <p>
            Compare retrieval quality and cost on one measured workload at a
            time.
          </p>
        </div>
        <Badge good>PHASE 3 REPORTS</Badge>
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
              {details.data.run.size.toLocaleString()} VECTORS /{" "}
              {details.data.run.dimension} DIMENSIONS /{" "}
              {details.data.run.metric.toUpperCase()}
            </span>
            <Badge good={details.data.run.complete}>
              {details.data.run.source} ·{" "}
              {details.data.run.complete ? "Complete" : "Contains failures"}
            </Badge>
          </div>
          {rows.some((r) => r.status === "ok") ? (
            <BenchmarkCharts rows={rows} />
          ) : (
            <Empty>No successful measurements match the filters.</Empty>
          )}
          <section className="panel detail">
            <div className="panel-heading">
              <h2>Measured results</h2>
              <Badge>{rows.length} CONFIGURATIONS</Badge>
            </div>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Engine / probes</th>
                    <th>Mean ms</th>
                    <th>p50</th>
                    <th>p95</th>
                    <th>p99</th>
                    <th>QPS</th>
                    <th>Recall@k</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.engine + ":" + r.nprobe}>
                      <td>
                        {engineName(r.engine)}{" "}
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
                        <td key={i} className={i === 5 ? "score" : ""}>
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
