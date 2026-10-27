import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { FlaskConical, Play, ArrowUpRight } from "lucide-react";
import { request } from "../api/client";
import { useResource } from "../hooks/useResource";
import type { Health, Job, Metric } from "../types";
import {
  Badge,
  Empty,
  ErrorBox,
  Feedback,
  Field,
  useAction,
} from "../components/UI";
export function Runner() {
  const health = useResource<Health>("/health"),
    jobs = useResource<Job[]>("/benchmarks/jobs"),
    action = useAction();
  const [metric, setMetric] = useState<Metric>("cosine"),
    [seed, setSeed] = useState(42);
  useEffect(() => {
    const timer = setInterval(jobs.reload, 1500);
    return () => clearInterval(timer);
  }, []);
  const preset = health.data?.benchmark_preset;
  return (
    <>
      <div className="eyebrow">REPRODUCIBLE EXPERIMENTS / 05</div>
      <div className="page-title">
        <div>
          <h1>Benchmark runner</h1>
          <p>
            A bounded experiment. Four engines. The same vectors and queries.
          </p>
        </div>
        <Badge>QUICK WORKLOAD</Badge>
      </div>
      <ErrorBox message={health.error || jobs.error} />
      <Feedback action={action} />
      <div className="grid-two">
        <section className="panel">
          <div className="panel-heading">
            <h2>Quick comparison</h2>
            <FlaskConical size={19} />
          </div>
          <div className="form">
            <p>
              Runs the existing Phase 3 CLI in an isolated process. Reports
              remain available after restart.
            </p>
            {preset && (
              <dl className="preset-grid">
                {Object.entries(preset).map(([key, value]) => (
                  <div key={key}>
                    <dt>{key.replaceAll("_", " ")}</dt>
                    <dd>
                      {Array.isArray(value)
                        ? value.join(" / ")
                        : value.toLocaleString()}
                    </dd>
                  </div>
                ))}
              </dl>
            )}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void action.run(async () => {
                  await request<Job>("/benchmarks/runs", "POST", {
                    preset: "quick",
                    metric,
                    seed,
                  });
                  jobs.reload();
                }, "Benchmark queued. Status updates appear below.");
              }}
            >
              <div className="form-row">
                <Field label="Benchmark metric">
                  <select
                    value={metric}
                    onChange={(e) => setMetric(e.target.value as Metric)}
                  >
                    <option>cosine</option>
                    <option>euclidean</option>
                    <option>dot</option>
                  </select>
                </Field>
                <Field label="Dataset seed">
                  <input
                    type="number"
                    min={0}
                    max={2147483647}
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value))}
                    required
                  />
                </Field>
              </div>
              <button
                className="button primary"
                disabled={action.busy || !health.data?.can_run_benchmarks}
              >
                <Play size={16} />
                Start quick benchmark
              </button>
            </form>
            {health.data && !health.data.can_run_benchmarks && (
              <ErrorBox
                message={
                  health.data.benchmark_enabled === false
                    ? "Benchmark execution is disabled by the server."
                    : `Missing optional dependencies: ${health.data.missing_dependencies.join(", ")}. Install with: uv sync --locked --extra web --extra benchmark`
                }
              />
            )}
            <p className="hint">
              {health.data?.benchmark_limits && (
                <>
                  At most {health.data.benchmark_limits.max_pending_jobs}{" "}
                  running/queued jobs in total; one runs at a time. Each job has
                  a {health.data.benchmark_limits.job_timeout_seconds}-second
                  deadline.{" "}
                </>
              )}
              Avoid competing CPU work while measuring.
            </p>
          </div>
        </section>
        <section className="panel">
          <div className="panel-heading">
            <h2>Job activity</h2>
            <Badge>LIVE STATUS</Badge>
          </div>
          {!jobs.data?.length ? (
            <Empty>
              <FlaskConical size={27} />
              <h3>No experiments yet</h3>
              <p>Start a quick benchmark, or explore the published reports.</p>
              <Link className="button" to="/benchmarks">
                View saved benchmarks
              </Link>
            </Empty>
          ) : (
            jobs.data.map((j) => (
              <article className="job" key={j.id}>
                <div className="panel-heading">
                  <strong>{new Date(j.created_at).toLocaleTimeString()}</strong>
                  <Badge good={j.status === "completed"}>{j.status}</Badge>
                </div>
                <small className="mono">{j.id}</small>
                {j.error && <ErrorBox message={j.error} />}
                <div className="button-row">
                  {j.run_ids.map((id) => (
                    <Link
                      className="button"
                      to={`/benchmarks?run=${id}`}
                      key={id}
                    >
                      View results <ArrowUpRight size={15} />
                    </Link>
                  ))}
                </div>
                {j.log && (
                  <details>
                    <summary>Execution log</summary>
                    <pre>{j.log}</pre>
                  </details>
                )}
              </article>
            ))
          )}
        </section>
      </div>
    </>
  );
}
