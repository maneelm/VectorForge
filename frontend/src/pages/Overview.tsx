import { Link } from "react-router-dom";
import { ArrowRight, Database, GitBranch } from "lucide-react";
import { useResource } from "../hooks/useResource";
import type { IndexInfo, RunList } from "../types";
import { Badge, Empty, ErrorBox, Loading } from "../components/UI";
export function Overview() {
  const indexes = useResource<IndexInfo[]>("/indexes"),
    runs = useResource<RunList>("/benchmarks/runs");
  const data = indexes.data;
  return (
    <>
      <div className="page-title">
        <div>
          <h1>Overview</h1>
          <p>Active indexes and measured benchmark reports.</p>
        </div>
        <Link className="button primary" to="/indexes">
          Create an index <ArrowRight size={16} />
        </Link>
      </div>
      <ErrorBox message={indexes.error || runs.error} />
      {indexes.loading ? (
        <Loading />
      ) : (
        data && (
          <div className="stats">
            {[
              ["Active indexes", data.length],
              [
                "Stored vectors",
                data.reduce((s, i) => s + i.size, 0).toLocaleString(),
              ],
              [
                "Exact / IVF indexes",
                `${data.filter((i) => i.type === "exact").length} / ${data.filter((i) => i.type === "ivf").length}`,
              ],
              [
                "Benchmark reports",
                runs.data
                  ? runs.data.runs.length
                  : runs.error
                    ? "Unavailable"
                    : "Loading...",
              ],
            ].map(([label, value]) => (
              <div className="stat" key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>
        )
      )}
      <div className="grid-two">
        <section className="panel">
          <div className="panel-heading">
            <h2>Active indexes</h2>
            <Link to="/indexes">
              Manage <ArrowRight size={14} />
            </Link>
          </div>
          {indexes.loading ? (
            <Loading />
          ) : indexes.error ? (
            <Empty>Active indexes could not be loaded.</Empty>
          ) : !data?.length ? (
            <Empty>
              <Database size={28} />
              <h3>No active indexes</h3>
              <p>
                Create an Exact index for a baseline, or train IVF to explore
                approximate search.
              </p>
              <Link className="button" to="/indexes">
                Open index manager
              </Link>
            </Empty>
          ) : (
            <div className="list">
              {data.map((i) => (
                <Link
                  to={`/workspace?index=${i.id}`}
                  className="list-row"
                  key={i.id}
                >
                  <div className="list-icon">
                    <Database size={18} />
                  </div>
                  <div>
                    <strong>{i.name}</strong>
                    <small>
                      {i.dimension} dimensions · {i.metric} · {i.size} vectors
                    </small>
                  </div>
                  <Badge good={i.is_trained}>{i.type.toUpperCase()}</Badge>
                  <ArrowRight size={16} />
                </Link>
              ))}
            </div>
          )}
        </section>
        <section className="panel">
          <div className="panel-heading">
            <h2>Recent benchmark runs</h2>
            <Link to="/runner">
              Run benchmark <ArrowRight size={14} />
            </Link>
          </div>
          {runs.loading ? (
            <Loading />
          ) : runs.error ? (
            <Empty>Benchmark reports could not be loaded.</Empty>
          ) : !runs.data?.runs.length ? (
            <Empty>
              No saved runs yet. Start a bounded benchmark from the runner.
            </Empty>
          ) : (
            <div className="list">
              {[...runs.data.runs]
                .sort(
                  (a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp),
                )
                .slice(0, 4)
                .map((r) => (
                  <Link
                    className="list-row"
                    to={`/benchmarks?run=${r.id}`}
                    key={r.id}
                  >
                    <div className="list-icon">
                      <GitBranch size={18} />
                    </div>
                    <div>
                      <strong>
                        {r.size.toLocaleString()} vectors / {r.dimension}d
                      </strong>
                      <small>
                        {r.metric} · {r.source} ·{" "}
                        {new Date(r.timestamp).toLocaleDateString()}
                      </small>
                    </div>
                    <Badge good={r.complete}>
                      {r.complete ? "Complete" : "Has failures"}
                    </Badge>
                  </Link>
                ))}
            </div>
          )}
        </section>
      </div>
      <p className="workspace-note">
        Active indexes are held in memory. Save an archive in the index manager
        before restarting.
      </p>
    </>
  );
}
