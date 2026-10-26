import { Link } from "react-router-dom";
import { ArrowRight, Database, GitBranch, Search } from "lucide-react";
import { useResource } from "../hooks/useResource";
import type { IndexInfo, RunList } from "../types";
import { Badge, Empty, ErrorBox, Loading } from "../components/UI";
export function Overview() {
  const indexes = useResource<IndexInfo[]>("/indexes"),
    runs = useResource<RunList>("/benchmarks/runs");
  const data = indexes.data;
  return (
    <>
      <div className="eyebrow">SEARCH INFRASTRUCTURE / 01</div>
      <div className="page-title">
        <div>
          <h1>Your search laboratory.</h1>
          <p>
            Build an index, explore its neighbors, and put performance to the
            test.
          </p>
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
              ["Exact indexes", data.filter((i) => i.type === "exact").length],
              ["IVF indexes", data.filter((i) => i.type === "ivf").length],
            ].map(([label, value]) => (
              <div className="stat" key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
                <small>LIVE WORKSPACE</small>
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
          {!data?.length ? (
            <Empty>
              <Database size={28} />
              <h3>Start with your first index</h3>
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
            <Badge>MEASURED DATA</Badge>
          </div>
          {runs.loading ? (
            <Loading />
          ) : !runs.data?.runs.length ? (
            <Empty>
              No saved runs yet. Start a bounded benchmark from the runner.
            </Empty>
          ) : (
            <div className="list">
              {runs.data.runs.slice(0, 4).map((r) => (
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
      <section className="callout">
        <Search size={26} />
        <div>
          <h2>Find what’s nearby.</h2>
          <p>
            Search your own vectors with cosine, Euclidean distance, or dot
            product.
          </p>
        </div>
        <Link className="button" to="/workspace">
          Open vector workspace <ArrowRight size={16} />
        </Link>
      </section>
    </>
  );
}
