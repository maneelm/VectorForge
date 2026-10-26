import { BrowserRouter, NavLink, Route, Routes, Link } from "react-router-dom";
import {
  Activity,
  ArrowUpRight,
  BarChart3,
  Boxes,
  Database,
  FlaskConical,
  Layers3,
  Terminal,
} from "lucide-react";
import { useResource } from "./hooks/useResource";
import type { Health } from "./types";
import { lazy, Suspense } from "react";
import { Loading } from "./components/UI";
import { Overview } from "./pages/Overview";
import { Indexes } from "./pages/Indexes";
import { Workspace } from "./pages/Workspace";
const Benchmarks = lazy(() =>
  import("./pages/Benchmarks").then((module) => ({
    default: module.Benchmarks,
  })),
);
import { Runner } from "./pages/Runner";
const nav = [
  ["/", "Overview", Activity],
  ["/indexes", "Index manager", Database],
  ["/workspace", "Vector workspace", Boxes],
  ["/benchmarks", "Benchmarks", BarChart3],
  ["/runner", "Benchmark runner", FlaskConical],
] as const;
export function Shell() {
  const health = useResource<Health>("/health");
  return (
    <div className="app">
      <aside>
        <Link to="/" className="brand">
          <span className="brand-mark">
            <Layers3 size={25} />
          </span>
          VectorForge<span className="version">LAB</span>
        </Link>
        <div className="nav-label">WORKSPACE</div>
        <nav>
          {nav.map(([to, label, Icon]) => (
            <NavLink to={to} end={to === "/"} key={to}>
              <Icon size={18} />
              {label}
              <span className="nav-arrow">↗</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="status-dot" />
          Local engine
          <span className="mono">{health.data ? "CONNECTED" : "OFFLINE"}</span>
          <p>Exact algorithms. Measurable tradeoffs.</p>
          <Link to="/benchmarks">
            Explore the evidence <ArrowUpRight size={14} />
          </Link>
        </div>
      </aside>
      <div className="main">
        <header>
          <span>
            <Terminal size={16} /> VectorForge <span className="muted">/</span>{" "}
            search laboratory
          </span>
          <span className="header-tag">
            LOCAL WORKSPACE <i className={health.data ? "online" : ""} />
          </span>
        </header>
        <main>
          <Suspense fallback={<Loading />}>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/indexes" element={<Indexes />} />
              <Route path="/workspace" element={<Workspace />} />
              <Route path="/benchmarks" element={<Benchmarks />} />
              <Route path="/runner" element={<Runner />} />
              <Route
                path="*"
                element={
                  <div>
                    <h1>Page not found</h1>
                    <Link to="/">Return to overview</Link>
                  </div>
                }
              />
            </Routes>
          </Suspense>
        </main>
        <footer>
          VECTORFORGE{" "}
          <span>Built from first principles. Evaluated with evidence.</span>
        </footer>
      </div>
    </div>
  );
}
export default function App() {
  return (
    <BrowserRouter>
      <Shell />
    </BrowserRouter>
  );
}
