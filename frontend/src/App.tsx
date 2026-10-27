import {
  BrowserRouter,
  NavLink,
  Route,
  Routes,
  Link,
  useLocation,
} from "react-router-dom";
import {
  Activity,
  BarChart3,
  Boxes,
  Database,
  FlaskConical,
  Layers3,
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
  const location = useLocation();
  const pageName =
    nav.find(([path]) => path === location.pathname)?.[1] ?? "Page not found";
  const connection = health.data
    ? "API connected"
    : health.error
      ? "API unavailable"
      : "Connecting to API";
  return (
    <div className="app">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <aside>
        <Link to="/" className="brand">
          <span className="brand-mark">
            <Layers3 size={25} />
          </span>
          VectorForge
        </Link>
        <nav aria-label="Main navigation">
          {nav.map(([to, label, Icon]) => (
            <div key={to}>
              {(to === "/" || to === "/benchmarks") && (
                <div className="nav-label">
                  {to === "/" ? "Workspace" : "Performance"}
                </div>
              )}
              <NavLink to={to} end={to === "/"}>
                <Icon size={17} aria-hidden="true" />
                {label}
              </NavLink>
            </div>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <span
            className={`connection ${health.data ? "online" : health.error ? "offline" : ""}`}
          >
            <span className="status-dot" />
            {connection}
          </span>
          {health.data && <span className="muted">v{health.data.version}</span>}
        </div>
      </aside>
      <div className="main">
        <header>
          <span>
            <span className="muted">VectorForge</span>
            <span className="muted">/</span>
            {pageName}
          </span>
          {location.pathname !== "/workspace" && (
            <Link className="text-button" to="/workspace">
              <Boxes size={14} />
              Vector workspace
            </Link>
          )}
        </header>
        <main id="main-content" tabIndex={-1} key={location.pathname}>
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
