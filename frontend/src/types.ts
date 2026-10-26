export type Metric = "cosine" | "euclidean" | "dot";
export interface IndexInfo {
  id: string;
  name: string;
  type: "exact" | "ivf";
  dimension: number;
  metric: Metric;
  size: number;
  is_trained: boolean;
  status: string;
  configuration: {
    nlist?: number;
    nprobe?: number;
    random_state?: number;
    max_iter?: number;
    tol?: number;
  };
}
export interface SavedIndex {
  archive_id: string;
  name: string;
  type: string;
  size_bytes: number;
}
export interface SearchResult {
  matches: { id: string; score: number; rank: number }[];
  latency_ms: number;
  latency_scope: string;
  index_type: string;
  metric: Metric;
  nprobe: number | null;
}
export interface Run {
  id: string;
  name: string;
  source: string;
  timestamp: string;
  complete: boolean;
  size: number;
  dimension: number;
  metric: Metric;
  distribution: string;
  result_count: number;
}
export interface RunList {
  runs: Run[];
  issues: { name: string; error: string }[];
}
export interface Result {
  engine: string;
  status: "ok" | "failed";
  error: string | null;
  nprobe: number | null;
  nlist: number | null;
  precision: string;
  full_probe_direct_path: boolean;
  latency_mean_ms: number | null;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  latency_p99_ms: number | null;
  qps: number | null;
  recall_at_k: number | null;
  construction_ms: number | null;
  training_ms: number | null;
  insertion_ms: number | null;
  build_total_ms: number | null;
  rss_baseline_bytes: number | null;
  rss_after_insertion_bytes: number | null;
  retained_rss_delta_bytes: number | null;
}
export interface RunResults {
  run: Run;
  results: Result[];
  metadata: Record<string, unknown>;
}
export interface Job {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  created_at: string;
  error: string | null;
  log: string;
  run_ids: string[];
}
export interface Health {
  status: string;
  version: string;
  can_run_benchmarks: boolean;
  missing_dependencies: string[];
  benchmark_preset: {
    size: number;
    dimension: number;
    query_count: number;
    nlist: number;
    nprobes: number[];
    repetitions: number;
    threads: number;
  };
}
