# Published benchmark results

This directory contains representative Phase 3 results included in Git.
Locally generated benchmark outputs remain in the ignored `results/` directory.

The published run used 10,000 indexed vectors, 128 dimensions, cosine similarity,
k=10, nlist=16, 50 queries, three repetitions, and one thread. All four engines
completed successfully. Files were copied unchanged from the existing run
`results/quick/benchmark_20261025T025027Z_4e5164f0`; no benchmark was rerun.
Metadata retains the original environment and output-directory provenance.

- [CSV summary](benchmark_20261025T025027Z_4e5164f0/summary.csv)
- [JSON summary](benchmark_20261025T025027Z_4e5164f0/summary.json)
- [Environment and configuration metadata](benchmark_20261025T025027Z_4e5164f0/metadata.json)
- [Raw latency samples](benchmark_20261025T025027Z_4e5164f0/latency_samples.csv)

The raw samples are approximately 52 KB and allow the reported latency statistics
to be independently recomputed. All published artifacts total less than 0.5 MiB.

The six charts are:

- [Search latency](benchmark_20261025T025027Z_4e5164f0/charts/search_latency.png)
- [Throughput](benchmark_20261025T025027Z_4e5164f0/charts/throughput.png)
- [Recall versus nprobe](benchmark_20261025T025027Z_4e5164f0/charts/recall_vs_nprobe.png)
- [Recall versus latency](benchmark_20261025T025027Z_4e5164f0/charts/recall_vs_latency.png)
- [Index build time](benchmark_20261025T025027Z_4e5164f0/charts/build_time.png)
- [Process RSS memory](benchmark_20261025T025027Z_4e5164f0/charts/memory.png)

See the [measured tables](../phase3_results.md) and
[benchmark methodology](../benchmarking.md) for interpretation and limitations.
