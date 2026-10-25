"""Benchmark CLI and saved-report chart regeneration."""

import argparse
import json
import sys
from dataclasses import replace

from vectorforge.benchmarks.config import ENGINES, BenchmarkConfig
from vectorforge.benchmarks.plotting import regenerate_charts
from vectorforge.benchmarks.reporting import summary_table, write_reports
from vectorforge.benchmarks.runner import run_benchmark

PRESETS = {
    "quick": {
        "sizes": (1000, 10000),
        "dimensions": (32, 128),
        "query_count": 50,
        "training_size": 2000,
        "nlist": 16,
        "repetitions": 3,
    },
    "standard": {
        "sizes": (10000, 50000),
        "dimensions": (32, 128),
        "query_count": 100,
        "training_size": 4096,
        "nlist": 32,
        "repetitions": 3,
    },
    "full": {
        "sizes": (10000, 50000, 100000),
        "dimensions": (32, 128),
        "query_count": 200,
        "training_size": 8192,
        "nlist": 64,
        "repetitions": 5,
    },
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Reproducible single-query vector search benchmarks"
    )
    result.add_argument(
        "--preset", choices=PRESETS, help="Explicit workload grid; larger presets are opt-in"
    )
    for flag, help_text in (
        ("size", "Indexed vector count (overrides preset sizes)"),
        ("dimension", "Vector dimension (overrides preset dimensions)"),
        ("training-size", "Independent training sample count"),
        ("query-count", "Independent query count"),
        ("k", "Retrieved neighbors"),
        ("nlist", "IVF centroid count"),
        ("seed", "Seed for independent data streams"),
        ("repetitions", "Timed repetitions per query"),
        ("warmup", "Untimed queries per probe setting"),
        ("threads", "BLAS and OpenMP thread limit (default 1)"),
        ("max-iter", "Lloyd/FAISS clustering iteration cap (default 25)"),
    ):
        result.add_argument(f"--{flag}", type=int, help=help_text)
    result.add_argument("--metric", choices=("cosine", "euclidean", "dot"), default="cosine")
    result.add_argument(
        "--distribution", choices=("uniform", "normal", "clustered"), default="normal"
    )
    result.add_argument("--duplicate-fraction", type=float, default=0.0)
    result.add_argument(
        "--nprobe", help="Comma-separated unique probe counts; default 1,2,4,8,16,nlist, filtered"
    )
    result.add_argument(
        "--engines", default=",".join(ENGINES), help="Comma-separated: " + ",".join(ENGINES)
    )
    result.add_argument(
        "--output", default="results", help="Parent directory for unique run folders"
    )
    result.add_argument("--no-charts", action="store_true", help="Explicitly omit chart generation")
    result.add_argument(
        "--dry-run", action="store_true", help="Validate and print configurations only"
    )
    result.add_argument(
        "--plot", metavar="RUN_DIRECTORY", help="Regenerate charts from saved summary.json only"
    )
    return result


def configurations(args: argparse.Namespace) -> list[BenchmarkConfig]:
    settings = dict(PRESETS[args.preset]) if args.preset else {}
    sizes = (args.size,) if args.size is not None else settings.pop("sizes", (1000,))
    dimensions = (
        (args.dimension,) if args.dimension is not None else settings.pop("dimensions", (32,))
    )
    settings.pop("sizes", None)
    settings.pop("dimensions", None)
    for key in (
        "training_size",
        "query_count",
        "k",
        "nlist",
        "seed",
        "repetitions",
        "warmup",
        "threads",
        "max_iter",
    ):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    nlist = settings.get("nlist", 16)
    probes = (
        tuple(int(x) for x in args.nprobe.split(","))
        if args.nprobe
        else tuple(sorted({p for p in (1, 2, 4, 8, 16, nlist) if p <= nlist}))
    )
    base = BenchmarkConfig(
        **settings,
        metric=args.metric,
        distribution=args.distribution,
        duplicate_fraction=args.duplicate_fraction,
        nprobes=probes,
        engines=tuple(args.engines.split(",")),
        output_dir=args.output,
    )
    return [
        replace(base, size=size, dimension=dimension) for size in sizes for dimension in dimensions
    ]


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.plot:
            paths = regenerate_charts(args.plot)
            print(f"Regenerated {len(paths)} charts in {args.plot}")
            return 0
        configs = configurations(args)
        if args.dry_run:
            print(json.dumps([config.to_dict() for config in configs], indent=2, allow_nan=False))
            return 0
        failed = False
        for config in configs:
            print(
                f"Running N={config.size}, d={config.dimension}, metric={config.metric}; "
                f"engines={config.engines}",
                flush=True,
            )
            run = run_benchmark(config)
            directory = write_reports(run, config.output_dir)
            print(summary_table(run.rows))
            print(f"Reports: {directory.resolve()}", flush=True)
            failed = failed or not run.complete
            if not args.no_charts and any(row["status"] == "ok" for row in run.rows):
                print(f"Charts: {len(regenerate_charts(directory))}", flush=True)
        return 1 if failed else 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
