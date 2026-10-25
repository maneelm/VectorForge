"""Regenerate six workload-specific figures from saved measurements, using Agg."""

import json
from pathlib import Path

from vectorforge.benchmarks.runner import SCHEMA_VERSION


def regenerate_charts(directory: str | Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("charts require: uv sync --locked --extra benchmark") from exc
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("benchmark_schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported benchmark report version")
    rows = [row for row in summary["results"] if row["status"] == "ok"]
    if not rows:
        raise ValueError("no successful measurements to plot")
    identities = {
        (
            r["dataset_size"],
            r["dimension"],
            r["metric"],
            r["distribution"],
            r["seed"],
            r["k"],
            r["query_count"],
            r["repetitions"],
            r["threads"],
            r["training_size"],
        )
        for r in rows
    }
    if len(identities) != 1:
        raise ValueError("refusing to mix unrelated benchmark workloads in charts")
    first = rows[0]
    title = (
        f"N={first['dataset_size']:,}, d={first['dimension']}, {first['metric']}, "
        f"{first['distribution']}, k={first['k']}"
    )
    labels = [r["engine"] + (f" p={r['nprobe']}" if r["nprobe"] is not None else "") for r in rows]
    chart_dir = directory / "charts"
    chart_dir.mkdir(exist_ok=True)
    paths = []

    def finish(fig, ax, filename, heading, xlabel, ylabel):
        ax.set(title=f"{heading}\n{title}", xlabel=xlabel, ylabel=ylabel)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = chart_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        for filename, field, heading, unit in (
            ("search_latency.png", "latency_mean_ms", "Mean single-query latency", "Latency (ms)"),
            (
                "throughput.png",
                "qps",
                "Measured single-query workload throughput",
                "Queries / second",
            ),
        ):
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.bar(labels, [r[field] for r in rows], label=heading)
            ax.tick_params(axis="x", labelrotation=55)
            ax.legend()
            finish(fig, ax, filename, heading, "Engine and probes", unit)
        fig, ax = plt.subplots(figsize=(8, 5))
        for engine in sorted({r["engine"] for r in rows if r["nprobe"] is not None}):
            selected = sorted([r for r in rows if r["engine"] == engine], key=lambda r: r["nprobe"])
            ax.plot(
                [r["nprobe"] for r in selected],
                [r["recall_at_k"] for r in selected],
                "o-",
                label=engine,
            )
        ax.set_ylim(0, 1.05)
        if ax.lines:
            ax.legend()
        finish(
            fig, ax, "recall_vs_nprobe.png", "Recall versus probes", "nprobe (lists)", "Recall@k"
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        for engine in sorted({row["engine"] for row in rows}):
            selected = sorted(
                [row for row in rows if row["engine"] == engine],
                key=lambda row: row["nprobe"] or 0,
            )
            ax.plot(
                [row["latency_mean_ms"] for row in selected],
                [row["recall_at_k"] for row in selected],
                "o-",
                label=engine,
                linewidth=1,
            )
            for row in selected:
                if row["nprobe"] is not None:
                    suffix = "*" if row["full_probe_direct_path"] else ""
                    ax.annotate(
                        f"p={row['nprobe']}{suffix}",
                        (row["latency_mean_ms"], row["recall_at_k"]),
                        xytext=(4, 5),
                        textcoords="offset points",
                        fontsize=8,
                    )
        ax.set_ylim(0, 1.05)
        logarithmic = all(row["latency_mean_ms"] > 0 for row in rows)
        if logarithmic:
            ax.set_xscale("log")
        if any(row["full_probe_direct_path"] for row in rows):
            ax.text(
                0.02,
                0.02,
                "* VectorForge full-probe direct ranking path",
                transform=ax.transAxes,
                fontsize=8,
            )
        ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
        finish(
            fig,
            ax,
            "recall_vs_latency.png",
            "Recall versus latency",
            "Mean latency (ms, log scale)" if logarithmic else "Mean latency (ms)",
            "Recall@k",
        )
        # Training/build and RSS snapshots are shared across probe settings; plot each engine once.
        engines = list({r["engine"]: r for r in rows}.values())
        names = [r["engine"] for r in engines]
        fig, ax = plt.subplots(figsize=(9, 5))
        bottom = [0.0] * len(engines)
        for field, label in (
            ("construction_ms", "Construction"),
            ("training_ms", "Training (IVF only)"),
            ("insertion_ms", "Insertion"),
        ):
            values = [r[field] or 0.0 for r in engines]
            ax.bar(names, values, bottom=bottom, label=label)
            bottom = [a + b for a, b in zip(bottom, values, strict=True)]
        ax.legend()
        finish(
            fig,
            ax,
            "build_time.png",
            "Index build stages (one build per engine)",
            "Engine",
            "Time (ms)",
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        positions = list(range(len(engines)))
        ax.bar(
            [i - 0.2 for i in positions],
            [r["rss_baseline_bytes"] / 2**20 for r in engines],
            width=0.4,
            label="Baseline (data + libraries)",
        )
        ax.bar(
            [i + 0.2 for i in positions],
            [r["rss_after_insertion_bytes"] / 2**20 for r in engines],
            width=0.4,
            label="After insertion (not peak)",
        )
        ax.set_xticks(positions, names)
        ax.legend()
        finish(fig, ax, "memory.png", "Isolated process RSS snapshots", "Engine", "RSS (MiB)")
    return paths
