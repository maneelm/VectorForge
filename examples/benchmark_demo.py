"""Run with: uv run --extra benchmark python examples/benchmark_demo.py."""

from vectorforge.benchmarks.cli import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--size",
                "1000",
                "--dimension",
                "32",
                "--nlist",
                "16",
                "--nprobe",
                "1,4,16",
                "--output",
                "results/demo",
            ]
        )
    )
