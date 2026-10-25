"""Run with: uv run python examples/ivf_search.py."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from vectorforge import ExactVectorIndex, IVFVectorIndex, SearchResult
from vectorforge.evaluation import evaluate_recall


def print_results(label: str, matches: list[SearchResult]) -> None:
    print(f"\n{label}")
    for match in matches:
        print(f"  ID {match.id:4d}: {match.score:.6f}")


def main() -> None:
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(1000, 32))
    ids = np.arange(-500, 500, dtype=np.int64)
    queries = rng.normal(size=(20, 32))
    exact = ExactVectorIndex(32, metric="cosine")
    ivf = IVFVectorIndex(32, nlist=16, nprobe=1, metric="cosine", random_state=42)
    ivf.train(vectors)
    exact.add(vectors, ids)
    ivf.add(vectors, ids)

    print("VectorForge - IVF Search Demo")
    print(f"Vectors: {ivf.size}\nDimensions: {ivf.dimension}\nClusters: {ivf.nlist}")
    print_results("Exact search (cosine similarity):", exact.search(queries[0], 5))
    for probes in (1, 4, ivf.nlist):
        print_results(f"IVF search (nprobe={probes}):", ivf.search(queries[0], 5, nprobe=probes))

    print(f"\nMean Recall@5 over {len(queries)} queries:")
    for probes in (1, 4, ivf.nlist):
        report = evaluate_recall(ivf, exact, queries, 5, nprobe=probes)
        print(f"  nprobe={probes:2d}: {report.mean:.4f}")
        if probes == ivf.nlist and report.mean != 1.0:
            raise RuntimeError("Full probing did not achieve perfect recall")

    with TemporaryDirectory(prefix="vectorforge-ivf-") as directory:
        path = Path(directory) / "ivf.npz"
        ivf.save(path)
        restored = IVFVectorIndex.load(path)
        for query in queries:
            for probes in (1, 4, ivf.nlist):
                if restored.search(query, 5, nprobe=probes) != ivf.search(query, 5, nprobe=probes):
                    raise RuntimeError("Restored IVF results differ")
        print("\nPersistence verification: PASSED")
        print("Verified all queries and probe settings; temporary archive cleaned up on exit.")


if __name__ == "__main__":
    main()
