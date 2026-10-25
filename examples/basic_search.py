"""Run with: uv run python examples/basic_search.py."""

from pathlib import Path
from tempfile import TemporaryDirectory

from vectorforge import ExactVectorIndex


def main() -> None:
    index = ExactVectorIndex(dimension=3, metric="cosine")
    index.add([[1, 0, 0], [0.8, 0.2, 0], [0, 1, 0], [-1, 0, 0]], [101, 102, 103, 104])
    query = [1, 0.1, 0]
    print("Top matches (cosine similarity; higher is better):")
    for result in index.search(query, k=3):
        print(f"  ID {result.id}: {result.score:.6f}")

    index.delete(101)
    print(f"After deleting ID 101: {index.size} vectors remain")
    expected = index.search(query, k=3)
    with TemporaryDirectory(prefix="vectorforge-") as directory:
        path = Path(directory) / "example.npz"
        index.save(path)
        restored = ExactVectorIndex.load(path)
        actual = restored.search(query, k=3)
        if actual != expected:
            raise RuntimeError("Persistence changed search results")
        print("Restored matches:")
        for result in actual:
            print(f"  ID {result.id}: {result.score:.6f}")
        print("Persistence verified; temporary example archive is cleaned up automatically.")


if __name__ == "__main__":
    main()
