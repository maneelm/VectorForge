"""Independent seeded training/index/query streams with verifiable fingerprints."""

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vectorforge.benchmarks.config import BenchmarkConfig


@dataclass(frozen=True, slots=True)
class Dataset:
    training: NDArray[np.float64]
    vectors: NDArray[np.float64]
    queries: NDArray[np.float64]
    ids: NDArray[np.int64]

    def fingerprints(self) -> dict[str, str]:
        return {
            key: hashlib.sha256(getattr(self, key).tobytes(order="C")).hexdigest()
            for key in ("training", "vectors", "queries", "ids")
        }


def generate_dataset(config: BenchmarkConfig) -> Dataset:
    """Streams share clustered population centers, never sampled data rows."""
    population_seed, train_seed, index_seed, query_seed = np.random.SeedSequence(config.seed).spawn(
        4
    )
    population = np.random.default_rng(population_seed)
    centers = population.normal(0, 4, size=(min(16, config.nlist), config.dimension))

    def sample(count: int, seed: np.random.SeedSequence) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        shape = (count, config.dimension)
        if config.distribution == "uniform":
            return rng.uniform(-1, 1, size=shape)
        if config.distribution == "normal":
            return rng.normal(size=shape)
        labels = rng.integers(len(centers), size=count)
        return np.ascontiguousarray(centers[labels] + rng.normal(0, 0.5, size=shape))

    training = sample(config.training_size, train_seed)
    vectors = sample(config.size, index_seed)
    queries = sample(config.query_count, query_seed)
    duplicate_count = int(config.size * config.duplicate_fraction)
    if duplicate_count:
        vectors[-duplicate_count:] = vectors[0]
    # Includes negative IDs and -1, independently of FAISS's missing-row sentinel.
    ids = np.arange(config.size, dtype=np.int64) - config.size // 2
    return Dataset(training, vectors, queries, ids)
