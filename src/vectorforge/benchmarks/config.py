"""Validated, serializable benchmark workload configuration."""

from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np

from vectorforge.engine._validation import Metric, positive_integer, validate_metric

ENGINES = ("vectorforge_exact", "vectorforge_ivf", "faiss_flat", "faiss_ivf")
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    size: int = 1000
    dimension: int = 32
    training_size: int = 2000
    query_count: int = 50
    distribution: str = "normal"
    metric: Metric = "cosine"
    k: int = 10
    nlist: int = 16
    nprobes: tuple[int, ...] = (1, 4, 16)
    seed: int = 42
    repetitions: int = 3
    warmup: int = 5
    threads: int = 1
    max_iter: int = 25
    duplicate_fraction: float = 0.0
    engines: tuple[str, ...] = ENGINES
    output_dir: str = "results"

    def __post_init__(self) -> None:
        for key in (
            "size",
            "dimension",
            "training_size",
            "query_count",
            "k",
            "nlist",
            "repetitions",
            "threads",
            "max_iter",
        ):
            object.__setattr__(self, key, positive_integer(getattr(self, key), key))
        validate_metric(self.metric)
        if self.distribution not in ("uniform", "normal", "clustered"):
            raise ValueError("distribution must be uniform, normal, or clustered")
        if self.training_size < self.nlist:
            raise ValueError("training_size must be at least nlist")
        for key, maximum in (("warmup", 2**63 - 1), ("seed", 2**31 - 1)):
            value = getattr(self, key)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or not 0 <= value <= maximum
            ):
                raise ValueError(f"{key} must be an integer in [0, {maximum}]")
        if not self.nprobes or len(set(self.nprobes)) != len(self.nprobes):
            raise ValueError("nprobes must be nonempty and unique")
        for value in self.nprobes:
            if positive_integer(value, "nprobe") > self.nlist:
                raise ValueError("nprobe cannot exceed nlist")
        object.__setattr__(self, "nprobes", tuple(int(value) for value in self.nprobes))
        if (
            not self.engines
            or len(set(self.engines)) != len(self.engines)
            or any(e not in ENGINES for e in self.engines)
        ):
            raise ValueError(f"engines must be a unique nonempty selection from {ENGINES}")
        object.__setattr__(self, "engines", tuple(self.engines))
        fraction = self.duplicate_fraction
        if (
            isinstance(fraction, (bool, np.bool_))
            or not isinstance(fraction, Real)
            or not np.isfinite(fraction)
            or not 0 <= fraction < 1
        ):
            raise ValueError("duplicate_fraction must be finite and in [0, 1)")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a nonempty string")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "BenchmarkConfig":
        return cls(
            **{**values, "nprobes": tuple(values["nprobes"]), "engines": tuple(values["engines"])}
        )
