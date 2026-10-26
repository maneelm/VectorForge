"""Wire contracts. Decimal string IDs preserve int64 precision in browsers."""

import re
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

Metric = Literal["cosine", "euclidean", "dot"]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Positive = Annotated[int, Field(strict=True, ge=1)]


def parse_id(value: object) -> int:
    if isinstance(value, str) and re.fullmatch(r"-?(0|[1-9][0-9]*)", value):
        value = int(value)
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise ValueError("IDs must be signed int64 integers or decimal strings")
    return value


VectorID = Annotated[int, BeforeValidator(parse_id)]
Row = Annotated[list[Number], Field(min_length=1, max_length=2048)]
Matrix = Annotated[list[Row], Field(min_length=1, max_length=10000)]


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class IndexCreate(RequestModel):
    name: str = Field(min_length=1, max_length=80)
    type: Literal["exact", "ivf"] = "exact"
    dimension: Annotated[int, Field(strict=True, ge=1, le=2048)]
    metric: Metric = "cosine"
    nlist: Annotated[int, Field(strict=True, ge=1, le=128)] = 16
    nprobe: Positive = 4
    random_state: Annotated[int, Field(strict=True, ge=0, le=2**31 - 1)] = 42
    max_iter: Annotated[int, Field(strict=True, ge=1, le=100)] = 100
    tol: Annotated[float, Field(strict=True, ge=0, le=1e6, allow_inf_nan=False)] = 1e-4

    @model_validator(mode="after")
    def configuration(self):
        fields = {"nlist", "nprobe", "random_state", "max_iter", "tol"}
        if self.type == "exact" and self.model_fields_set & fields:
            raise ValueError("Exact indexes do not accept IVF configuration")
        if self.type == "ivf" and self.nprobe > self.nlist:
            raise ValueError("nprobe cannot exceed nlist")
        return self


class IndexInfo(BaseModel):
    id: UUID
    name: str
    type: Literal["exact", "ivf"]
    dimension: int
    metric: Metric
    size: int
    is_trained: bool
    status: Literal["untrained", "ready"]
    configuration: dict


class Training(RequestModel):
    vectors: Matrix

    @model_validator(mode="after")
    def cells(self):
        if sum(map(len, self.vectors)) > 200000:
            raise ValueError("At most 200,000 coordinates per request")
        return self


class Insert(Training):
    ids: Annotated[list[VectorID], Field(min_length=1, max_length=2000)]

    @model_validator(mode="after")
    def counts(self):
        if len(self.ids) != len(self.vectors) or len(self.vectors) > 2000:
            raise ValueError("One ID per vector; at most 2,000 inserted vectors")
        return self


class DeleteVectors(RequestModel):
    ids: Annotated[list[VectorID], Field(min_length=1, max_length=2000)]


class Search(RequestModel):
    query: Row
    k: Annotated[int, Field(strict=True, ge=1, le=1000)] = 5
    nprobe: Positive | None = None


class Match(BaseModel):
    id: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    matches: list[Match]
    latency_ms: float
    latency_scope: str = "API-measured engine-call latency; not Phase 3 benchmark latency"
    index_type: str
    metric: Metric
    nprobe: int | None


class LoadIndex(RequestModel):
    archive_id: UUID


class SavedIndex(BaseModel):
    archive_id: UUID
    name: str
    type: str
    size_bytes: int


class QuickRun(RequestModel):
    preset: Literal["quick"] = "quick"
    metric: Metric = "cosine"
    seed: Annotated[int, Field(strict=True, ge=0, le=2**31 - 1)] = 42


class JobInfo(BaseModel):
    id: UUID
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    error: str | None = None
    log: str = ""
    run_ids: list[str] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    engine: Literal["vectorforge_exact", "vectorforge_ivf", "faiss_flat", "faiss_ivf"]
    status: Literal["ok", "failed"]
    error: str | None = None
    metric: Metric
    dataset_size: Positive
    dimension: Positive
    distribution: str
    seed: int
    k: Positive
    query_count: Positive
    repetitions: Positive
    training_size: Positive
    threads: Positive
    precision: Literal["float32", "float64"]
    nlist: int | None
    nprobe: int | None
    full_probe_direct_path: bool = False
    latency_mean_ms: Number | None = None
    latency_p50_ms: Number | None = None
    latency_p95_ms: Number | None = None
    latency_p99_ms: Number | None = None
    qps: Number | None = None
    recall_at_k: Number | None = None
    construction_ms: Number | None = None
    training_ms: Number | None = None
    insertion_ms: Number | None = None
    build_total_ms: Number | None = None
    rss_baseline_bytes: int | None = None
    rss_after_construction_bytes: int | None = None
    rss_after_training_bytes: int | None = None
    rss_after_insertion_bytes: int | None = None
    retained_rss_delta_bytes: int | None = None

    @model_validator(mode="after")
    def valid_measurements(self):
        expected_precision = "float32" if self.engine.startswith("faiss") else "float64"
        if self.precision != expected_precision:
            raise ValueError("Precision does not match engine")
        if self.full_probe_direct_path != (
            self.engine == "vectorforge_ivf" and self.nprobe == self.nlist
        ):
            raise ValueError("Invalid full-probe path label")
        if self.status == "ok":
            for field in (
                "latency_mean_ms",
                "latency_p50_ms",
                "latency_p95_ms",
                "latency_p99_ms",
                "qps",
                "recall_at_k",
                "construction_ms",
                "insertion_ms",
                "build_total_ms",
                "rss_baseline_bytes",
                "rss_after_insertion_bytes",
            ):
                if getattr(self, field) is None or getattr(self, field) < 0:
                    raise ValueError(f"Missing or invalid {field}")
            if self.recall_at_k > 1 or self.qps <= 0:
                raise ValueError("Invalid recall or throughput")
        elif not self.error:
            raise ValueError("Failed rows need an error")
        if self.engine.endswith("ivf"):
            if not self.nlist or not self.nprobe or not 1 <= self.nprobe <= self.nlist:
                raise ValueError("Invalid IVF probes")
            if self.status == "ok" and (self.training_ms is None or self.training_ms < 0):
                raise ValueError("Successful IVF rows need a nonnegative training time")
        elif self.nlist is not None or self.nprobe is not None:
            raise ValueError("Flat/exact rows cannot have IVF probes")
        return self


class RunInfo(BaseModel):
    id: str
    name: str
    source: Literal["published", "local"]
    timestamp: str
    complete: bool
    size: int
    dimension: int
    metric: Metric
    distribution: str
    result_count: int


class RunList(BaseModel):
    runs: list[RunInfo]
    issues: list[dict]


class RunResults(BaseModel):
    run: RunInfo
    metadata: dict
    results: list[BenchmarkResult]
