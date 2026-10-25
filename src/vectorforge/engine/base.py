"""Common interface for exact and future approximate indexes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import PathLike
from typing import Self

from numpy.typing import ArrayLike

from vectorforge.engine._validation import Ids, Metric


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A stable vector ID and raw metric score (no conversion to a distance)."""

    id: int
    score: float


class BaseVectorIndex(ABC):
    """Indexes return best-first results, breaking equal scores by ascending ID.

    Cosine and dot scores are larger-is-better; Euclidean scores are distances
    and smaller-is-better. Implementations must validate mutations atomically.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of coordinates in every vector."""

    @property
    @abstractmethod
    def metric(self) -> Metric:
        """Configured scoring metric."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of stored vectors."""

    @abstractmethod
    def add(self, vectors: ArrayLike, ids: Ids) -> None:
        """Insert one vector or a batch, rejecting existing or repeated IDs."""

    @abstractmethod
    def search(self, query: ArrayLike, k: int) -> list[SearchResult]:
        """Return up to k results; valid searches on an empty index return []."""

    @abstractmethod
    def delete(self, ids: Ids) -> None:
        """Delete IDs; repeated and missing IDs are ignored."""

    @abstractmethod
    def save(self, path: str | PathLike[str]) -> None:
        """Atomically save the current index to the exact given path."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | PathLike[str]) -> Self:
        """Construct a new index from validated persisted state."""
