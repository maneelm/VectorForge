"""Shared IVF representation and nearest-centroid assignment."""

import numpy as np
from numpy.typing import NDArray

from vectorforge.engine._validation import Metric
from vectorforge.engine.distance import _unit
from vectorforge.engine.kmeans import squared_distances


def representation(vectors: NDArray[np.float64], metric: Metric) -> NDArray[np.float64]:
    """Cosine routes normalized vectors; other metrics route original vectors."""
    return _unit(vectors) if metric == "cosine" else vectors


def assign(
    vectors: NDArray[np.float64], centers: NDArray[np.float64], metric: Metric
) -> NDArray[np.int64]:
    """Each row goes to its nearest L2 centroid, ties to the lowest centroid ID."""
    return np.argmin(squared_distances(representation(vectors, metric), centers), axis=1).astype(
        np.int64
    )
