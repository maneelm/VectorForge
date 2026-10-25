from dataclasses import replace

import numpy as np
import pytest

from vectorforge.benchmarks.cli import configurations, parser
from vectorforge.benchmarks.config import BenchmarkConfig
from vectorforge.benchmarks.datasets import generate_dataset


@pytest.mark.parametrize("distribution", ["uniform", "normal", "clustered"])
def test_reproducible_shapes_dtypes_and_independent_streams(distribution):
    config = BenchmarkConfig(
        size=40, dimension=7, training_size=40, query_count=40, distribution=distribution
    )
    first, second = generate_dataset(config), generate_dataset(config)
    assert first.fingerprints() == second.fingerprints()
    for name in ("training", "vectors", "queries"):
        values = getattr(first, name)
        assert values.shape == (40, 7)
        assert values.dtype == np.float64 and values.flags.c_contiguous
        assert np.isfinite(values).all()
    assert first.ids.dtype == np.int64
    assert len(set(first.ids)) == 40 and -1 in first.ids
    assert not np.array_equal(first.training, first.queries)
    assert not np.array_equal(first.training, first.vectors)
    assert not np.shares_memory(first.training, first.queries)
    assert generate_dataset(replace(config, seed=43)).fingerprints() != first.fingerprints()


def test_query_count_never_changes_training_or_indexed_vectors():
    config = BenchmarkConfig(size=30, training_size=40, query_count=10)
    first = generate_dataset(config)
    second = generate_dataset(replace(config, query_count=20))
    np.testing.assert_array_equal(first.training, second.training)
    np.testing.assert_array_equal(first.vectors, second.vectors)
    np.testing.assert_array_equal(first.queries, second.queries[:10])


def test_duplicate_option_only_changes_indexed_samples():
    config = BenchmarkConfig(size=20, duplicate_fraction=0.5)
    data = generate_dataset(config)
    np.testing.assert_array_equal(data.vectors[-10:], np.tile(data.vectors[0], (10, 1)))
    plain = generate_dataset(replace(config, duplicate_fraction=0))
    np.testing.assert_array_equal(data.training, plain.training)
    np.testing.assert_array_equal(data.queries, plain.queries)


@pytest.mark.parametrize(
    "change",
    [
        {"size": 0},
        {"size": True},
        {"dimension": -1},
        {"training_size": 0},
        {"training_size": 1, "nlist": 2},
        {"query_count": 0},
        {"nlist": 0},
        {"k": 0},
        {"threads": 0},
        {"threads": True},
        {"warmup": -1},
        {"warmup": 1.5},
        {"repetitions": 0},
        {"max_iter": 0},
        {"seed": -1},
        {"seed": 2**31},
        {"distribution": "bad"},
        {"metric": "bad"},
        {"nprobes": ()},
        {"nprobes": (1, 1)},
        {"nprobes": (17,)},
        {"nprobes": (True,)},
        {"engines": ()},
        {"engines": ("bad",)},
        {"engines": ("faiss_flat", "faiss_flat")},
        {"duplicate_fraction": -1},
        {"duplicate_fraction": 1},
        {"duplicate_fraction": np.nan},
        {"duplicate_fraction": True},
        {"output_dir": ""},
    ],
)
def test_invalid_configurations(change):
    with pytest.raises(ValueError):
        BenchmarkConfig(**change)


def test_config_round_trip():
    config = BenchmarkConfig(nprobes=(1, 4, 16))
    assert BenchmarkConfig.from_dict(config.to_dict()) == config


def test_presets_and_cli_override_filter_probes():
    quick = configurations(parser().parse_args(["--preset", "quick"]))
    assert {(c.size, c.dimension) for c in quick} == {
        (1000, 32),
        (1000, 128),
        (10000, 32),
        (10000, 128),
    }
    for preset in ("quick", "standard", "full"):
        for config in configurations(parser().parse_args(["--preset", preset])):
            assert config.training_size >= 39 * config.nlist
    custom = configurations(
        parser().parse_args(
            ["--preset", "quick", "--size", "80", "--dimension", "4", "--nlist", "2"]
        )
    )
    assert len(custom) == 1
    assert (custom[0].size, custom[0].dimension, custom[0].nprobes) == (80, 4, (1, 2))
