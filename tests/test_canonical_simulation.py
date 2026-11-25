from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from experiments.run_canonical_simulation import (
    CANONICAL_OUTPUT_SCHEMA_VERSION,
    DEFAULT_NONPILOT_CELLS,
    EQUAL_SIZE_LEAK_SAMPLING_SALT,
    PILOT_CELLS,
    REJECTION5_DEFINITION,
    SIMULATION_ARMS,
    THETA_ZERO,
    build_parser,
    generate_plr_dgp,
    nuisance_coefficients,
    one_replication,
    oracle_factories,
    simulation_cells,
    summarize_cell,
)


def test_replication_override_preserves_legacy_defaults_and_fixed_pilot() -> None:
    assert simulation_cells(pilot=False, replications_per_cell=None) == list(
        DEFAULT_NONPILOT_CELLS
    )
    overridden = simulation_cells(pilot=False, replications_per_cell=1000)
    assert [(n, p) for n, p, _ in overridden] == [
        (n, p) for n, p, _ in DEFAULT_NONPILOT_CELLS
    ]
    assert {repetitions for _, _, repetitions in overridden} == {1000}
    assert simulation_cells(pilot=True, replications_per_cell=1000) == list(
        PILOT_CELLS
    )


def test_replications_per_cell_cli_requires_positive_integer() -> None:
    parser = build_parser()
    assert parser.parse_args(["--replications-per-cell", "1000"]).replications_per_cell == 1000
    for invalid in ("0", "-1"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--replications-per-cell", invalid])


def test_canonical_dgp_remains_bitwise_compatible() -> None:
    arrays = generate_plr_dgp(20, 5, 123)
    digests = [hashlib.sha256(array.tobytes()).hexdigest() for array in arrays]
    assert digests == [
        "57f67da5fda0eb470b6cd69bd870d071987651b894d546d9ee752d224fef9064",
        "b27f908619b8362d7130828cb21ba8741087f00530a34a4579e5c225ea372028",
        "3a913411eae8e17168f5b2a108ee93c7f74b2d059c281e0442b65f2518183f1a",
        "8d9e49a4dcb4f21fec731f2e3bb71d55f2fc72749c16fca3ca6eae07f132d3a9",
        "8a87baf1a3ef88a0bfd0a757327e962cf59fa114ff0171c17c684ac497efde45",
    ]


def test_oracle_factories_equal_known_conditional_nuisances() -> None:
    _, _, x, _, _ = generate_plr_dgp(20, 5, 321)
    beta, gamma = nuisance_coefficients(x.shape[1])
    outcome_factory, treatment_factory = oracle_factories(x.shape[1])
    outcome = clone(outcome_factory()).fit(x[:50], np.zeros(50))
    treatment = clone(treatment_factory()).fit(x[:50], np.zeros(50))
    assert np.allclose(outcome.predict(x), x @ (THETA_ZERO * beta + gamma))
    assert np.allclose(treatment.predict(x), x @ beta)


def test_oracle_arm_shares_scores_and_solves_exact_moment() -> None:
    result = one_replication((30, 8, 0, 20260809))
    for arm in ("oracle", "matched_leaky", "equal_size_leaky"):
        assert result[f"{arm}_n_scored"] == result["strict_n_scored"]
    for arm in SIMULATION_ARMS:
        assert abs(float(result[f"{arm}_moment"])) < 1e-12
    assert EQUAL_SIZE_LEAK_SAMPLING_SALT == 20260809


def test_summary_adds_oracle_without_removing_existing_keys() -> None:
    assert CANONICAL_OUTPUT_SCHEMA_VERSION == 2
    assert REJECTION5_DEFINITION == (
        "false rejection probability for the two-sided H0: theta = theta_zero "
        "at alpha = 0.05"
    )
    rows = [one_replication((30, 8, repetition, 100 + repetition)) for repetition in range(3)]
    summary = summarize_cell(pl.DataFrame(rows))
    for split in SIMULATION_ARMS:
        assert f"{split}_bias" in summary
        assert f"{split}_dyadic_coverage90" in summary
        assert f"{split}_iid_coverage90" in summary
        for variance in ("dyadic", "iid"):
            coverage95 = summary[f"{split}_{variance}_coverage95"]
            rejection5 = summary[f"{split}_{variance}_rejection5"]
            assert np.isclose(coverage95 + rejection5, 1.0)
            assert summary[f"{split}_{variance}_coverage95_mcse"] >= 0.0
