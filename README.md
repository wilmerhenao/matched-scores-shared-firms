# Matched Scores, Shared Firms

Source code for **Matched Scores, Shared Firms: A Controlled Benchmark for Endpoint Exposure and False Precision in Firm-Pair Machine Learning** (arXiv preprint, 2026).

**Wilmer Emisser Henao**

This repository contains the implementation underlying the paper. It implements the established dyadic double/debiased machine learning estimator. The paper contributes a controlled benchmark and a financial application, not a new estimator or limit theory. The release preserves the estimator, data pipeline, and experiment logic used for the reported results, and later changes can be recorded as separate commits.

Firm-pair observations reuse the same firms across many pairs. The benchmark separates two choices that are easy to confuse: admitting scoring firms into auxiliary-model training, and calculating uncertainty as though pair observations were independent. Start with the comparison below, then use the paper-to-code map to inspect the implementation.

## What the benchmark compares

An observation is an unordered pair of firms. Reusing firms across pairs creates dependence that the pair count does not capture.

- **Training composition:** strict node-disjoint, full-complement endpoint-exposed, and equal-count endpoint-exposed nuisance training use the same folds and scored pair identities. The equal-count arm holds the training-row count fixed, while the full-complement arm also increases it.
- **Variance aggregation:** the node-overlap and independent-edge calculations use the same fitted strict numerical scores. Only the variance calculation changes.

In the Gaussian complete-graph ridge simulations, equal-count exposure reduces RMSE without a detectable paired mean shift. This finding is specific to that design and is not a general endorsement of endpoint exposure. In the application, exposed-arm point shifts are descriptive, and better nuisance prediction does not establish valid inference for those arms.

The primary public-data application uses 493 firms. For a +0.10 increase in binary reporting-filer cosine, the paper reports the following. The outcome is the Fisher-z-transformed FF5-residual correlation.

| Calculation on the same fitted strict scores | Association | Standard error | Reported p-value | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Node-overlap | 0.00630 | 0.00503 | .211 | Primary inference under the maintained conditional model |
| Independent-edge | 0.00630 | 0.00277 | .023 | Deliberately unsupported diagnostic illustrating false precision |

The primary 95% interval is [-0.00357, 0.01616]. Its uncertainty accounts for shared firms under the conditional random-node and dissociation assumptions. It does not cover unrestricted dependence between firm-disjoint pairs induced by shared reporting filers, and common-date resampling does not resolve that limitation. The wide interval shows imprecision, not that the association is absent. The application is descriptive, not causal.

For inference under the dyadic DML framework used in the paper, keep nuisance training node-disjoint and use node-overlap variance under its remaining assumptions. Accuracy gains under endpoint-exposed training do not establish valid confidence intervals for those arms. For a prediction task, the endpoint regime should reflect which firms will have been observed at deployment.

## Paper-to-code map

The references to tables, figures, and sections below describe the arXiv version of the paper. Each link opens a file included in this release.

| Paper component | Implementation to inspect |
| --- | --- |
| Strict cross-fitting and the estimating equation | [src/dyadic_dml.py](src/dyadic_dml.py): `make_node_folds`, `fold_edge_masks`, `fit_plr_dml` |
| Full-complement endpoint exposure | [src/dyadic_dml.py](src/dyadic_dml.py): `fit_matched_score_leaky_complement_plr_diagnostic` |
| Equal-count endpoint exposure | [src/dyadic_dml.py](src/dyadic_dml.py): `fit_matched_score_equal_size_maximal_leak_plr_diagnostic`; deterministic outcome-blind sampling uses `sampling_salt` |
| Same-score node-overlap/IID comparison | [src/dyadic_dml.py](src/dyadic_dml.py): `dyadic_variance_component`; the fitted result carries both `se_dyadic` and `se_iid` |
| Primary 25-partition reporting rule | [src/dyadic_dml.py](src/dyadic_dml.py): `aggregate_repeated_splits`; [run_continuous_dml.py](experiments/run_continuous_dml.py): `PRIMARY_SEEDS` and repeated-split summaries |
| Figure 2: canonical three-arm calibration | [run_canonical_simulation.py](experiments/run_canonical_simulation.py) |
| Figure 2: node-dependence and repeated-split calibration | [run_node_share_simulation.py](experiments/run_node_share_simulation.py), [run_repeated_split_calibration.py](experiments/run_repeated_split_calibration.py) |
| Table 1: public identity and analysis gates | [public_13f.py](src/public_13f.py), [public_universe.py](src/public_universe.py), [build_public_ownership.py](experiments/build_public_ownership.py), [build_immutable_price_panel.py](scripts/build_immutable_price_panel.py), [build_return_design.py](experiments/build_return_design.py) |
| Table 2: primary and sensitivity estimates | [run_continuous_dml.py](experiments/run_continuous_dml.py); generated outputs include `repeated_split_aggregates.csv`, `estimates.csv`, and `summary.json` |
| Table 2: OLS diagnostics | [run_full_graph_ols_diagnostic.py](experiments/run_full_graph_ols_diagnostic.py) |
| Section 6.5: common-date sensitivity | [run_date_block_bootstrap.py](experiments/run_date_block_bootstrap.py) |
| Figure 3: ridge/HGB exposure shifts | [run_ridge_equal_count_audit.py](experiments/run_ridge_equal_count_audit.py), [run_hgb_learner_audit.py](experiments/run_hgb_learner_audit.py) |

"Matched scores" has two operational meanings. The training arms share the identities of the scored pairs, while their fitted numerical scores may differ. The variance comparison shares the actual fitted strict numerical scores.

## Contents

| Location | Purpose |
| --- | --- |
| `src/` | Estimator, node-overlap variance, filing-chain reconstruction, issuer linkage, and provenance helpers |
| `scripts/` | Public-input retrieval and immutable price-panel construction |
| `experiments/` | Ownership/return designs, empirical fits, exposure diagnostics, and simulations |
| `tests/` | Estimator identities, split invariants, source gates, design roles, and output checks |
| `support/source_contracts.py` | Public-source contracts, materialized by the setup helper |
| `reviewer_instructions.py` | Setup metadata and workflow commands; filename retained from an earlier release |
| `release_manifest.py`, `verify_release.py` | File hashes and integrity/documentation verification |

This is a source-only release. It does not distribute the manuscript, figures, cached data, private review correspondence, or frozen result files. The tracked tree contains Python source plus this README and the MIT license, and setup generates dependency metadata and two JSON source contracts locally.

## Setup and local checks

Python **3.13** was used for the reported runs. From the repository root:

```bash
python3.13 -B verify_release.py
python3.13 reviewer_instructions.py --prepare
python3.13 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/python -B verify_release.py --tests
```

The unit suite uses synthetic fixtures and does not require downloading the empirical dataset or rerunning the full experiments. `--prepare` writes `pyproject.toml` and `source_manifests/*.json` and refuses to overwrite different existing files. Run commands from this checkout's root, because the release is organized as research scripts, not a general importable library.

To check the core estimator invariants first:

```bash
.venv/bin/python -m pytest tests/test_dyadic_dml.py -q
```

These tests cover node-disjoint masks, matched scored-pair identities, equal training counts, deterministic sampling, the moment equation, brute-force variance agreement, endpoint symmetry, and repeated-split uncertainty. They show what the controlled comparisons hold fixed.

## Canonical simulation: no empirical inputs required

After installing dependencies, the canonical Gaussian simulation generates its own data:

```bash
.venv/bin/python experiments/run_canonical_simulation.py \
  --replications-per-cell 1000 --workers 8
```

This runs the full declared experiment and may be expensive. Consult `--help` and reduce workers to suit available memory. The unit suite is enough for local code checks, and the full simulation is a separate operation.

The node-share and repeated-split simulations also bind to a validated application design, so they are not stand-alone alternatives to reconstructing the empirical inputs.

## Empirical workflow and remaining reconstruction work

Print setup and workflow commands with:

```bash
python3 reviewer_instructions.py
```

The empirical path needs network access, Poppler's `pdftotext`, `curl`, a configured Kaggle CLI, and a descriptive SEC user agent containing the researcher's name and contact email. The helper does not install these prerequisites. Keep credentials outside the repository.

**The release does not yet provide a verified, from-scratch replay of every empirical input.** In particular, the immutable panel expects historical Yahoo baseline and reference batches under `data_public/raw/yfinance_2023_2024_v2/` and `data_public/raw/yfinance_2023_2024/`. The frozen baseline used 473 requested tickers, with 26 separately specified recoveries completing the 499-firm ownership universe. The printed workflow does not recreate both historical baseline sequences, and calling the downloader on today's complete 499-row ownership table is not an equivalent replay. The immutable-panel stage cannot run end to end without those inputs or a validated reconstruction of their original selection and acquisition steps.

Public APIs and vendor histories can change, and even current SEC endpoints may no longer match the pinned source hashes. A fresh reconstruction can stop when an input is unavailable or fails a source or coverage check. Keep those checks intact. Local tests validate code behavior. They do not show that external sources are still available or that the reported numbers can be reproduced exactly.

Once valid baseline inputs and the application design are available, the included drivers cover primary DML, OLS, ridge/HGB exposure comparisons, date sensitivity, node-share calibration, and repeated-split calibration. Production stages record configuration and file fingerprints in `results/MANIFEST.jsonl`.

## Releasing later changes

Keep code changes and result-producing runs traceable. `release_manifest.py` records hashes for all release files except itself; refresh it when intentionally changing the release and rerun the verifier and relevant tests. A hash check shows integrity against that manifest, not independent scientific validation.

Keep generated `data_public/`, `results/`, `source_manifests/`, `pyproject.toml`, environments, package metadata, and caches out of commits. When publishing a revised paper version, record the exact commit or release tag in the paper and describe only the contents actually distributed.

See [LICENSE.md](LICENSE.md) for the existing MIT license.

## Citation and methodological attribution

The paper is an arXiv preprint. Its arXiv identifier will be added here after posting; none is inferred.

```bibtex
@misc{henao2026matchedscores,
  author = {Henao, Wilmer Emisser},
  title = {Matched Scores, Shared Firms: A Controlled Benchmark for Endpoint Exposure and False Precision in Firm-Pair Machine Learning},
  year = {2026},
  note = {arXiv preprint}
}
```

Wilmer Emisser Henao authored this benchmark implementation. The underlying methods should also be credited:

- Chiang, Ma, Rodrigue, and Sasaki (2025), *Double/Debiased Machine Learning for Dyadic Data*: [10.1017/S0266466625100273](https://doi.org/10.1017/S0266466625100273).
- Chernozhukov et al. (2018), *Double/Debiased Machine Learning for Treatment and Structural Parameters*, including the repeated-split reporting rule: [10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097).
- Aronow, Samii, and Assenova (2015), *Cluster-Robust Variance Estimation for Dyadic Data*, used for the full-graph OLS diagnostic: [10.1093/pan/mpv018](https://doi.org/10.1093/pan/mpv018).
