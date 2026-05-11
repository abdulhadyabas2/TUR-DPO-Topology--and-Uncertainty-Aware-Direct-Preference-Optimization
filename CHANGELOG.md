# Changelog

All notable changes to TUR-DPO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-11

### Added (Post-Review Revision)

- **Evaluation Script** (`evaluate.py`)
  - Exact Match with regex post-processing for formatting errors
  - Win-rate evaluation scaffold (LLM judge)
  - Calibration metrics (ECE, Brier)
  - Graph extraction quality audit
  - Memory/time profiling
  - Failure taxonomy breakdown
  - `--print_results` to display all paper/rebuttal benchmark tables

- **Graph Extraction Evaluator** (`utils.py`)
  - `GraphExtractionEvaluator` class with claim precision, edge validity, logical completeness
  - Matches blind manual audit protocol (200 graphs, Cohen's kappa >= 0.82)

- **Multi-Seed Runner** (`utils.py`)
  - `run_multi_seed()` for running experiments across 3 seeds with std reporting
  - `MultiSeedResult` dataclass with `summary_table()` method

- **Memory/Time Profiler** (`utils.py`)
  - `MemoryTimeProfiler` class using nvidia-smi peak VRAM tracking
  - `ProfileResult` with reference benchmarks (DPO: 38.2 GB, TUR-DPO: 41.6 GB)

- **Failure Taxonomy** (`utils.py`)
  - `FAILURE_TAXONOMY` catalogue: Formatting (38%), Arithmetic (24%),
    Logical Leap (18%), Hallucinated Entity (12%), Contradiction (8%)
  - `classify_failure()` heuristic classifier

- **DPO Preliminary and Proof Sketch** (`loss.py`)
  - Module docstring now includes formal DPO derivation
  - 5-step proof sketch showing TUR-DPO as a formal DPO generalisation
  - Shows gamma=0 reduces to standard DPO

- **Three-Tier Hyperparameter Framework** (`configs/default.json`)
  - Tier 1: Offline calibration (topology alphas)
  - Tier 2: Validation grid search (beta, gamma, a, lambda)
  - Tier 3: Fixed globally (lambda_epi, lambda_ale, tau_w, w_min)

### Changed

- **README.md** -- comprehensive rewrite with:
  - Standard deviations over 3 seeds
  - 70B scalability results (LLaMA-3-70B)
  - Mistral-7B-v0.3 and Gemma-7B-v1.1 full results tables
  - Domain-shift evaluation (MedQA, LexGLUE)
  - Topology component ablation table
  - Resource benchmarks (VRAM, throughput, wall time)
  - Graph extraction quality metrics
  - Failure taxonomy and mitigation strategies
  - Graph extraction pipeline explanation (fully offline, no feedback loop)
  - DPO preliminary and mathematical proof sketch
  - Supported models table

- **pyproject.toml** -- real author names, arXiv URL
- **__init__.py** -- real authors, paper URL, new exports
- **configs/default.json** -- three-tier framework with comments

## [0.1.0] - 2026-01-15

### Added

- **Core Algorithm Implementation**
  - TUR-DPO loss function with uncertainty-weighted preference optimization
  - Listwise TUR-DPO loss for comparing multiple responses
  - Baseline DPO and IPO loss implementations

- **Topology Module**
  - `TopologyGraph` class for representing reasoning graphs
  - `TopologyExtractor` for eliciting graphs from LLM responses
  - `TopologyScorer` implementing Equation 1 from the paper
  - Graph quality metrics: path coverage, cycle detection, dangling nodes, contradiction detection

- **Uncertainty Estimation**
  - `EpistemicUncertainty` using variance + Jensen-Shannon divergence
  - `AleatoricUncertainty` using smoothed entropy
  - `UncertaintyEstimator` combining both types (Equation 3)
  - `PairWeightComputer` for computing preference pair weights (Equation 6)

- **Reward Shaping**
  - `ShapedReward` implementing Equation 7
  - `SemanticScorer` for semantic quality assessment (Equation 2)
  - `LinearCalibrator` for reward calibration (Equation 8)
  - `RewardDifferenceComputer` for computing reward differences

- **Training Infrastructure**
  - `TURDPOTrainer` with complete training loop
  - `TURDPOConfig` dataclass for configuration management
  - EMA model updates
  - Gradient accumulation support
  - Checkpoint saving and loading

- **Verification Module**
  - `NodeVerifier` for multi-type node verification
  - `FactChecker` for factual correctness
  - `ArithmeticVerifier` for mathematical expressions
  - `ContradictionDetector` for finding logical contradictions

- **Calibration Module**
  - Expected Calibration Error (ECE) computation
  - Brier score calculation
  - Temperature scaling calibration
  - Isotonic regression calibration

- **Data Handling**
  - `PreferenceDataset` for pairwise preferences
  - `ListwisePreferenceDataset` for listwise ranking
  - Flexible data loading utilities

- **Utilities**
  - Evaluation metrics: Exact Match, Token F1, ROUGE-L
  - Statistical utilities: bootstrap CI, Cohen's d
  - EMA model wrapper
  - Logging setup

### Documentation

- Comprehensive README with installation and usage instructions
- API documentation with mathematical formulations
- Example scripts demonstrating usage
- Contributing guidelines

### Dependencies

- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- tqdm >= 4.60.0

## [Unreleased]

### Planned Features

- Multi-GPU distributed training support
- Integration with Hugging Face TRL library
- Pre-trained model checkpoints
- Visualisation tools for reasoning graphs
- Extended benchmark evaluation scripts

---

## Version History

- **0.2.0**: Post-review revision with evaluation, profiling, and expanded results
- **0.1.0**: Initial release with core TUR-DPO implementation
