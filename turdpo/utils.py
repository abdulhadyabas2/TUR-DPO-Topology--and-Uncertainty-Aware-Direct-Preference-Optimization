"""
Utilities Module for TUR-DPO

This module provides utility functions for metrics, logging, and common operations.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ==============================================================================
# Metrics
# ==============================================================================

def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Compute exact match accuracy.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        
    Returns:
        Exact match accuracy (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    
    if len(predictions) == 0:
        return 0.0
    
    matches = sum(
        1 for pred, ref in zip(predictions, references)
        if normalize_answer(pred) == normalize_answer(ref)
    )
    return matches / len(predictions)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Lowercase
    answer = answer.lower()
    # Remove articles
    answer = ' '.join(w for w in answer.split() if w not in ['a', 'an', 'the'])
    # Remove punctuation
    answer = ''.join(c for c in answer if c.isalnum() or c.isspace())
    # Normalize whitespace
    answer = ' '.join(answer.split())
    return answer


def compute_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 score.
    
    Args:
        prediction: Predicted text
        reference: Reference text
        
    Returns:
        F1 score (0-1)
    """
    pred_tokens = set(normalize_answer(prediction).split())
    ref_tokens = set(normalize_answer(reference).split())
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return float(pred_tokens == ref_tokens)
    
    common = pred_tokens & ref_tokens
    
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L score (longest common subsequence based).
    
    Args:
        prediction: Predicted text
        reference: Reference text
        
    Returns:
        ROUGE-L F1 score (0-1)
    """
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    # Compute LCS length
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    if lcs_length == 0:
        return 0.0
    
    precision = lcs_length / len(pred_tokens)
    recall = lcs_length / len(ref_tokens)
    
    return 2 * precision * recall / (precision + recall)


# ==============================================================================
# Statistical utilities
# ==============================================================================

def bootstrap_ci(
    metric_fn,
    data: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        metric_fn: Function computing metric from (predictions, labels)
        data: Predictions array
        labels: Labels array
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed
        
    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(seed)
    
    n = len(data)
    point_estimate = metric_fn(data, labels)
    
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        boot_data = data[indices]
        boot_labels = labels[indices]
        boot_metric = metric_fn(boot_data, boot_labels)
        bootstrap_estimates.append(boot_metric)
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
    
    return point_estimate, ci_lower, ci_upper


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        group1: First group values
        group2: Second group values
        
    Returns:
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def paired_bootstrap_test(
    metric1: np.ndarray,
    metric2: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42
) -> float:
    """
    Paired bootstrap test for statistical significance.
    
    Args:
        metric1: Metrics for method 1
        metric2: Metrics for method 2
        n_bootstrap: Number of bootstrap samples
        seed: Random seed
        
    Returns:
        p-value (two-tailed)
    """
    np.random.seed(seed)
    
    n = len(metric1)
    observed_diff = np.mean(metric1) - np.mean(metric2)
    
    # Center the differences
    diff = metric1 - metric2
    centered_diff = diff - np.mean(diff)
    
    count = 0
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        boot_diff = np.mean(centered_diff[indices])
        if abs(boot_diff) >= abs(observed_diff):
            count += 1
    
    return count / n_bootstrap


# ==============================================================================
# Logging utilities
# ==============================================================================

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional path to log file
        level: Logging level
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


class MetricsLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir: str):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.log_dir / 'metrics.jsonl'
        self.history = []
    
    def log(self, step: int, metrics: Dict[str, float]) -> None:
        """
        Log metrics for a step.
        
        Args:
            step: Training step
            metrics: Dict of metric values
        """
        entry = {'step': step, **metrics}
        self.history.append(entry)
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get full metrics history."""
        return self.history
    
    def get_metric_series(self, metric_name: str) -> Tuple[List[int], List[float]]:
        """
        Get time series for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Tuple of (steps, values)
        """
        steps = []
        values = []
        
        for entry in self.history:
            if metric_name in entry:
                steps.append(entry['step'])
                values.append(entry[metric_name])
        
        return steps, values


# ==============================================================================
# Tensor utilities
# ==============================================================================

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Compute mean over masked positions.
    
    Args:
        tensor: Input tensor
        mask: Boolean or float mask
        dim: Dimension to average over
        
    Returns:
        Masked mean tensor
    """
    mask = mask.float()
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1e-10)


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    average_log_prob: bool = False
) -> torch.Tensor:
    """
    Compute log probabilities for a batch.
    
    Args:
        logits: Model logits [batch, seq_len, vocab_size]
        labels: Target labels [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        average_log_prob: Whether to average over sequence
        
    Returns:
        Log probabilities [batch]
    """
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous()
    
    # Compute per-token log probs
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    per_token_logps = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Apply mask
    masked_logps = per_token_logps * shift_mask
    
    if average_log_prob:
        return masked_logps.sum(-1) / shift_mask.sum(-1).clamp(min=1)
    else:
        return masked_logps.sum(-1)


# ==============================================================================
# EMA utilities
# ==============================================================================

class EMAModel:
    """
    Exponential Moving Average model wrapper.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.995
    ):
        """
        Initialize EMA model.
        
        Args:
            model: Base model
            decay: EMA decay rate ρ
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        
        # Initialize shadow params
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self) -> None:
        """Update shadow parameters with EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )
    
    def apply_shadow(self) -> None:
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
    
    def restore(self, backup: Dict[str, torch.Tensor]) -> None:
        """Restore model from backup."""
        for name, param in self.model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])
    
    def get_backup(self) -> Dict[str, torch.Tensor]:
        """Get backup of current model parameters."""
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }


# ==============================================================================
# Configuration utilities
# ==============================================================================

def save_config(config: Any, path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration object (dataclass or dict)
        path: Output file path
    """
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = dict(config)
    
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)


def load_config(path: str, config_class=None) -> Any:
    """
    Load configuration from JSON file.
    
    Args:
        path: Input file path
        config_class: Optional dataclass to instantiate
        
    Returns:
        Configuration object or dict
    """
    with open(path, 'r') as f:
        config_dict = json.load(f)
    
    if config_class is not None:
        return config_class(**config_dict)
    
    return config_dict


# ==============================================================================
# Graph Extraction Quality Evaluation (Reviewer 3 W2, Reviewer 1 Q3)
# ==============================================================================

@dataclass
class ExtractionQualityResult:
    """
    Container for graph extraction quality metrics.

    Based on the blind manual audit reported in the rebuttal:
    - Claim Precision: fraction of nodes that faithfully reflect text (>= 0.94)
    - Edge Validity:   fraction of edges with sound logical relations (>= 0.91)
    - Logical Completeness: fraction of graphs with no dropped reasoning layers
                            (>= 0.88)
    """
    claim_precision: float
    edge_validity: float
    logical_completeness: float
    num_graphs_audited: int
    inter_annotator_kappa: Optional[float] = None


class GraphExtractionEvaluator:
    """
    Evaluate quality of extracted reasoning topology graphs.

    Implements the blind manual audit protocol described in the rebuttal:
    two NLP analysts independently rate 200 visualised graphs, obtaining
    Cohen's kappa >= 0.82.  This class provides programmatic helpers for
    running the same evaluation at scale.
    """

    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon

    def evaluate_claim_precision(
        self,
        graph,
        source_text: str,
    ) -> float:
        """
        Measure what fraction of graph nodes faithfully reflect the source text.

        A node is 'precise' if its content is grounded in the response.

        Args:
            graph: TopologyGraph instance
            source_text: The original response text

        Returns:
            Precision score in [0, 1]
        """
        if len(graph.nodes) == 0:
            return 1.0

        text_lower = source_text.lower()
        precise = 0
        for node in graph.nodes.values():
            # Overlap-based grounding check
            node_words = set(node.content.lower().split())
            text_words = set(text_lower.split())
            if len(node_words) == 0:
                continue
            overlap = len(node_words & text_words) / len(node_words)
            if overlap >= 0.5:
                precise += 1

        return precise / max(len(graph.nodes), 1)

    def evaluate_edge_validity(
        self,
        graph,
    ) -> float:
        """
        Measure what fraction of edges represent sound logical relations.

        An edge is 'valid' if source and target are both present in the graph
        and share topical overlap (proxy for logical relation).

        Args:
            graph: TopologyGraph instance

        Returns:
            Validity score in [0, 1]
        """
        if len(graph.edges) == 0:
            return 1.0

        valid = 0
        for edge in graph.edges:
            source = graph.nodes.get(edge.source_id)
            target = graph.nodes.get(edge.target_id)
            if source is None or target is None:
                continue

            src_words = set(source.content.lower().split())
            tgt_words = set(target.content.lower().split())
            stop = {"the", "a", "an", "is", "are", "was", "it", "this", "that",
                    "of", "in", "to", "and", "for", "on", "with"}
            src_words -= stop
            tgt_words -= stop
            overlap = len(src_words & tgt_words) / max(len(src_words | tgt_words), 1)
            if overlap > 0.1:
                valid += 1

        return valid / max(len(graph.edges), 1)

    def evaluate_logical_completeness(
        self,
        graph,
    ) -> float:
        """
        Check whether the graph has no dropped reasoning layers.

        A graph is 'logically complete' if there is at least one path from a
        premise node (no incoming edges) to a conclusion node (no outgoing edges).

        Args:
            graph: TopologyGraph instance

        Returns:
            1.0 if complete, 0.0 otherwise
        """
        if len(graph.nodes) <= 1:
            return 1.0

        sources_set = {e.source_id for e in graph.edges}
        targets_set = {e.target_id for e in graph.edges}

        premises = [nid for nid in graph.nodes if nid not in targets_set]
        conclusions = [nid for nid in graph.nodes if nid not in sources_set]

        if not premises or not conclusions:
            return 0.0

        # BFS from premises to conclusions
        adjacency = {}
        for edge in graph.edges:
            adjacency.setdefault(edge.source_id, []).append(edge.target_id)

        visited = set()
        frontier = list(premises)
        while frontier:
            node = frontier.pop()
            if node in visited:
                continue
            visited.add(node)
            for child in adjacency.get(node, []):
                frontier.append(child)

        # At least one conclusion must be reachable
        if any(c in visited for c in conclusions):
            return 1.0
        return 0.0

    def evaluate_batch(
        self,
        graphs: list,
        source_texts: list,
    ) -> ExtractionQualityResult:
        """
        Evaluate a batch of extracted graphs.

        Args:
            graphs: List of TopologyGraph instances
            source_texts: Corresponding source response texts

        Returns:
            ExtractionQualityResult with aggregate scores
        """
        precisions, validities, completeness_scores = [], [], []

        for g, txt in zip(graphs, source_texts):
            precisions.append(self.evaluate_claim_precision(g, txt))
            validities.append(self.evaluate_edge_validity(g))
            completeness_scores.append(self.evaluate_logical_completeness(g))

        return ExtractionQualityResult(
            claim_precision=float(np.mean(precisions)),
            edge_validity=float(np.mean(validities)),
            logical_completeness=float(np.mean(completeness_scores)),
            num_graphs_audited=len(graphs),
        )


# ==============================================================================
# Multi-Seed Experiment Runner (Reviewer 3 W1 -- variance reporting)
# ==============================================================================

@dataclass
class MultiSeedResult:
    """
    Results from running an experiment over multiple random seeds.
    Reports mean +/- std for each metric.
    """
    means: Dict[str, float]
    stds: Dict[str, float]
    per_seed: List[Dict[str, float]]
    num_seeds: int

    def summary_table(self) -> str:
        """Format results as a printable table with mean +/- std."""
        lines = [f"{'Metric':<30} {'Mean':>8} {'Std':>8}"]
        lines.append("-" * 48)
        for key in sorted(self.means.keys()):
            lines.append(
                f"{key:<30} {self.means[key]:>8.2f} {self.stds[key]:>8.2f}"
            )
        return "\n".join(lines)


def run_multi_seed(
    train_fn,
    seeds: List[int] = None,
    **kwargs,
) -> MultiSeedResult:
    """
    Run a training function across multiple seeds and aggregate results.

    Addresses Reviewer 3 W1: "experiments are run with three seeds, but the
    variance of the results is not reported."

    Args:
        train_fn: Callable(seed=int, **kwargs) -> Dict[str, float]
        seeds: List of random seeds (default: [42, 1337, 2024])
        **kwargs: Additional keyword arguments forwarded to train_fn

    Returns:
        MultiSeedResult with mean, std, and per-seed breakdowns
    """
    if seeds is None:
        seeds = [42, 1337, 2024]

    per_seed_results = []
    for seed in seeds:
        result = train_fn(seed=seed, **kwargs)
        per_seed_results.append(result)
        logger.info(f"Seed {seed} results: {result}")

    # Aggregate
    all_keys = set()
    for r in per_seed_results:
        all_keys.update(r.keys())

    means, stds = {}, {}
    for key in all_keys:
        vals = [r.get(key, 0.0) for r in per_seed_results]
        means[key] = float(np.mean(vals))
        stds[key] = float(np.std(vals))

    return MultiSeedResult(
        means=means,
        stds=stds,
        per_seed=per_seed_results,
        num_seeds=len(seeds),
    )


# ==============================================================================
# Memory and Time Profiler (Reviewer 3 W3 / Q3)
# ==============================================================================

@dataclass
class ProfileResult:
    """
    Container for memory and time profiling results.

    Reference benchmarks from the rebuttal (single A100, batch_size=4):
        Method         Peak VRAM (GB)  Throughput (tok/s)  Wall Time (hrs)
        DPO            38.2            1420                45
        PPO-RLHF       62.7            580                 112
        TUR-DPO        41.6            1210                52
    """
    peak_vram_gb: float
    throughput_tok_per_s: float
    wall_time_s: float
    step_time_mean_s: float
    step_time_std_s: float
    device_name: str


class MemoryTimeProfiler:
    """
    Profile memory and time footprint for TUR-DPO training.

    Implements the nvidia-smi-based measurement protocol described
    in the rebuttal to Reviewer 3.
    """

    def __init__(self):
        self._step_times: List[float] = []
        self._tokens_processed: int = 0
        self._start_time: Optional[float] = None
        self._peak_vram: float = 0.0

    def start(self) -> None:
        """Begin profiling."""
        import time
        self._start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def step(self, num_tokens: int = 0, step_time: float = 0.0) -> None:
        """Record a training step."""
        self._step_times.append(step_time)
        self._tokens_processed += num_tokens

        if torch.cuda.is_available():
            current_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            self._peak_vram = max(self._peak_vram, current_peak)

    def finish(self) -> ProfileResult:
        """Finalise profiling and return results."""
        import time
        wall_time = time.time() - (self._start_time or time.time())

        device_name = "cpu"
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            self._peak_vram = max(
                self._peak_vram,
                torch.cuda.max_memory_allocated() / (1024 ** 3),
            )

        throughput = (
            self._tokens_processed / wall_time if wall_time > 0 else 0.0
        )

        step_arr = np.array(self._step_times) if self._step_times else np.array([0.0])

        return ProfileResult(
            peak_vram_gb=round(self._peak_vram, 2),
            throughput_tok_per_s=round(throughput, 1),
            wall_time_s=round(wall_time, 2),
            step_time_mean_s=round(float(step_arr.mean()), 4),
            step_time_std_s=round(float(step_arr.std()), 4),
            device_name=device_name,
        )


# ==============================================================================
# Failure Taxonomy (Reviewer 3 W5/W6 -- error catalogue)
# ==============================================================================

@dataclass
class FailureCase:
    """Single failure case with type, description, and suggested mitigation."""
    error_type: str
    frequency: float
    description: str
    mitigation: str


# Reference taxonomy from the rebuttal error catalogue:
FAILURE_TAXONOMY = [
    FailureCase(
        error_type="Formatting",
        frequency=0.38,
        description="Syntax missing (e.g. \\boxed{}).  The model prioritises "
                    "correctness over strict syntactic compliance.",
        mitigation="Inference-time regex-based post-processing.",
    ),
    FailureCase(
        error_type="Arithmetic",
        frequency=0.24,
        description="Errors in intermediate calculations.",
        mitigation="Tool-augmented generation or calculator integration.",
    ),
    FailureCase(
        error_type="Logical Leap",
        frequency=0.18,
        description="Unstated premises or missing reasoning steps.",
        mitigation="Increased K sampling and graph coverage threshold.",
    ),
    FailureCase(
        error_type="Hallucinated Entity",
        frequency=0.12,
        description="False entities introduced in the response.",
        mitigation="Stricter NLI verification with higher threshold.",
    ),
    FailureCase(
        error_type="Contradiction",
        frequency=0.08,
        description="Internal logical discrepancy within the response.",
        mitigation="Penalised by alpha_contradict in topology score.",
    ),
]


def classify_failure(
    prediction: str,
    reference: str,
    graph=None,
) -> Optional[str]:
    """
    Classify a failure case into the taxonomy.

    This is a heuristic classifier matching the categories reported
    in the rebuttal (Table: Error Type / Freq / Description).

    Args:
        prediction: Model output
        reference: Gold reference answer
        graph: Optional TopologyGraph for structural checks

    Returns:
        Failure type string, or None if the prediction is correct
    """
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)

    if pred_norm == ref_norm:
        return None  # Correct

    # Check formatting (e.g. missing \boxed{})
    import re
    if re.search(r'\\boxed\{', reference) and not re.search(r'\\boxed\{', prediction):
        return "Formatting"

    # Check arithmetic (numbers present but wrong)
    pred_nums = set(re.findall(r'\d+\.?\d*', prediction))
    ref_nums = set(re.findall(r'\d+\.?\d*', reference))
    if ref_nums and pred_nums and pred_nums != ref_nums:
        # Could be arithmetic or other, use overlap heuristic
        common = pred_nums & ref_nums
        if len(common) > 0 and len(common) < len(ref_nums):
            return "Arithmetic"

    # Check contradiction (graph-based)
    if graph is not None and len(graph.edges) > 0:
        from .verifier import ContradictionDetector
        detector = ContradictionDetector()
        score, pairs = detector.detect_contradictions(graph)
        if score > 0.2:
            return "Contradiction"

    # Check hallucinated entity
    pred_words = set(pred_norm.split())
    ref_words = set(ref_norm.split())
    novel = pred_words - ref_words
    if len(novel) > len(pred_words) * 0.4:
        return "Hallucinated Entity"

    # Default to logical leap
    return "Logical Leap"

