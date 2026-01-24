"""
Uncertainty Module for TUR-DPO

This module implements uncertainty estimation combining epistemic and aleatoric components.
Based on Equations (3), (4), and (5) from the paper:

Total uncertainty: u(G) = λ_epi * u_epi(G) + λ_ale * u_ale(G)

Epistemic uncertainty (Eq. 4):
    u_epi(G) = Var({s_topo(G^k)}_k) + JSD({P^k}_k)

Aleatoric uncertainty (Eq. 5):
    u_ale(G) = (1/|V|) Σ_v [-p̃_v log p̃_v - (1-p̃_v) log(1-p̃_v)]
    where p̃_v = (p_v + τ) / (1 + 2τ)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .topology import TopologyGraph, TopologyScorer


@dataclass
class UncertaintyResult:
    """Container for uncertainty estimation results."""
    total: float
    epistemic: float
    aleatoric: float
    components: Dict[str, float]


class EpistemicUncertainty:
    """
    Epistemic uncertainty from re-elicited graphs.
    
    Measures dispersion in structure and scores across K perturbed extractions.
    """
    
    def __init__(
        self,
        scorer: Optional[TopologyScorer] = None,
        epsilon: float = 1e-10
    ):
        """
        Initialize epistemic uncertainty estimator.
        
        Args:
            scorer: TopologyScorer instance for computing topology scores
            epsilon: Small constant for numerical stability
        """
        self.scorer = scorer or TopologyScorer()
        self.epsilon = epsilon
    
    def compute(
        self,
        graphs: List[TopologyGraph],
        contradiction_scores: Optional[List[float]] = None
    ) -> float:
        """
        Compute epistemic uncertainty from multiple graph samples.
        
        Based on Equation (4):
            u_epi(G) = Var({s_topo(G^k)}) + JSD({P^k})
        
        Args:
            graphs: List of K re-elicited topology graphs
            contradiction_scores: Optional list of contradiction scores for each graph
            
        Returns:
            Epistemic uncertainty score
        """
        if len(graphs) == 0:
            return 1.0  # Maximum uncertainty for no graphs
        
        if len(graphs) == 1:
            return 0.0  # No variance with single sample
        
        if contradiction_scores is None:
            contradiction_scores = [0.0] * len(graphs)
        
        # Compute topology scores for each graph
        topo_scores = [
            self.scorer.compute_score(g, c)
            for g, c in zip(graphs, contradiction_scores)
        ]
        
        # Variance of topology scores
        score_variance = np.var(topo_scores)
        
        # JSD of path/edge distributions
        distributions = [g.get_path_distribution() for g in graphs]
        jsd = self._compute_jsd(distributions)
        
        return score_variance + jsd
    
    def _compute_jsd(self, distributions: List[Dict[str, float]]) -> float:
        """
        Compute Jensen-Shannon divergence among multiple distributions.
        
        JSD(P1, P2, ..., Pk) = H(mixture) - (1/k) Σ H(Pi)
        """
        if len(distributions) == 0:
            return 0.0
        
        # Get all keys
        all_keys = set()
        for dist in distributions:
            all_keys.update(dist.keys())
        
        if not all_keys:
            return 0.0
        
        # Convert to probability vectors
        k = len(distributions)
        vectors = []
        for dist in distributions:
            vec = np.array([dist.get(key, self.epsilon) for key in all_keys])
            vec = vec / (vec.sum() + self.epsilon)  # Normalize
            vectors.append(vec)
        
        vectors = np.array(vectors)
        
        # Compute mixture distribution
        mixture = vectors.mean(axis=0)
        
        # Compute entropies
        def entropy(p):
            p = np.clip(p, self.epsilon, 1.0)
            return -np.sum(p * np.log2(p))
        
        mixture_entropy = entropy(mixture)
        individual_entropies = [entropy(v) for v in vectors]
        mean_entropy = np.mean(individual_entropies)
        
        # JSD = H(mixture) - mean(H(individual))
        jsd = mixture_entropy - mean_entropy
        
        return max(0.0, jsd)  # Ensure non-negative


class AleatoricUncertainty:
    """
    Aleatoric uncertainty from node-level verification.
    
    Measures ambiguity in correctness probabilities across graph nodes.
    """
    
    def __init__(
        self,
        tau: float = 0.05,
        epsilon: float = 1e-10
    ):
        """
        Initialize aleatoric uncertainty estimator.
        
        Args:
            tau: Smoothing prior strength (default: 0.05 from paper)
            epsilon: Small constant for numerical stability
        """
        self.tau = tau
        self.epsilon = epsilon
    
    def compute(
        self,
        graph: TopologyGraph,
        correctness_probs: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute aleatoric uncertainty from node correctness probabilities.
        
        Based on Equation (5):
            u_ale(G) = (1/|V|) Σ_v [-p̃_v log p̃_v - (1-p̃_v) log(1-p̃_v)]
            where p̃_v = (p_v + τ) / (1 + 2τ)
        
        Args:
            graph: Topology graph
            correctness_probs: Dict mapping node_id to correctness probability
            
        Returns:
            Aleatoric uncertainty score
        """
        if len(graph.nodes) == 0:
            return 1.0  # Maximum uncertainty for empty graph
        
        # Get correctness probabilities
        if correctness_probs is None:
            # Use node's stored correctness_prob
            probs = [node.correctness_prob for node in graph.nodes.values()]
        else:
            probs = [correctness_probs.get(node_id, 0.5) for node_id in graph.nodes]
        
        # Compute smoothed binary entropy for each node
        entropies = []
        for p_v in probs:
            # Smoothed probability: p̃_v = (p_v + τ) / (1 + 2τ)
            p_tilde = (p_v + self.tau) / (1 + 2 * self.tau)
            
            # Binary entropy: -p log p - (1-p) log(1-p)
            p_tilde = np.clip(p_tilde, self.epsilon, 1.0 - self.epsilon)
            h = -p_tilde * np.log(p_tilde) - (1 - p_tilde) * np.log(1 - p_tilde)
            entropies.append(h)
        
        # Average across nodes
        return np.mean(entropies)


class UncertaintyEstimator:
    """
    Combined uncertainty estimator for TUR-DPO.
    
    Combines epistemic (structural dispersion) and aleatoric (node-level) uncertainty.
    """
    
    def __init__(
        self,
        lambda_epi: float = 0.5,
        lambda_ale: float = 0.5,
        tau: float = 0.05,
        scorer: Optional[TopologyScorer] = None
    ):
        """
        Initialize combined uncertainty estimator.
        
        Args:
            lambda_epi: Weight for epistemic uncertainty (default: 0.5)
            lambda_ale: Weight for aleatoric uncertainty (default: 0.5)
            tau: Smoothing prior for aleatoric uncertainty
            scorer: TopologyScorer for computing topology scores
        """
        self.lambda_epi = lambda_epi
        self.lambda_ale = lambda_ale
        
        self.epistemic = EpistemicUncertainty(scorer=scorer)
        self.aleatoric = AleatoricUncertainty(tau=tau)
    
    def compute(
        self,
        graphs: List[TopologyGraph],
        correctness_probs: Optional[Dict[str, float]] = None,
        contradiction_scores: Optional[List[float]] = None
    ) -> UncertaintyResult:
        """
        Compute total uncertainty from graph samples.
        
        Based on Equation (3):
            u(G) = λ_epi * u_epi(G) + λ_ale * u_ale(G)
        
        Args:
            graphs: List of K re-elicited topology graphs
            correctness_probs: Dict mapping node_id to correctness probability
            contradiction_scores: List of contradiction scores for each graph
            
        Returns:
            UncertaintyResult with total, epistemic, and aleatoric components
        """
        if len(graphs) == 0:
            return UncertaintyResult(
                total=1.0,
                epistemic=1.0,
                aleatoric=1.0,
                components={"variance": 0.0, "jsd": 0.0, "node_entropy": 1.0}
            )
        
        # Compute epistemic uncertainty
        u_epi = self.epistemic.compute(graphs, contradiction_scores)
        
        # Compute aleatoric uncertainty (use first graph or average)
        u_ale = self.aleatoric.compute(graphs[0], correctness_probs)
        
        # Combined uncertainty: Equation (3)
        u_total = self.lambda_epi * u_epi + self.lambda_ale * u_ale
        
        return UncertaintyResult(
            total=u_total,
            epistemic=u_epi,
            aleatoric=u_ale,
            components={
                "epistemic": u_epi,
                "aleatoric": u_ale,
                "lambda_epi": self.lambda_epi,
                "lambda_ale": self.lambda_ale
            }
        )
    
    def compute_pair_weight(
        self,
        u_pos: float,
        u_neg: float,
        tau_w: float = 1.2,
        w_min: float = 0.05
    ) -> float:
        """
        Compute pair weight from uncertainties of positive and negative samples.
        
        Based on Equation (6):
            w = clip(τ_w / (1 + ū), w_min, 1)
            where ū = (u(G+) + u(G-)) / 2
        
        Args:
            u_pos: Uncertainty of positive (preferred) sample
            u_neg: Uncertainty of negative (dispreferred) sample
            tau_w: Temperature for weight mapping (default: 1.2)
            w_min: Minimum weight floor (default: 0.05)
            
        Returns:
            Pair weight in [w_min, 1.0]
        """
        # Average pair uncertainty
        u_bar = (u_pos + u_neg) / 2
        
        # Weight mapping: Equation (6)
        w = tau_w / (1 + u_bar)
        
        # Clip to valid range
        w = np.clip(w, w_min, 1.0)
        
        return w


class PairWeightComputer:
    """
    Compute per-pair weights for TUR-DPO training.
    
    Attenuates learning on high-uncertainty pairs while keeping a floor
    to avoid discarding data.
    """
    
    def __init__(
        self,
        tau_w: float = 1.2,
        w_min: float = 0.05,
        uncertainty_estimator: Optional[UncertaintyEstimator] = None
    ):
        """
        Initialize pair weight computer.
        
        Args:
            tau_w: Temperature for weight mapping
            w_min: Minimum weight floor
            uncertainty_estimator: UncertaintyEstimator instance
        """
        self.tau_w = tau_w
        self.w_min = w_min
        self.uncertainty_estimator = uncertainty_estimator or UncertaintyEstimator()
    
    def compute_weight(
        self,
        graphs_pos: List[TopologyGraph],
        graphs_neg: List[TopologyGraph],
        correctness_probs_pos: Optional[Dict[str, float]] = None,
        correctness_probs_neg: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute pair weight and uncertainty breakdown.
        
        Args:
            graphs_pos: Re-elicited graphs for positive (preferred) sample
            graphs_neg: Re-elicited graphs for negative (dispreferred) sample
            correctness_probs_pos: Node correctness probs for positive sample
            correctness_probs_neg: Node correctness probs for negative sample
            
        Returns:
            Tuple of (weight, uncertainty_dict)
        """
        # Compute uncertainties for both samples
        u_pos_result = self.uncertainty_estimator.compute(graphs_pos, correctness_probs_pos)
        u_neg_result = self.uncertainty_estimator.compute(graphs_neg, correctness_probs_neg)
        
        # Compute pair weight
        weight = self.uncertainty_estimator.compute_pair_weight(
            u_pos=u_pos_result.total,
            u_neg=u_neg_result.total,
            tau_w=self.tau_w,
            w_min=self.w_min
        )
        
        return weight, {
            "u_pos_total": u_pos_result.total,
            "u_neg_total": u_neg_result.total,
            "u_pos_epistemic": u_pos_result.epistemic,
            "u_neg_epistemic": u_neg_result.epistemic,
            "u_pos_aleatoric": u_pos_result.aleatoric,
            "u_neg_aleatoric": u_neg_result.aleatoric,
            "weight": weight
        }
