"""
Rewards Module for TUR-DPO

This module implements the shaped reward computation combining semantic, topology,
and uncertainty signals.

Based on Equations (7) and (8) from the paper:

Shaped reward (Eq. 7):
    r_φ(x, y, G) = a * f^sem_φ(s_sem) + (1-a) * f^topo_φ(s_topo) - λ * u(G)

Linear calibrators (Eq. 8):
    f^sem_φ(z) = γ_sem * z + b_sem
    f^topo_φ(z) = γ_topo * z + b_topo
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class RewardComponents:
    """Container for reward computation components."""
    total_reward: float
    semantic_component: float
    topology_component: float
    uncertainty_penalty: float
    raw_semantic_score: float
    raw_topology_score: float
    raw_uncertainty: float


class LinearCalibrator:
    """
    Linear calibrator for score transformation.
    
    Based on Equation (8):
        f_φ(z) = γ * z + b
    
    Provides monotonic transformation with learnable parameters.
    """
    
    def __init__(
        self,
        gamma: float = 1.0,
        bias: float = 0.0,
        requires_grad: bool = True
    ):
        """
        Initialize linear calibrator.
        
        Args:
            gamma: Scale parameter (slope)
            bias: Bias parameter (intercept)
            requires_grad: Whether parameters are learnable
        """
        self.gamma = gamma
        self.bias = bias
        self.requires_grad = requires_grad
    
    def forward(self, z: float) -> float:
        """Apply linear transformation."""
        return self.gamma * z + self.bias
    
    def __call__(self, z: float) -> float:
        return self.forward(z)
    
    def get_params(self) -> Dict[str, float]:
        return {"gamma": self.gamma, "bias": self.bias}
    
    def set_params(self, gamma: float, bias: float) -> None:
        self.gamma = gamma
        self.bias = bias


class SemanticScorer:
    """
    Compute semantic score balancing task success, factuality, and hallucination.
    
    Based on Equation (2):
        s_sem(x, y) = β₁ * q_fact + β₂ * q_task - β₃ * q_hall
    """
    
    def __init__(
        self,
        beta_fact: float = 0.4,
        beta_task: float = 0.4,
        beta_hall: float = 0.2,
        verifier=None
    ):
        """
        Initialize semantic scorer.
        
        Args:
            beta_fact: Weight for factuality score
            beta_task: Weight for task-specific metric
            beta_hall: Weight for hallucination penalty
            verifier: Optional verifier for fact checking
        """
        self.beta_fact = beta_fact
        self.beta_task = beta_task
        self.beta_hall = beta_hall
        self.verifier = verifier
    
    def compute(
        self,
        prompt: str,
        response: str,
        graph=None,
        task_score: Optional[float] = None,
        fact_scores: Optional[Dict[str, float]] = None,
        hallucination_score: Optional[float] = None
    ) -> float:
        """
        Compute semantic score for a response.
        
        Args:
            prompt: Input prompt
            response: Model response
            graph: Topology graph (for node-level scoring)
            task_score: Pre-computed task metric (e.g., exact match, ROUGE)
            fact_scores: Dict of fact scores per node
            hallucination_score: Pre-computed hallucination penalty
            
        Returns:
            Semantic score (higher is better)
        """
        # Get factuality score
        q_fact = self._compute_factuality(graph, fact_scores)
        
        # Get task score
        q_task = task_score if task_score is not None else 0.5
        
        # Get hallucination score
        q_hall = hallucination_score if hallucination_score is not None else 0.0
        
        # Apply Equation (2)
        score = (
            self.beta_fact * q_fact
            + self.beta_task * q_task
            - self.beta_hall * q_hall
        )
        
        return score
    
    def _compute_factuality(
        self,
        graph,
        fact_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """Aggregate factuality from node-level scores."""
        if graph is None or len(graph.nodes) == 0:
            return 0.5
        
        if fact_scores is None:
            # Use node correctness probabilities
            scores = [node.correctness_prob for node in graph.nodes.values()]
        else:
            scores = [fact_scores.get(node_id, 0.5) for node_id in graph.nodes]
        
        return np.mean(scores)


class ShapedReward:
    """
    Shaped reward combining semantic, topology, and uncertainty signals.
    
    Based on Equation (7):
        r_φ(x, y, G) = a * f^sem_φ(s_sem) + (1-a) * f^topo_φ(s_topo) - λ * u(G)
    """
    
    def __init__(
        self,
        a: float = 0.6,
        lambda_uncertainty: float = 0.5,
        semantic_calibrator: Optional[LinearCalibrator] = None,
        topology_calibrator: Optional[LinearCalibrator] = None,
        semantic_scorer: Optional[SemanticScorer] = None
    ):
        """
        Initialize shaped reward.
        
        Args:
            a: Mixing parameter between semantic (a) and topology (1-a)
            lambda_uncertainty: Weight for uncertainty penalty
            semantic_calibrator: Linear calibrator for semantic scores
            topology_calibrator: Linear calibrator for topology scores
            semantic_scorer: SemanticScorer instance
        """
        self.a = a
        self.lambda_uncertainty = lambda_uncertainty
        
        self.sem_calibrator = semantic_calibrator or LinearCalibrator(
            gamma=1.0, bias=0.0
        )
        self.topo_calibrator = topology_calibrator or LinearCalibrator(
            gamma=1.0, bias=0.0
        )
        self.semantic_scorer = semantic_scorer or SemanticScorer()
    
    def compute(
        self,
        semantic_score: float,
        topology_score: float,
        uncertainty: float
    ) -> RewardComponents:
        """
        Compute shaped reward from pre-computed scores.
        
        Based on Equation (7):
            r_φ = a * f^sem(s_sem) + (1-a) * f^topo(s_topo) - λ * u(G)
        
        Args:
            semantic_score: Raw semantic score s_sem
            topology_score: Raw topology score s_topo
            uncertainty: Total uncertainty u(G)
            
        Returns:
            RewardComponents with total reward and breakdown
        """
        # Apply calibrators
        calibrated_sem = self.sem_calibrator(semantic_score)
        calibrated_topo = self.topo_calibrator(topology_score)
        
        # Compute weighted components
        semantic_component = self.a * calibrated_sem
        topology_component = (1 - self.a) * calibrated_topo
        uncertainty_penalty = self.lambda_uncertainty * uncertainty
        
        # Total reward: Equation (7)
        total_reward = semantic_component + topology_component - uncertainty_penalty
        
        return RewardComponents(
            total_reward=total_reward,
            semantic_component=semantic_component,
            topology_component=topology_component,
            uncertainty_penalty=uncertainty_penalty,
            raw_semantic_score=semantic_score,
            raw_topology_score=topology_score,
            raw_uncertainty=uncertainty
        )
    
    def compute_from_inputs(
        self,
        prompt: str,
        response: str,
        graph,
        uncertainty: float,
        task_score: Optional[float] = None,
        topology_score: Optional[float] = None
    ) -> RewardComponents:
        """
        Compute shaped reward from raw inputs.
        
        Args:
            prompt: Input prompt
            response: Model response
            graph: Topology graph
            uncertainty: Total uncertainty
            task_score: Optional task-specific score
            topology_score: Pre-computed topology score (if available)
            
        Returns:
            RewardComponents with total reward and breakdown
        """
        # Compute semantic score if not provided
        semantic_score = self.semantic_scorer.compute(
            prompt=prompt,
            response=response,
            graph=graph,
            task_score=task_score
        )
        
        # Use provided topology score or default
        if topology_score is None:
            topology_score = 0.5
        
        return self.compute(
            semantic_score=semantic_score,
            topology_score=topology_score,
            uncertainty=uncertainty
        )
    
    def compute_reward_difference(
        self,
        reward_pos: RewardComponents,
        reward_neg: RewardComponents
    ) -> float:
        """
        Compute reward difference for preference pair.
        
        Δr_φ = r_φ(x, y+, G+) - r_φ(x, y-, G-)
        
        Args:
            reward_pos: Reward for preferred response
            reward_neg: Reward for dispreferred response
            
        Returns:
            Reward difference Δr_φ
        """
        return reward_pos.total_reward - reward_neg.total_reward
    
    def get_params(self) -> Dict[str, Any]:
        """Get all parameters."""
        return {
            "a": self.a,
            "lambda_uncertainty": self.lambda_uncertainty,
            "sem_calibrator": self.sem_calibrator.get_params(),
            "topo_calibrator": self.topo_calibrator.get_params()
        }
    
    def set_mixing_param(self, a: float) -> None:
        """Set the semantic/topology mixing parameter."""
        self.a = np.clip(a, 0.0, 1.0)
    
    def set_uncertainty_weight(self, lambda_u: float) -> None:
        """Set the uncertainty penalty weight."""
        self.lambda_uncertainty = max(0.0, lambda_u)


class RewardDifferenceComputer:
    """
    Compute reward differences for preference pairs.
    
    Used in the TUR-DPO loss to augment the DPO margin.
    """
    
    def __init__(
        self,
        shaped_reward: Optional[ShapedReward] = None,
        gamma: float = 1.0
    ):
        """
        Initialize reward difference computer.
        
        Args:
            shaped_reward: ShapedReward instance
            gamma: Scaling factor for reward difference in loss
        """
        self.shaped_reward = shaped_reward or ShapedReward()
        self.gamma = gamma
    
    def compute(
        self,
        sem_score_pos: float,
        sem_score_neg: float,
        topo_score_pos: float,
        topo_score_neg: float,
        uncertainty_pos: float,
        uncertainty_neg: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute scaled reward difference for loss computation.
        
        Returns γ * Δr_φ for use in TUR-DPO loss.
        
        Args:
            sem_score_pos: Semantic score for preferred response
            sem_score_neg: Semantic score for dispreferred response
            topo_score_pos: Topology score for preferred response
            topo_score_neg: Topology score for dispreferred response
            uncertainty_pos: Uncertainty for preferred response
            uncertainty_neg: Uncertainty for dispreferred response
            
        Returns:
            Tuple of (gamma * delta_reward, components_dict)
        """
        # Compute rewards for both samples
        reward_pos = self.shaped_reward.compute(
            semantic_score=sem_score_pos,
            topology_score=topo_score_pos,
            uncertainty=uncertainty_pos
        )
        
        reward_neg = self.shaped_reward.compute(
            semantic_score=sem_score_neg,
            topology_score=topo_score_neg,
            uncertainty=uncertainty_neg
        )
        
        # Reward difference
        delta_reward = self.shaped_reward.compute_reward_difference(reward_pos, reward_neg)
        
        # Scaled for loss
        scaled_delta = self.gamma * delta_reward
        
        return scaled_delta, {
            "delta_reward": delta_reward,
            "gamma_delta_reward": scaled_delta,
            "reward_pos": reward_pos.total_reward,
            "reward_neg": reward_neg.total_reward,
            "sem_component_pos": reward_pos.semantic_component,
            "sem_component_neg": reward_neg.semantic_component,
            "topo_component_pos": reward_pos.topology_component,
            "topo_component_neg": reward_neg.topology_component,
        }
