"""
Trainer Module for TUR-DPO

This module implements the complete training pipeline for TUR-DPO.
Based on the training protocol described in Section 2 of the paper:

1. Elicit graphs for positive and negative candidates
2. Compute semantic and topology scores
3. Compute epistemic and aleatoric uncertainties
4. Map to pair weight
5. Update calibrator parameters (optional)
6. Update policy parameters
7. Optionally update reference by EMA
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
import logging

from .topology import TopologyExtractor, TopologyScorer, TopologyGraph
from .uncertainty import UncertaintyEstimator, PairWeightComputer
from .rewards import ShapedReward, RewardDifferenceComputer
from .loss import TURDPOLoss, ListwiseTURDPOLoss
from .verifier import NodeVerifier, ContradictionDetector

logger = logging.getLogger(__name__)


@dataclass
class TURDPOConfig:
    """Configuration for TUR-DPO training."""
    
    # Temperature and reward parameters
    beta: float = 2.0  # DPO temperature
    gamma: float = 1.0  # Reward difference weight
    
    # Shaped reward parameters
    a: float = 0.6  # Semantic vs topology mixing
    lambda_uncertainty: float = 0.5  # Uncertainty penalty in reward
    
    # Topology scoring weights (Equation 1)
    alpha_path: float = 1.0
    alpha_cycle: float = 0.5
    alpha_dangling: float = 0.3
    alpha_contradict: float = 0.4
    
    # Uncertainty estimation
    lambda_epi: float = 0.5  # Epistemic weight
    lambda_ale: float = 0.5  # Aleatoric weight
    tau_smoothing: float = 0.05  # Smoothing prior for aleatoric
    
    # Pair weighting
    tau_w: float = 1.2  # Weight mapping temperature
    w_min: float = 0.05  # Minimum weight floor
    
    # Graph re-elicitation
    k_samples: int = 3  # Number of re-elicited graphs
    
    # Reference policy
    use_ema_reference: bool = True
    ema_decay: float = 0.995  # EMA decay ρ
    
    # Training
    learning_rate: float = 1e-6
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 2000
    logging_steps: int = 100
    
    # Calibrator training
    train_calibrators: bool = True
    calibrator_lr: float = 1e-4
    
    # Listwise training
    use_listwise: bool = False
    num_candidates: int = 4


@dataclass
class TrainingState:
    """Training state for checkpointing."""
    step: int = 0
    epoch: int = 0
    best_metric: float = 0.0
    metrics_history: List[Dict[str, float]] = field(default_factory=list)


class TURDPOTrainer:
    """
    Complete TUR-DPO training pipeline.
    
    Implements the training protocol from the paper:
    1. Graph elicitation with perturbations
    2. Score computation (semantic, topology)
    3. Uncertainty estimation and pair weighting
    4. Loss computation with shaped reward augmentation
    5. Optional EMA reference update
    """
    
    def __init__(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        tokenizer,
        config: TURDPOConfig,
        topology_extractor: Optional[TopologyExtractor] = None,
        verifier: Optional[NodeVerifier] = None,
        device: str = "cuda"
    ):
        """
        Initialize TUR-DPO trainer.
        
        Args:
            model: Policy model to train
            reference_model: Reference policy (frozen or EMA-updated)
            tokenizer: Tokenizer for the model
            config: Training configuration
            topology_extractor: Extractor for reasoning graphs
            verifier: Verifier for node correctness
            device: Device to use for training
        """
        self.model = model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Initialize components
        self.topology_extractor = topology_extractor or TopologyExtractor()
        self.topology_scorer = TopologyScorer(
            alpha_path=config.alpha_path,
            alpha_cycle=config.alpha_cycle,
            alpha_dangling=config.alpha_dangling,
            alpha_contradict=config.alpha_contradict
        )
        
        self.uncertainty_estimator = UncertaintyEstimator(
            lambda_epi=config.lambda_epi,
            lambda_ale=config.lambda_ale,
            tau=config.tau_smoothing,
            scorer=self.topology_scorer
        )
        
        self.pair_weight_computer = PairWeightComputer(
            tau_w=config.tau_w,
            w_min=config.w_min,
            uncertainty_estimator=self.uncertainty_estimator
        )
        
        self.shaped_reward = ShapedReward(
            a=config.a,
            lambda_uncertainty=config.lambda_uncertainty
        )
        
        self.reward_computer = RewardDifferenceComputer(
            shaped_reward=self.shaped_reward,
            gamma=config.gamma
        )
        
        self.verifier = verifier or NodeVerifier()
        self.contradiction_detector = ContradictionDetector()
        
        # Loss functions
        if config.use_listwise:
            self.loss_fn = ListwiseTURDPOLoss(
                beta=config.beta,
                gamma=config.gamma,
                num_candidates=config.num_candidates
            )
        else:
            self.loss_fn = TURDPOLoss(
                beta=config.beta,
                gamma=config.gamma
            )
        
        # Move models to device
        self.model.to(device)
        self.reference_model.to(device)
        self.reference_model.eval()
        
        # Training state
        self.state = TrainingState()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup AdamW optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            return 1.0
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 1
    ) -> Dict[str, Any]:
        """
        Run TUR-DPO training.
        
        Args:
            train_dataloader: DataLoader for training data
            eval_dataloader: Optional DataLoader for evaluation
            num_epochs: Number of training epochs
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting TUR-DPO training...")
        logger.info(f"Config: beta={self.config.beta}, gamma={self.config.gamma}, "
                   f"a={self.config.a}, lambda_u={self.config.lambda_uncertainty}")
        
        self.model.train()
        
        for epoch in range(num_epochs):
            self.state.epoch = epoch
            epoch_metrics = self._train_epoch(train_dataloader)
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs} completed. "
                       f"Loss: {epoch_metrics['loss']:.4f}")
            
            # Evaluation
            if eval_dataloader is not None and (epoch + 1) % 1 == 0:
                eval_metrics = self.evaluate(eval_dataloader)
                logger.info(f"Eval metrics: {eval_metrics}")
                
                # Track best model
                if eval_metrics.get('accuracy', 0) > self.state.best_metric:
                    self.state.best_metric = eval_metrics['accuracy']
        
        return {
            "final_loss": epoch_metrics['loss'],
            "best_metric": self.state.best_metric,
            "total_steps": self.state.step,
            "metrics_history": self.state.metrics_history
        }
    
    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.state.epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            metrics = self._train_step(batch)
            
            total_loss += metrics['loss']
            num_batches += 1
            self.state.step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': metrics['loss'],
                'weight': metrics.get('weights', 1.0),
                'reward_diff': metrics.get('reward_diff', 0.0)
            })
            
            # Logging
            if self.state.step % self.config.logging_steps == 0:
                self.state.metrics_history.append(metrics)
            
            # EMA reference update
            if self.config.use_ema_reference:
                self._update_ema_reference()
        
        return {'loss': total_loss / max(num_batches, 1)}
    
    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single training step implementing the TUR-DPO protocol.
        
        Protocol:
        1. Elicit graphs for positive and negative candidates
        2. Compute semantic and topology scores
        3. Compute uncertainties and pair weight
        4. Compute loss with shaped reward
        5. Update model
        """
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = self._prepare_batch(batch)
        
        # Step 1: Elicit graphs
        graphs_pos, graphs_neg = self._elicit_graphs(batch)
        
        # Step 2: Compute scores
        topo_scores_pos, topo_scores_neg = self._compute_topology_scores(
            graphs_pos, graphs_neg
        )
        sem_scores_pos, sem_scores_neg = self._compute_semantic_scores(
            batch, graphs_pos, graphs_neg
        )
        
        # Step 3: Compute uncertainties and weights
        weights, uncertainties = self._compute_weights(
            graphs_pos, graphs_neg
        )
        
        # Step 4: Compute reward differences
        reward_diffs = self._compute_reward_differences(
            sem_scores_pos, sem_scores_neg,
            topo_scores_pos, topo_scores_neg,
            uncertainties
        )
        
        # Step 5: Forward pass and loss computation
        loss, metrics = self._compute_loss(batch, reward_diffs, weights)
        
        # Step 6: Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
        
        # Step 7: Update
        self.optimizer.step()
        self.scheduler.step()
        
        return metrics
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        prepared = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value
        return prepared
    
    def _elicit_graphs(
        self,
        batch: Dict[str, Any]
    ) -> Tuple[List[List[TopologyGraph]], List[List[TopologyGraph]]]:
        """
        Elicit topology graphs for positive and negative responses.
        
        Returns K re-elicited graphs per response for uncertainty estimation.
        """
        prompts = batch.get('prompts', [])
        chosen_responses = batch.get('chosen_responses', [])
        rejected_responses = batch.get('rejected_responses', [])
        
        graphs_pos = []
        graphs_neg = []
        
        for prompt, chosen, rejected in zip(prompts, chosen_responses, rejected_responses):
            # Extract K graphs for each response
            pos_graphs = self.topology_extractor.extract_multiple(
                prompt=prompt,
                response=chosen,
                k=self.config.k_samples
            )
            neg_graphs = self.topology_extractor.extract_multiple(
                prompt=prompt,
                response=rejected,
                k=self.config.k_samples
            )
            
            graphs_pos.append(pos_graphs)
            graphs_neg.append(neg_graphs)
        
        return graphs_pos, graphs_neg
    
    def _compute_topology_scores(
        self,
        graphs_pos: List[List[TopologyGraph]],
        graphs_neg: List[List[TopologyGraph]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute topology scores for all graphs."""
        batch_size = len(graphs_pos)
        
        scores_pos = []
        scores_neg = []
        
        for pos_graphs, neg_graphs in zip(graphs_pos, graphs_neg):
            # Use first graph for score (or average over K)
            if pos_graphs:
                # Compute contradiction score
                contra_score, _ = self.contradiction_detector.detect_contradictions(pos_graphs[0])
                score = self.topology_scorer.compute_score(pos_graphs[0], contra_score)
            else:
                score = 0.5
            scores_pos.append(score)
            
            if neg_graphs:
                contra_score, _ = self.contradiction_detector.detect_contradictions(neg_graphs[0])
                score = self.topology_scorer.compute_score(neg_graphs[0], contra_score)
            else:
                score = 0.5
            scores_neg.append(score)
        
        return (
            torch.tensor(scores_pos, device=self.device),
            torch.tensor(scores_neg, device=self.device)
        )
    
    def _compute_semantic_scores(
        self,
        batch: Dict[str, Any],
        graphs_pos: List[List[TopologyGraph]],
        graphs_neg: List[List[TopologyGraph]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute semantic scores for responses."""
        # Use task-specific scores if provided
        task_scores_pos = batch.get('task_scores_chosen', None)
        task_scores_neg = batch.get('task_scores_rejected', None)
        
        if task_scores_pos is not None:
            return task_scores_pos, task_scores_neg
        
        # Compute from graphs
        batch_size = len(graphs_pos)
        scores_pos = []
        scores_neg = []
        
        for i, (pos_graphs, neg_graphs) in enumerate(zip(graphs_pos, graphs_neg)):
            # Use node correctness as factuality proxy
            if pos_graphs:
                probs = [n.correctness_prob for n in pos_graphs[0].nodes.values()]
                score = np.mean(probs) if probs else 0.5
            else:
                score = 0.5
            scores_pos.append(score)
            
            if neg_graphs:
                probs = [n.correctness_prob for n in neg_graphs[0].nodes.values()]
                score = np.mean(probs) if probs else 0.5
            else:
                score = 0.5
            scores_neg.append(score)
        
        return (
            torch.tensor(scores_pos, device=self.device),
            torch.tensor(scores_neg, device=self.device)
        )
    
    def _compute_weights(
        self,
        graphs_pos: List[List[TopologyGraph]],
        graphs_neg: List[List[TopologyGraph]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute pair weights from uncertainties."""
        weights = []
        u_pos_list = []
        u_neg_list = []
        
        for pos_graphs, neg_graphs in zip(graphs_pos, graphs_neg):
            weight, u_dict = self.pair_weight_computer.compute_weight(
                graphs_pos=pos_graphs,
                graphs_neg=neg_graphs
            )
            weights.append(weight)
            u_pos_list.append(u_dict['u_pos_total'])
            u_neg_list.append(u_dict['u_neg_total'])
        
        return (
            torch.tensor(weights, device=self.device),
            {
                'u_pos': torch.tensor(u_pos_list, device=self.device),
                'u_neg': torch.tensor(u_neg_list, device=self.device)
            }
        )
    
    def _compute_reward_differences(
        self,
        sem_pos: torch.Tensor,
        sem_neg: torch.Tensor,
        topo_pos: torch.Tensor,
        topo_neg: torch.Tensor,
        uncertainties: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute shaped reward differences."""
        batch_size = sem_pos.shape[0]
        reward_diffs = []
        
        for i in range(batch_size):
            delta_r, _ = self.reward_computer.compute(
                sem_score_pos=sem_pos[i].item(),
                sem_score_neg=sem_neg[i].item(),
                topo_score_pos=topo_pos[i].item(),
                topo_score_neg=topo_neg[i].item(),
                uncertainty_pos=uncertainties['u_pos'][i].item(),
                uncertainty_neg=uncertainties['u_neg'][i].item()
            )
            reward_diffs.append(delta_r)
        
        return torch.tensor(reward_diffs, device=self.device)
    
    def _compute_loss(
        self,
        batch: Dict[str, Any],
        reward_diffs: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute TUR-DPO loss."""
        # Get model outputs
        with torch.no_grad():
            ref_chosen_logps = self._compute_logps(
                self.reference_model,
                batch['chosen_input_ids'],
                batch['chosen_attention_mask'],
                batch.get('chosen_labels')
            )
            ref_rejected_logps = self._compute_logps(
                self.reference_model,
                batch['rejected_input_ids'],
                batch['rejected_attention_mask'],
                batch.get('rejected_labels')
            )
        
        policy_chosen_logps = self._compute_logps(
            self.model,
            batch['chosen_input_ids'],
            batch['chosen_attention_mask'],
            batch.get('chosen_labels')
        )
        policy_rejected_logps = self._compute_logps(
            self.model,
            batch['rejected_input_ids'],
            batch['rejected_attention_mask'],
            batch.get('rejected_labels')
        )
        
        # Compute loss
        loss, metrics = self.loss_fn(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=ref_chosen_logps,
            reference_rejected_logps=ref_rejected_logps,
            reward_diff=reward_diffs,
            weights=weights
        )
        
        # Convert metrics to Python floats
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v 
                  for k, v in metrics.items()}
        
        return loss, metrics
    
    def _compute_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute log probabilities for sequences."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        if labels is None:
            labels = input_ids
        
        return self.loss_fn.compute_logps(logits, labels, attention_mask)
    
    def _update_ema_reference(self) -> None:
        """Update reference model with exponential moving average."""
        decay = self.config.ema_decay
        
        with torch.no_grad():
            for param, ref_param in zip(
                self.model.parameters(),
                self.reference_model.parameters()
            ):
                ref_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation data."""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = self._prepare_batch(batch)
                
                # Simplified evaluation without full graph elicitation
                ref_chosen_logps = self._compute_logps(
                    self.reference_model,
                    batch['chosen_input_ids'],
                    batch['chosen_attention_mask'],
                    batch.get('chosen_labels')
                )
                ref_rejected_logps = self._compute_logps(
                    self.reference_model,
                    batch['rejected_input_ids'],
                    batch['rejected_attention_mask'],
                    batch.get('rejected_labels')
                )
                
                policy_chosen_logps = self._compute_logps(
                    self.model,
                    batch['chosen_input_ids'],
                    batch['chosen_attention_mask'],
                    batch.get('chosen_labels')
                )
                policy_rejected_logps = self._compute_logps(
                    self.model,
                    batch['rejected_input_ids'],
                    batch['rejected_attention_mask'],
                    batch.get('rejected_labels')
                )
                
                # Compute accuracy
                chosen_rewards = self.config.beta * (policy_chosen_logps - ref_chosen_logps)
                rejected_rewards = self.config.beta * (policy_rejected_logps - ref_rejected_logps)
                accuracy = (chosen_rewards > rejected_rewards).float().mean()
                
                total_accuracy += accuracy.item()
                num_batches += 1
        
        self.model.train()
        
        return {
            'accuracy': total_accuracy / max(num_batches, 1),
            'num_batches': num_batches
        }
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'reference_state_dict': self.reference_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'state': self.state,
            'config': self.config
        }, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.reference_model.load_state_dict(checkpoint['reference_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.state = checkpoint['state']
        
        logger.info(f"Checkpoint loaded from {path}")
