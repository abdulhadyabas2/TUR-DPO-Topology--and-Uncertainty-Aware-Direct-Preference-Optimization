"""
Calibration Module for TUR-DPO

This module implements calibration metrics and methods for uncertainty estimation.
Based on Section 2.4 of the paper:

Expected Calibration Error (ECE):
    ECE = Σ_m (|B_m|/N) * |p̄_m - ā_m|

Brier Score:
    Brier = (1/N) Σ_i (p_i - y_i)²

Temperature scaling and isotonic regression for verifier calibration.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Container for calibration metrics."""
    ece: float
    brier: float
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]


class CalibrationMetrics:
    """
    Compute calibration metrics for probabilistic predictions.
    """
    
    def __init__(self, num_bins: int = 10):
        """
        Initialize calibration metrics calculator.
        
        Args:
            num_bins: Number of bins for ECE computation (default: 10)
        """
        self.num_bins = num_bins
        self.bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    def compute_ece(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray
    ) -> Tuple[float, List[float], List[float], List[int]]:
        """
        Compute Expected Calibration Error.
        
        ECE = Σ_m (|B_m|/N) * |p̄_m - ā_m|
        
        Args:
            confidences: Predicted probabilities (N,)
            accuracies: Binary correctness labels (N,)
            
        Returns:
            Tuple of (ECE, bin_accuracies, bin_confidences, bin_counts)
        """
        n = len(confidences)
        if n == 0:
            return 0.0, [], [], []
        
        bin_accs = []
        bin_confs = []
        bin_counts = []
        ece = 0.0
        
        for i in range(self.num_bins):
            lower = self.bin_boundaries[i]
            upper = self.bin_boundaries[i + 1]
            
            # Find samples in this bin
            if i == self.num_bins - 1:
                # Include upper boundary for last bin
                mask = (confidences >= lower) & (confidences <= upper)
            else:
                mask = (confidences >= lower) & (confidences < upper)
            
            bin_size = mask.sum()
            bin_counts.append(int(bin_size))
            
            if bin_size > 0:
                bin_acc = accuracies[mask].mean()
                bin_conf = confidences[mask].mean()
                
                bin_accs.append(float(bin_acc))
                bin_confs.append(float(bin_conf))
                
                # Weighted contribution to ECE
                ece += (bin_size / n) * abs(bin_acc - bin_conf)
            else:
                bin_accs.append(0.0)
                bin_confs.append((lower + upper) / 2)
        
        return float(ece), bin_accs, bin_confs, bin_counts
    
    def compute_brier(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute Brier score (mean squared error).
        
        Brier = (1/N) Σ_i (p_i - y_i)²
        
        Args:
            probabilities: Predicted probabilities (N,)
            labels: True binary labels (N,)
            
        Returns:
            Brier score
        """
        if len(probabilities) == 0:
            return 0.0
        
        return float(np.mean((probabilities - labels) ** 2))
    
    def compute_all(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> CalibrationResult:
        """
        Compute all calibration metrics.
        
        Args:
            confidences: Predicted confidences for ECE
            accuracies: Binary accuracy indicators for ECE
            probabilities: Predicted probabilities for Brier (optional)
            labels: True labels for Brier (optional)
            
        Returns:
            CalibrationResult with all metrics
        """
        # Compute ECE
        ece, bin_accs, bin_confs, bin_counts = self.compute_ece(
            confidences, accuracies
        )
        
        # Compute Brier score if provided
        if probabilities is not None and labels is not None:
            brier = self.compute_brier(probabilities, labels)
        else:
            brier = self.compute_brier(confidences, accuracies)
        
        return CalibrationResult(
            ece=ece,
            brier=brier,
            bin_accuracies=bin_accs,
            bin_confidences=bin_confs,
            bin_counts=bin_counts
        )


class TemperatureScaler:
    """
    Temperature scaling for calibration.
    
    Calibrated probability: p_cal = σ(logit / T)
    """
    
    def __init__(self, initial_temperature: float = 1.0):
        """
        Initialize temperature scaler.
        
        Args:
            initial_temperature: Starting temperature value
        """
        self.temperature = initial_temperature
        self.fitted = False
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> float:
        """
        Fit temperature parameter using NLL minimization.
        
        Args:
            logits: Raw model logits (N,)
            labels: True binary labels (N,)
            lr: Learning rate for optimization
            max_iter: Maximum iterations
            
        Returns:
            Fitted temperature value
        """
        # Grid search for simplicity
        best_nll = float('inf')
        best_temp = 1.0
        
        for temp in np.linspace(0.1, 5.0, 50):
            scaled_probs = self._apply_temperature(logits, temp)
            nll = self._compute_nll(scaled_probs, labels)
            
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        
        self.temperature = best_temp
        self.fitted = True
        
        return self.temperature
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw logits (N,)
            
        Returns:
            Calibrated probabilities (N,)
        """
        return self._apply_temperature(logits, self.temperature)
    
    def _apply_temperature(
        self,
        logits: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        """Apply temperature scaling."""
        scaled = logits / temperature
        return 1 / (1 + np.exp(-scaled))
    
    def _compute_nll(
        self,
        probs: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute negative log-likelihood."""
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        return -np.mean(
            labels * np.log(probs) + 
            (1 - labels) * np.log(1 - probs)
        )


class IsotonicCalibrator:
    """
    Isotonic regression for calibration.
    
    Fits a piecewise-constant monotone function.
    """
    
    def __init__(self, y_min: float = 0.01, y_max: float = 0.99):
        """
        Initialize isotonic calibrator.
        
        Args:
            y_min: Minimum calibrated probability
            y_max: Maximum calibrated probability
        """
        self.y_min = y_min
        self.y_max = y_max
        self._isotonic = None
        self.fitted = False
    
    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> 'IsotonicCalibrator':
        """
        Fit isotonic regression.
        
        Args:
            scores: Uncalibrated scores (N,)
            labels: True binary labels (N,)
            
        Returns:
            Self
        """
        try:
            from sklearn.isotonic import IsotonicRegression
            
            self._isotonic = IsotonicRegression(
                y_min=self.y_min,
                y_max=self.y_max,
                out_of_bounds='clip'
            )
            self._isotonic.fit(scores, labels)
            self.fitted = True
        except ImportError:
            # Fallback to simple binning
            self._fit_binned(scores, labels)
            self.fitted = True
        
        return self
    
    def _fit_binned(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 20
    ) -> None:
        """Simple binned calibration as fallback."""
        self._bins = []
        bin_edges = np.linspace(0, 1, num_bins + 1)
        
        for i in range(num_bins):
            lower, upper = bin_edges[i], bin_edges[i + 1]
            mask = (scores >= lower) & (scores < upper)
            
            if mask.sum() > 0:
                bin_prob = np.clip(labels[mask].mean(), self.y_min, self.y_max)
            else:
                bin_prob = (lower + upper) / 2
            
            self._bins.append((lower, upper, bin_prob))
    
    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.
        
        Args:
            scores: Uncalibrated scores (N,)
            
        Returns:
            Calibrated probabilities (N,)
        """
        if not self.fitted:
            raise RuntimeError("Calibrator must be fitted before use")
        
        if self._isotonic is not None:
            return self._isotonic.predict(scores)
        else:
            # Use binned fallback
            calibrated = np.zeros_like(scores)
            for lower, upper, prob in self._bins:
                mask = (scores >= lower) & (scores < upper)
                calibrated[mask] = prob
            return calibrated


class ReliabilityDiagram:
    """
    Create reliability diagram data for visualization.
    """
    
    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins
        self.metrics = CalibrationMetrics(num_bins)
    
    def compute(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Compute reliability diagram data.
        
        Args:
            confidences: Predicted confidences (N,)
            accuracies: True accuracies (N,)
            
        Returns:
            Dict with 'bin_centers', 'accuracies', 'confidences', 'counts'
        """
        result = self.metrics.compute_all(confidences, accuracies)
        
        bin_centers = [
            (self.metrics.bin_boundaries[i] + self.metrics.bin_boundaries[i + 1]) / 2
            for i in range(self.num_bins)
        ]
        
        return {
            'bin_centers': bin_centers,
            'accuracies': result.bin_accuracies,
            'confidences': result.bin_confidences,
            'counts': result.bin_counts,
            'ece': result.ece
        }


def compute_per_task_calibration(
    predictions: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    num_bins: int = 10
) -> Dict[str, CalibrationResult]:
    """
    Compute calibration metrics for multiple tasks.
    
    Args:
        predictions: Dict mapping task name to predictions
        labels: Dict mapping task name to labels
        num_bins: Number of bins for ECE
        
    Returns:
        Dict mapping task name to CalibrationResult
    """
    metrics = CalibrationMetrics(num_bins)
    results = {}
    
    for task in predictions.keys():
        if task in labels:
            preds = predictions[task]
            labs = labels[task]
            
            # Compute calibration
            result = metrics.compute_all(
                confidences=preds,
                accuracies=labs
            )
            results[task] = result
    
    return results


def aggregate_calibration(
    task_results: Dict[str, CalibrationResult],
    weights: Optional[Dict[str, float]] = None
) -> CalibrationResult:
    """
    Aggregate calibration results across tasks.
    
    Args:
        task_results: Dict of per-task CalibrationResults
        weights: Optional task weights (default: uniform)
        
    Returns:
        Aggregated CalibrationResult
    """
    if not task_results:
        return CalibrationResult(
            ece=0.0,
            brier=0.0,
            bin_accuracies=[],
            bin_confidences=[],
            bin_counts=[]
        )
    
    if weights is None:
        weights = {task: 1.0 for task in task_results}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Weighted average of ECE and Brier
    ece = sum(weights.get(task, 0) * result.ece 
              for task, result in task_results.items())
    brier = sum(weights.get(task, 0) * result.brier 
                for task, result in task_results.items())
    
    return CalibrationResult(
        ece=ece,
        brier=brier,
        bin_accuracies=[],
        bin_confidences=[],
        bin_counts=[]
    )
