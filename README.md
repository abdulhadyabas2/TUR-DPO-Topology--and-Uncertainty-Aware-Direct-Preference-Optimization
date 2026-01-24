# TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/xxxx.xxxxx)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official implementation of **TUR-DPO: Structure- and Uncertainty-Aware Direct Preference Optimization**.

## Overview

TUR-DPO is a topology- and uncertainty-aware extension of Direct Preference Optimization (DPO) that rewards **how answers are derived**, not only what they say. For each candidate response, TUR-DPO:

1. Elicits a lightweight **reasoning topology** (a graph of sub-claims and support relations)
2. Computes dual signals: **semantic faithfulness/utility** and **topology quality**
3. Derives a **calibrated uncertainty score** from graph- and node-level evidence
4. Uses these signals to shape the DPO loss with uncertainty-weighted pairs

## Key Features

- 🔧 **RL-free**: No online rollouts, value heads, or reward model training
- 📊 **Structure-aware**: Rewards coherent multi-step reasoning via topology scoring
- 🎯 **Uncertainty-weighted**: Down-weights noisy/brittle preference pairs
- ⚡ **Efficient**: Adds only ~15% overhead compared to vanilla DPO
- 🔄 **Compatible**: Works with any DPO-style codebase and dataset

## Installation

```bash
git clone https://github.com/yourusername/turdpo.git
cd turdpo
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- transformers >= 4.30
- numpy, scipy, tqdm

## Quick Start

### Basic Training

```python
from turdpo.trainer import TURDPOTrainer, TURDPOConfig
from turdpo.data import PreferenceDataset, create_dataloader
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
reference_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure TUR-DPO
config = TURDPOConfig(
    beta=2.0,           # DPO temperature
    gamma=1.0,          # Shaped reward weight
    a=0.6,              # Semantic vs topology mixing
    lambda_uncertainty=0.5,  # Uncertainty penalty
    tau_w=1.2,          # Weight mapping temperature
    w_min=0.05,         # Minimum pair weight
    k_samples=3,        # Graph re-elicitation samples
    use_ema_reference=True,
    ema_decay=0.995
)

# Load data and create trainer
dataset = PreferenceDataset.from_json("data/preferences.json", tokenizer)
dataloader = create_dataloader(dataset, batch_size=4)

trainer = TURDPOTrainer(
    model=model,
    reference_model=reference_model,
    tokenizer=tokenizer,
    config=config
)

# Train
results = trainer.train(dataloader, num_epochs=1)
```

### Command Line Training

```bash
python train.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --train_data data/train.json \
    --beta 2.0 \
    --gamma 1.0 \
    --a 0.6 \
    --lambda_uncertainty 0.5 \
    --learning_rate 1e-6 \
    --batch_size 4 \
    --num_epochs 1 \
    --output_dir outputs/
```

## Algorithm Overview

### Mathematical Formulation

**Topology Score** (Equation 1):
```
s_topo(G) = α₁·q_path - α₂·c_cycle - α₃·d_dangling - α₄·q_contradict
```

**Semantic Score** (Equation 2):
```
s_sem(x,y) = β₁·q_fact + β₂·q_task - β₃·q_hall
```

**Total Uncertainty** (Equation 3):
```
u(G) = λ_epi·u_epi(G) + λ_ale·u_ale(G)
```

**Shaped Reward** (Equation 7):
```
r_φ(x,y,G) = a·f_sem(s_sem) + (1-a)·f_topo(s_topo) - λ·u(G)
```

**TUR-DPO Loss** (Equation 9):
```
L = -w · log σ(β·[Δlog π_θ - Δlog π_ref] + γ·Δr_φ)
```

### Training Protocol

1. **Graph Elicitation**: Extract K reasoning topologies per response with perturbations
2. **Score Computation**: Compute semantic and topology scores
3. **Uncertainty Estimation**: Epistemic (graph variance) + Aleatoric (node entropy)
4. **Weight Computation**: Map uncertainty to per-pair weight
5. **Loss Computation**: DPO loss with shaped reward augmentation
6. **Reference Update**: Optional EMA update

## Project Structure

```
turdpo/
├── turdpo/
│   ├── __init__.py         # Package exports
│   ├── topology.py         # Graph extraction and scoring
│   ├── uncertainty.py      # Uncertainty estimation
│   ├── rewards.py          # Shaped reward computation
│   ├── loss.py             # TUR-DPO loss functions
│   ├── trainer.py          # Main training loop
│   ├── verifier.py         # Node verification
│   ├── calibration.py      # Calibration metrics
│   ├── data.py             # Data loading utilities
│   └── utils.py            # General utilities
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── configs/                # Configuration files
└── README.md
```

## Data Format

Preference pairs should be in JSON/JSONL format:

```json
{
  "prompt": "What is 15 + 27?",
  "chosen": "To solve 15 + 27, I'll add the ones: 5 + 7 = 12. Then add the tens: 10 + 20 = 30. Total: 30 + 12 = 42.",
  "rejected": "15 + 27 = 43",
  "task_score_chosen": 1.0,
  "task_score_rejected": 0.0
}
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | 2.0 | DPO temperature (controls sharpness) |
| `gamma` | 1.0 | Shaped reward weight in loss |
| `a` | 0.6 | Semantic vs topology mixing (0=all topo, 1=all sem) |
| `lambda_uncertainty` | 0.5 | Uncertainty penalty in shaped reward |
| `lambda_epi` | 0.5 | Weight for epistemic uncertainty |
| `lambda_ale` | 0.5 | Weight for aleatoric uncertainty |
| `tau_w` | 1.2 | Temperature for weight mapping |
| `w_min` | 0.05 | Minimum pair weight floor |
| `k_samples` | 3 | Number of re-elicited graphs |
| `ema_decay` | 0.995 | EMA decay for reference policy |

## Results

TUR-DPO achieves consistent improvements over DPO across multiple benchmarks:

| Task | DPO | TUR-DPO | Δ |
|------|-----|---------|---|
| GSM8K (EM%) | 58.7 | **62.8** | +4.1 |
| MATH (EM%) | 33.4 | **36.0** | +2.6 |
| BBH (Acc%) | 43.9 | **46.7** | +2.8 |
| Open QA (EM) | 41.8 | **45.1** | +3.3 |
| TLDR (Win%) | 61.2 | **64.8** | +3.6 |
| HH (Win%) | 65.5 | **67.9** | +2.4 |

## Citation

```bibtex
@article{turdpo2025,
  title={TUR-DPO: Structure- and Uncertainty-Aware Direct Preference Optimization},
  author={...},
  journal={...},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The TUR-DPO method builds on [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)
- Graph extraction techniques inspired by [reasoning structure research](...)
- Uncertainty estimation draws from [semantic entropy work](...)
