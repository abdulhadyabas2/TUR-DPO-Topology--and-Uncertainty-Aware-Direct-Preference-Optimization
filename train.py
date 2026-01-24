#!/usr/bin/env python
"""
TUR-DPO Training Script

This script provides the main entry point for training models with TUR-DPO.
Based on the paper: "TUR-DPO: Structure- and Uncertainty-Aware Direct Preference Optimization"

Usage:
    python train.py --config configs/default.json
    python train.py --model_name meta-llama/Llama-2-7b-hf --dataset path/to/data.json
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turdpo.trainer import TURDPOTrainer, TURDPOConfig
from turdpo.data import PreferenceDataset, create_dataloader, split_dataset
from turdpo.topology import TopologyExtractor
from turdpo.verifier import NodeVerifier
from turdpo.utils import setup_logging, save_config

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model with TUR-DPO")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--reference_model",
        type=str,
        default=None,
        help="Reference model (defaults to model_name)"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data (JSON or JSONL)"
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help="Path to evaluation data"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    
    # TUR-DPO hyperparameters
    parser.add_argument("--beta", type=float, default=2.0, help="DPO temperature")
    parser.add_argument("--gamma", type=float, default=1.0, help="Reward weight")
    parser.add_argument("--a", type=float, default=0.6, help="Semantic/topology mixing")
    parser.add_argument("--lambda_uncertainty", type=float, default=0.5, help="Uncertainty penalty")
    parser.add_argument("--lambda_epi", type=float, default=0.5, help="Epistemic weight")
    parser.add_argument("--lambda_ale", type=float, default=0.5, help="Aleatoric weight")
    parser.add_argument("--tau_w", type=float, default=1.2, help="Weight mapping temperature")
    parser.add_argument("--w_min", type=float, default=0.05, help="Minimum weight floor")
    parser.add_argument("--k_samples", type=int, default=3, help="Graph re-elicitation samples")
    
    # Reference policy
    parser.add_argument("--use_ema", action="store_true", help="Use EMA reference")
    parser.add_argument("--ema_decay", type=float, default=0.995, help="EMA decay rate")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=500, help="Save frequency")
    parser.add_argument("--eval_steps", type=int, default=100, help="Eval frequency")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 training")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 training")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup
    setup_logging()
    logger.info("Starting TUR-DPO training")
    logger.info(f"Arguments: {args}")
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer and models
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine dtype
    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    
    # Load policy model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if args.device == "cuda" else None
    )
    
    # Load reference model
    ref_model_name = args.reference_model or args.model_name
    reference_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        torch_dtype=dtype,
        device_map="auto" if args.device == "cuda" else None
    )
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    # Load data
    logger.info(f"Loading training data from: {args.train_data}")
    if args.train_data.endswith('.jsonl'):
        train_dataset = PreferenceDataset.from_jsonl(
            args.train_data, tokenizer, max_length=args.max_length
        )
    else:
        train_dataset = PreferenceDataset.from_json(
            args.train_data, tokenizer, max_length=args.max_length
        )
    
    # Split for calibration
    train_dataset, val_dataset, calib_dataset = split_dataset(
        train_dataset, train_ratio=0.9, calibration_ratio=0.02, seed=args.seed
    )
    
    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Val size: {len(val_dataset)}")
    logger.info(f"Calibration size: {len(calib_dataset)}")
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    eval_dataloader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Create config
    config = TURDPOConfig(
        beta=args.beta,
        gamma=args.gamma,
        a=args.a,
        lambda_uncertainty=args.lambda_uncertainty,
        lambda_epi=args.lambda_epi,
        lambda_ale=args.lambda_ale,
        tau_w=args.tau_w,
        w_min=args.w_min,
        k_samples=args.k_samples,
        use_ema_reference=args.use_ema,
        ema_decay=args.ema_decay,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps
    )
    
    # Save config
    save_config(config, output_dir / "config.json")
    
    # Initialize trainer
    trainer = TURDPOTrainer(
        model=model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        config=config,
        device=args.device
    )
    
    # Train
    logger.info("Starting training...")
    results = trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_epochs=args.num_epochs
    )
    
    logger.info(f"Training completed! Results: {results}")
    
    # Save final model
    final_model_path = output_dir / "final_model"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Save checkpoint
    trainer.save_checkpoint(output_dir / "checkpoint.pt")
    
    logger.info(f"Model saved to {final_model_path}")


if __name__ == "__main__":
    main()
