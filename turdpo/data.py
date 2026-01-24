"""
Data Module for TUR-DPO

This module implements data loading and preprocessing for TUR-DPO training.
Supports preference pair datasets for DPO-style training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    """A single preference pair for training."""
    prompt: str
    chosen: str
    rejected: str
    prompt_id: Optional[str] = None
    task_score_chosen: Optional[float] = None
    task_score_rejected: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class PreferenceDataset(Dataset):
    """
    Dataset for preference pairs (x, y+, y-).
    
    Supports multiple formats:
    - JSON/JSONL files
    - HuggingFace datasets
    - Direct list of PreferencePair objects
    """
    
    def __init__(
        self,
        data: List[PreferencePair],
        tokenizer,
        max_length: int = 2048,
        max_prompt_length: int = 512,
        truncation_mode: str = "keep_end"
    ):
        """
        Initialize preference dataset.
        
        Args:
            data: List of PreferencePair objects
            tokenizer: Tokenizer for encoding
            max_length: Maximum total sequence length
            max_prompt_length: Maximum prompt length
            truncation_mode: How to truncate ("keep_start" or "keep_end")
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training example."""
        pair = self.data[idx]
        
        # Encode chosen response
        chosen_encoded = self._encode_pair(pair.prompt, pair.chosen)
        
        # Encode rejected response
        rejected_encoded = self._encode_pair(pair.prompt, pair.rejected)
        
        return {
            'prompts': pair.prompt,
            'chosen_responses': pair.chosen,
            'rejected_responses': pair.rejected,
            'chosen_input_ids': chosen_encoded['input_ids'],
            'chosen_attention_mask': chosen_encoded['attention_mask'],
            'chosen_labels': chosen_encoded['labels'],
            'rejected_input_ids': rejected_encoded['input_ids'],
            'rejected_attention_mask': rejected_encoded['attention_mask'],
            'rejected_labels': rejected_encoded['labels'],
            'task_scores_chosen': pair.task_score_chosen,
            'task_scores_rejected': pair.task_score_rejected,
        }
    
    def _encode_pair(
        self,
        prompt: str,
        response: str
    ) -> Dict[str, torch.Tensor]:
        """Encode a prompt-response pair."""
        # Format as conversation
        full_text = f"{prompt}\n\n{response}"
        
        # Tokenize
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (mask prompt tokens)
        prompt_encoded = self.tokenizer(
            prompt + "\n\n",
            max_length=self.max_prompt_length,
            truncation=True,
            return_tensors='pt'
        )
        prompt_length = prompt_encoded['input_ids'].shape[1]
        
        labels = encoded['input_ids'].clone()
        labels[:, :prompt_length] = -100  # Ignore prompt in loss
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }
    
    @classmethod
    def from_json(
        cls,
        path: str,
        tokenizer,
        **kwargs
    ) -> 'PreferenceDataset':
        """Load dataset from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        pairs = []
        for item in raw_data:
            pair = PreferencePair(
                prompt=item.get('prompt', ''),
                chosen=item.get('chosen', item.get('preferred', '')),
                rejected=item.get('rejected', item.get('dispreferred', '')),
                prompt_id=item.get('id'),
                task_score_chosen=item.get('task_score_chosen'),
                task_score_rejected=item.get('task_score_rejected'),
                metadata=item.get('metadata')
            )
            pairs.append(pair)
        
        return cls(pairs, tokenizer, **kwargs)
    
    @classmethod
    def from_jsonl(
        cls,
        path: str,
        tokenizer,
        **kwargs
    ) -> 'PreferenceDataset':
        """Load dataset from JSONL file."""
        pairs = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    pair = PreferencePair(
                        prompt=item.get('prompt', ''),
                        chosen=item.get('chosen', item.get('preferred', '')),
                        rejected=item.get('rejected', item.get('dispreferred', '')),
                        prompt_id=item.get('id'),
                        task_score_chosen=item.get('task_score_chosen'),
                        task_score_rejected=item.get('task_score_rejected'),
                        metadata=item.get('metadata')
                    )
                    pairs.append(pair)
        
        return cls(pairs, tokenizer, **kwargs)


class ListwisePreferenceDataset(Dataset):
    """
    Dataset for listwise preference optimization with multiple candidates.
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        num_candidates: int = 4,
        max_length: int = 2048
    ):
        """
        Initialize listwise dataset.
        
        Args:
            data: List of items with 'prompt' and 'candidates' keys
            tokenizer: Tokenizer for encoding
            num_candidates: Number of candidates per prompt (k)
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.num_candidates = num_candidates
        self.max_length = max_length
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training example with k candidates."""
        item = self.data[idx]
        prompt = item['prompt']
        candidates = item['candidates'][:self.num_candidates]
        preferences = item.get('preferences', [1] + [0] * (len(candidates) - 1))
        
        # Encode all candidates
        encoded_candidates = []
        for candidate in candidates:
            encoded = self._encode_pair(prompt, candidate['response'])
            encoded_candidates.append(encoded)
        
        # Pad to num_candidates if needed
        while len(encoded_candidates) < self.num_candidates:
            encoded_candidates.append(encoded_candidates[-1])
            preferences.append(0)
        
        return {
            'prompt': prompt,
            'candidates': [c['response'] for c in candidates],
            'input_ids': torch.stack([e['input_ids'] for e in encoded_candidates]),
            'attention_mask': torch.stack([e['attention_mask'] for e in encoded_candidates]),
            'labels': torch.stack([e['labels'] for e in encoded_candidates]),
            'preferences': torch.tensor(preferences[:self.num_candidates]),
            'rewards': torch.tensor([c.get('reward', 0.0) for c in candidates[:self.num_candidates]])
        }
    
    def _encode_pair(
        self,
        prompt: str,
        response: str
    ) -> Dict[str, torch.Tensor]:
        """Encode a prompt-response pair."""
        full_text = f"{prompt}\n\n{response}"
        
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        prompt_encoded = self.tokenizer(
            prompt + "\n\n",
            truncation=True,
            return_tensors='pt'
        )
        prompt_length = prompt_encoded['input_ids'].shape[1]
        
        labels = encoded['input_ids'].clone()
        labels[:, :prompt_length] = -100
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }


def collate_preference_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for preference pairs.
    """
    collated = {
        'prompts': [item['prompts'] for item in batch],
        'chosen_responses': [item['chosen_responses'] for item in batch],
        'rejected_responses': [item['rejected_responses'] for item in batch],
        'chosen_input_ids': torch.stack([item['chosen_input_ids'] for item in batch]),
        'chosen_attention_mask': torch.stack([item['chosen_attention_mask'] for item in batch]),
        'chosen_labels': torch.stack([item['chosen_labels'] for item in batch]),
        'rejected_input_ids': torch.stack([item['rejected_input_ids'] for item in batch]),
        'rejected_attention_mask': torch.stack([item['rejected_attention_mask'] for item in batch]),
        'rejected_labels': torch.stack([item['rejected_labels'] for item in batch]),
    }
    
    # Handle optional task scores
    if batch[0].get('task_scores_chosen') is not None:
        collated['task_scores_chosen'] = torch.tensor(
            [item['task_scores_chosen'] or 0.5 for item in batch]
        )
        collated['task_scores_rejected'] = torch.tensor(
            [item['task_scores_rejected'] or 0.5 for item in batch]
        )
    
    return collated


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn=None
) -> DataLoader:
    """
    Create a DataLoader for training/evaluation.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        collate_fn: Custom collate function
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn or collate_preference_batch,
        pin_memory=True
    )


def split_dataset(
    dataset: PreferenceDataset,
    train_ratio: float = 0.9,
    calibration_ratio: float = 0.02,
    seed: int = 42
) -> Tuple[PreferenceDataset, PreferenceDataset, PreferenceDataset]:
    """
    Split dataset into train, validation, and calibration sets.
    
    Args:
        dataset: Full dataset
        train_ratio: Fraction for training
        calibration_ratio: Fraction for calibration (from paper: 2%)
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset, calibration_dataset)
    """
    import random
    random.seed(seed)
    
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)
    
    n_calib = int(n * calibration_ratio)
    n_train = int(n * train_ratio)
    
    calib_indices = indices[:n_calib]
    train_indices = indices[n_calib:n_calib + n_train]
    val_indices = indices[n_calib + n_train:]
    
    train_data = [dataset.data[i] for i in train_indices]
    val_data = [dataset.data[i] for i in val_indices]
    calib_data = [dataset.data[i] for i in calib_indices]
    
    return (
        PreferenceDataset(train_data, dataset.tokenizer, 
                         dataset.max_length, dataset.max_prompt_length),
        PreferenceDataset(val_data, dataset.tokenizer,
                         dataset.max_length, dataset.max_prompt_length),
        PreferenceDataset(calib_data, dataset.tokenizer,
                         dataset.max_length, dataset.max_prompt_length)
    )
