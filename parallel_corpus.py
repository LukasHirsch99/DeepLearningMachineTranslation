"""
Dataset Loading and Preprocessing for Neural Machine Translation
Handles WMT datasets, parallel corpus processing, and PyTorch Dataset/DataLoader
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Optional
from tokenization_vocab import Tokenizer


class ParallelCorpus:
    """
    Handles loading and preprocessing of parallel translation corpora.
    Works with any parallel text files (sentence-aligned source and target).
    """
    
    def __init__(self, 
                 source_path: str, 
                 target_path: str,
                 max_length: Optional[int] = None,
                 filter_long: bool = True):
        """
        Initialize parallel corpus loader.
        
        Args:
            source_path: Path to source language file (one sentence per line)
            target_path: Path to target language file (one sentence per line)
            max_length: Maximum sequence length (filters longer sequences)
            filter_long: Whether to filter out sequences longer than max_length
        """
        self.source_path = source_path
        self.target_path = target_path
        self.max_length = max_length
        self.filter_long = filter_long
        
        self.source_sentences: List[str] = []
        self.target_sentences: List[str] = []
        
    def load(self) -> Tuple[List[str], List[str]]:
        """
        Load parallel sentences from files.
        
        Returns:
            Tuple of (source_sentences, target_sentences)
        """
        print("Loading parallel corpus from:")
        print(f"  Source: {self.source_path}")
        print(f"  Target: {self.target_path}")
        
        # Read source sentences
        with open(self.source_path, 'r', encoding='utf-8') as f:
            source_lines = [line.strip() for line in f if line.strip()]
        
        # Read target sentences
        with open(self.target_path, 'r', encoding='utf-8') as f:
            target_lines = [line.strip() for line in f if line.strip()]
        
        # Verify alignment
        if len(source_lines) != len(target_lines):
            raise ValueError(
                f"Misaligned corpus: {len(source_lines)} source sentences "
                f"but {len(target_lines)} target sentences"
            )
        
        print(f"Loaded {len(source_lines)} parallel sentence pairs")
        
        self.source_sentences = source_lines
        self.target_sentences = target_lines
        
        return self.source_sentences, self.target_sentences
    
    def filter_by_length(self, 
                        source_tokenized: List[List[str]], 
                        target_tokenized: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Filter sentence pairs by length.
        
        Args:
            source_tokenized: List of tokenized source sentences
            target_tokenized: List of tokenized target sentences
            
        Returns:
            Filtered (source, target) tokenized sentences
        """
        if not self.filter_long or self.max_length is None:
            return source_tokenized, target_tokenized
        
        filtered_source = []
        filtered_target = []
        
        original_count = len(source_tokenized)
        
        for src, tgt in zip(source_tokenized, target_tokenized):
            # Filter if either sentence is too long
            if len(src) <= self.max_length and len(tgt) <= self.max_length:
                filtered_source.append(src)
                filtered_target.append(tgt)
        
        filtered_count = len(filtered_source)
        removed_count = original_count - filtered_count
        
        print(f"Filtered by length (max={self.max_length}):")
        print(f"  Kept: {filtered_count} pairs")
        print(f"  Removed: {removed_count} pairs ({removed_count/original_count*100:.1f}%)")
        
        return filtered_source, filtered_target


class TranslationDataset(Dataset):
    """
    PyTorch Dataset for neural machine translation.
    Handles tokenization, encoding, and batching.
    """
    
    def __init__(self,
                 source_sentences: List[str],
                 target_sentences: List[str],
                 source_tokenizer: Tokenizer,
                 target_tokenizer: Tokenizer,
                 max_length: Optional[int] = None):
        """
        Initialize translation dataset.
        
        Args:
            source_sentences: List of source language sentences
            target_sentences: List of target language sentences
            source_vocab: Vocabulary for source language
            target_vocab: Vocabulary for target language
            max_length: Maximum sequence length
        """
        self._source_sentences = source_sentences
        self._target_sentences = target_sentences
        self._source_tokenizer = source_tokenizer
        self._target_tokenizer = target_tokenizer
        self.max_length = max_length
        
        # Preprocess all data
        self._preprocess()
        
    def _preprocess(self):
        """Tokenize and encode all sentences."""
        print("Preprocessing dataset...")
        
        self.source_encoded = []
        self.target_encoded = []
        
        skipped = 0
        
        for src_sent, tgt_sent in zip(self._source_sentences, self._target_sentences):

            src_tokens = self._source_tokenizer.tokenize(src_sent)
            tgt_tokens = self._target_tokenizer.tokenize(tgt_sent)
            
            # Skip if too long
            if self.max_length and (len(src_tokens) > self.max_length or len(tgt_tokens) > self.max_length):
                skipped += 1
                continue
            
            # Encode
            # Source: add <eos> only
            src_indices = self._source_tokenizer.encode(src_tokens, add_sos=False, add_eos=True)
            
            # Target: add <sos> and <eos>
            tgt_indices = self._target_tokenizer.encode(tgt_tokens, add_sos=True, add_eos=True)
            
            self.source_encoded.append(src_indices)
            self.target_encoded.append(tgt_indices)
        
        print(f"Preprocessed {len(self.source_encoded)} sentence pairs")
        if skipped > 0:
            print(f"Skipped {skipped} pairs (too long)")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.source_encoded)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx: Index of the example

        Returns:
            Tuple of (source_tensor, target_tensor)
        """
        src = torch.tensor(self.source_encoded[idx], dtype=torch.long)
        tgt = torch.tensor(self.target_encoded[idx], dtype=torch.long)
        
        return src, tgt


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], 
               pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for batching variable-length sequences.
    Pads sequences to the same length within a batch.
    
    Args:
        batch: List of (source, target) tensor pairs
        pad_idx: Padding token index
        
    Returns:
        Dictionary containing:
            - 'src': Padded source sequences [batch_size, src_len]
            - 'tgt': Padded target sequences [batch_size, tgt_len]
            - 'src_lengths': Original source lengths [batch_size]
            - 'tgt_lengths': Original target lengths [batch_size]
    """
    # Separate source and target
    src_batch = [item[0] for item in batch]
    tgt_batch = [item[1] for item in batch]

    
    # Pad sequences to max length in batch
    # pad_sequence pads to max length and stacks into tensor
    # batch_first=True means shape will be [batch_size, seq_len]
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    
    return (src_padded, tgt_padded)

class DataLoaderFactory:
    """Factory for creating train/val/test dataloaders with consistent settings."""
    
    @staticmethod
    def create_dataloader(
        dataset: TranslationDataset,
        batch_size: int,
        pad_idx: int,
        num_workers: int = 0,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create dataloaders for train, validation, and test sets.
        
        Args:
            dataset: Dataset to create dataloader for
            batch_size: Batch size for training
            pad_idx: Padding index for collate function
            num_workers: Number of worker processes for data loading
            shuffle_train: Whether to shuffle training data
            
        Returns:
            Dictionary with 'train', 'val', and optionally 'test' dataloaders
        """
        # Create collate function with padding index
        collate = lambda batch: collate_fn(batch, pad_idx)
        
        # Training dataloader (shuffled)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=True  # Faster GPU transfer
        )

    @staticmethod
    def create_dataloaders(
        train_dataset: TranslationDataset,
        val_dataset: TranslationDataset,
        test_dataset: Optional[TranslationDataset],
        batch_size: int,
        pad_idx: int,
        num_workers: int = 0,
        shuffle_train: bool = True
    ) -> Dict[str, DataLoader]:
        """
        Create dataloaders for train, validation, and test sets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset (optional)
            batch_size: Batch size for training
            pad_idx: Padding index for collate function
            num_workers: Number of worker processes for data loading
            shuffle_train: Whether to shuffle training data
            
        Returns:
            Dictionary with 'train', 'val', and optionally 'test' dataloaders
        """
        # Create collate function with padding index
        collate = lambda batch: collate_fn(batch, pad_idx)
        
        # Training dataloader (shuffled)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=True  # Faster GPU transfer
        )
        
        # Validation dataloader (not shuffled)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=True
        )
        
        loaders = {
            'train': train_loader,
            'val': val_loader
        }
        
        # Test dataloader (optional, not shuffled)
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate,
                num_workers=num_workers,
                pin_memory=True
            )
            loaders['test'] = test_loader
        
        return loaders


def create_sample_data(output_dir: str = "./sample_data"):
    """
    Create sample parallel corpus files for testing.
    Simulates WMT-style parallel text files.
    
    Args:
        output_dir: Directory to save sample files
    """
    from pathlib import Path
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Sample English-French parallel corpus
    en_sentences = [
        "Hello, how are you?",
        "I am learning machine translation.",
        "This is a neural network project.",
        "The weather is nice today.",
        "Machine learning is fascinating.",
        "I enjoy studying artificial intelligence.",
        "Neural networks can translate languages.",
        "This dataset contains parallel sentences.",
        "Translation quality depends on training data.",
        "Transformer models are very powerful.",
    ]
    
    fr_sentences = [
        "Bonjour, comment allez-vous?",
        "J'apprends la traduction automatique.",
        "C'est un projet de réseau neuronal.",
        "Le temps est beau aujourd'hui.",
        "L'apprentissage automatique est fascinant.",
        "J'aime étudier l'intelligence artificielle.",
        "Les réseaux neuronaux peuvent traduire des langues.",
        "Cet ensemble de données contient des phrases parallèles.",
        "La qualité de traduction dépend des données d'entraînement.",
        "Les modèles Transformer sont très puissants.",
    ]
    
    # Write to files
    with open(f"{output_dir}/train.en", 'w', encoding='utf-8') as f:
        f.write('\n'.join(en_sentences))
    
    with open(f"{output_dir}/train.fr", 'w', encoding='utf-8') as f:
        f.write('\n'.join(fr_sentences))
    
    print(f"Sample data created in {output_dir}/")
    print(f"  - train.en: {len(en_sentences)} sentences")
    print(f"  - train.fr: {len(fr_sentences)} sentences")
