"""
Dataset Loading and Preprocessing for Neural Machine Translation
Handles WMT datasets, parallel corpus processing, and PyTorch Dataset/DataLoader
"""

import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Optional, Union, Any
from utils.tokenization_vocab import Tokenizer


class LazyTranslationPairs:
    """
    Wrapper for Hugging Face dataset to provide lazy access to translation pairs.
    Avoids materializing the entire dataset into memory.
    """

    def __init__(
        self, hf_dataset, src_lang: str = "de", tgt_lang: str = "en", mode: str = "both"
    ):
        """
        Args:
            hf_dataset: Hugging Face dataset (e.g., ds['train'])
            src_lang: Source language key
            tgt_lang: Target language key
            mode: 'src' for source only, 'tgt' for target only, 'both' for tuple
        """
        self.hf_dataset = hf_dataset
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.mode = mode

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        """Returns source, target, or (source, target) tuple based on mode."""
        translation = self.hf_dataset[idx]["translation"]
        if self.mode == "src":
            return translation[self.src_lang]
        elif self.mode == "tgt":
            return translation[self.tgt_lang]
        else:  # 'both'
            return (translation[self.src_lang], translation[self.tgt_lang])


class TranslationDataset(Dataset):
    """
    PyTorch Dataset for neural machine translation with lazy loading.
    Processes data on-the-fly instead of materializing in memory.
    """

    def __init__(
        self,
        source_sentences: Union[List[str], Any],
        target_sentences: Union[List[str], Any],
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        max_length: Optional[int] = None,
        lazy: bool = True,
    ):
        """
        Initialize translation dataset.

        Args:
            source_sentences: List or iterable of source language sentences
            target_sentences: List or iterable of target language sentences
            source_tokenizer: Tokenizer for source language
            target_tokenizer: Tokenizer for target language
            max_length: Maximum sequence length (filters out longer sequences)
            lazy: If True, process on-the-fly. If False, preprocess all data (legacy mode)
        """
        self._source_sentences = source_sentences
        self._target_sentences = target_sentences
        self._source_tokenizer = source_tokenizer
        self._target_tokenizer = target_tokenizer
        self.max_length = max_length
        self.lazy = lazy

        if not lazy:
            # Legacy mode: preprocess all data
            self._preprocess()
        else:
            # Lazy mode: just store length
            self._length = len(source_sentences)
            print(f"Initialized lazy dataset with {self._length} sentence pairs")

    def _preprocess(self):
        """Tokenize and encode all sentences (legacy mode)."""
        print("Preprocessing dataset (materialized mode)...")

        self.source_encoded = []
        self.target_encoded = []

        skipped = 0

        for src_sent, tgt_sent in zip(self._source_sentences, self._target_sentences):

            src_indices = self._source_tokenizer.encode(src_sent, add_sos=False, add_eos=True)
            tgt_indices = self._target_tokenizer.encode(tgt_sent, add_sos=True, add_eos=True)

            # Skip if too long
            if self.max_length and (
                len(src_indices) > self.max_length or len(tgt_indices) > self.max_length
            ):
                skipped += 1
                continue

            self.source_encoded.append(src_indices)
            self.target_encoded.append(tgt_indices)

        print(f"Preprocessed {len(self.source_encoded)} sentence pairs")
        if skipped > 0:
            print(f"Skipped {skipped} pairs (too long)")

    def __len__(self) -> int:
        """Return dataset size."""
        if self.lazy:
            return self._length
        else:
            return len(self.source_encoded)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.

        Args:
            idx: Index of the example

        Returns:
            Tuple of (source_tensor, target_tensor)
        """
        if self.lazy:
            # Process on-the-fly
            src_sent = self._source_sentences[idx]
            tgt_sent = self._target_sentences[idx]

            # Tokenize
            src_indices = self._source_tokenizer.encode(src_sent, add_sos=False, add_eos=True)
            tgt_indices = self._target_tokenizer.encode(tgt_sent, add_sos=True, add_eos=True)

            # Apply max_length filter (truncate instead of skip for lazy mode)
            if self.max_length:
                src_indices = src_indices[: self.max_length]
                tgt_indices = tgt_indices[: self.max_length]

            src = torch.tensor(src_indices, dtype=torch.long)
            tgt = torch.tensor(tgt_indices, dtype=torch.long)
        else:
            # Use preprocessed data
            src = torch.tensor(self.source_encoded[idx], dtype=torch.long)
            tgt = torch.tensor(self.target_encoded[idx], dtype=torch.long)

        return {"src": src, "tgt": tgt}


def collate_fn(
    batch: List[Dict[str, torch.Tensor]], pad_idx: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for batching variable-length sequences.
    Pads sequences to the same length within a batch.

    Args:
        batch: List of (source, target) tensor pairs
        pad_idx: Padding token index

    Returns:
        Tuple containing:
            - src_padded: Padded source sequences [batch_size, src_len]
            - tgt_padded: Padded target sequences [batch_size, tgt_len]
            - src_key_padding_mask: Source padding mask [batch_size, src_len] (True = padding)
            - tgt_key_padding_mask: Target padding mask [batch_size, tgt_len] (True = padding)
    """
    # Separate source and target
    src_batch = [item["src"] for item in batch]
    tgt_batch = [item["tgt"] for item in batch]

    # Pad sequences to max length in batch
    # pad_sequence pads to max length and stacks into tensor
    # batch_first=True means shape will be [batch_size, seq_len]
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)

    # Create padding masks (True where padding, False where real tokens)
    # PyTorch transformer expects True for positions to IGNORE
    src_key_padding_mask = src_padded == pad_idx  # [batch_size, src_len]
    tgt_key_padding_mask = tgt_padded == pad_idx  # [batch_size, tgt_len]

    return (src_padded, tgt_padded, src_key_padding_mask, tgt_key_padding_mask)


class DataLoaderFactory:
    """Factory for creating train/val/test dataloaders with consistent settings."""

    @staticmethod
    def create_dataloader(
        dataset: TranslationDataset,
        batch_size: int,
        pad_idx: int,
        num_workers: int = 0,
        shuffle: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
    ) -> DataLoader:
        """
        Create dataloaders for train, validation, and test sets.

        Args:
            dataset: Dataset to create dataloader for
            batch_size: Batch size for training
            pad_idx: Padding index for collate function
            num_workers: Number of worker processes for data loading
            shuffle: Whether to shuffle data
            persistent_workers: Keep workers alive between epochs (faster, uses more memory)
            prefetch_factor: Number of batches to prefetch per worker

        Returns:
            Dictionary with 'train', 'val', and optionally 'test' dataloaders
        """
        # Use partial so collate is picklable for multiprocessing workers
        collate = partial(collate_fn, pad_idx=pad_idx)

        # Build dataloader kwargs
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "collate_fn": collate,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),  # Only for CUDA
        }

        # Add persistent_workers and prefetch_factor only if num_workers > 0
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            loader_kwargs["prefetch_factor"] = prefetch_factor

        return DataLoader(**loader_kwargs)

    @staticmethod
    def create_dataloaders(
        train_dataset: TranslationDataset,
        val_dataset: TranslationDataset,
        test_dataset: Optional[TranslationDataset],
        batch_size: int,
        pad_idx: int,
        num_workers: int = 0,
        shuffle_train: bool = True,
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
        # Use partial so collate is picklable for multiprocessing workers
        collate = partial(collate_fn, pad_idx=pad_idx)

        # Training dataloader (shuffled)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=True,  # Faster GPU transfer
        )

        # Validation dataloader (not shuffled)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=True,
        )

        loaders = {"train": train_loader, "val": val_loader}

        # Test dataloader (optional, not shuffled)
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate,
                num_workers=num_workers,
                pin_memory=True,
            )
            loaders["test"] = test_loader

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
    with open(f"{output_dir}/train.en", "w", encoding="utf-8") as f:
        f.write("\n".join(en_sentences))

    with open(f"{output_dir}/train.fr", "w", encoding="utf-8") as f:
        f.write("\n".join(fr_sentences))

    print(f"Sample data created in {output_dir}/")
    print(f"  - train.en: {len(en_sentences)} sentences")
    print(f"  - train.fr: {len(fr_sentences)} sentences")
