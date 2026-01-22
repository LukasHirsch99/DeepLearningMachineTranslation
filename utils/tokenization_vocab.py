"""
Tokenization and Vocabulary Building Module for Neural Machine Translation
Implements character-level and word-level tokenization with vocabulary management
"""

import json
from collections import Counter
from typing import List, Dict, Optional
import re
from tokenizers import Tokenizer as HFTokenizer

class Vocabulary:
    """
    Vocabulary class that maps tokens to indices and vice versa.
    Handles special tokens and OOV (out-of-vocabulary) words.
    """
    
    def __init__(self, max_vocab_size: Optional[int] = None, min_freq: int = 1):
        """
        Initialize vocabulary.
        
        Args:
            max_vocab_size: Maximum vocabulary size (None for unlimited)
            min_freq: Minimum frequency for a token to be included
        """
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        
        # Token to index mappings
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        
        # Special token indices
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        
        # Initialize with special tokens
        self._add_token(Tokenizer.PAD_TOKEN, self.pad_idx)
        self._add_token(Tokenizer.SOS_TOKEN, self.sos_idx)
        self._add_token(Tokenizer.EOS_TOKEN, self.eos_idx)
        self._add_token(Tokenizer.UNK_TOKEN, self.unk_idx)
        
        self.token_freqs: Counter = Counter()
        
    def _add_token(self, token: str, idx: int):
        """Add a token with a specific index"""
        self.token2idx[token] = idx
        self.idx2token[idx] = token
        
    def build_from_texts(self, texts: List[List[str]]):
        """
        Build vocabulary from a list of tokenized texts.
        
        Args:
            texts: List of tokenized texts (list of token lists)
        """
        # Count token frequencies
        for tokens in texts:
            self.token_freqs.update(tokens)
        
        # Sort by frequency (descending)
        sorted_tokens = sorted(self.token_freqs.items(), 
                              key=lambda x: x[1], 
                              reverse=True)
        
        # Add tokens to vocabulary
        idx = len(self.token2idx)  # Start after special tokens
        for token, freq in sorted_tokens:
            # Skip if already added (special tokens)
            if token in self.token2idx:
                continue
                
            # Check frequency threshold
            if freq < self.min_freq:
                break
                
            # Check vocab size limit
            if self.max_vocab_size and idx >= self.max_vocab_size:
                break
                
            self._add_token(token, idx)
            idx += 1
            
        print(f"Vocabulary built with {len(self.token2idx)} tokens")
        print(f"Total unique tokens in corpus: {len(self.token_freqs)}")
        
    def encode(self, tokens: List[str], add_sos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Convert tokens to indices.
        
        Args:
            tokens: List of tokens
            add_sos: Add start-of-sequence token
            add_eos: Add end-of-sequence token
            
        Returns:
            List of token indices
        """
        indices = []
        
        if add_sos:
            indices.append(self.sos_idx)
            
        for token in tokens:
            indices.append(self.token2idx.get(token, self.unk_idx))
            
        if add_eos:
            indices.append(self.eos_idx)
            
        return indices
    
    def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
        """
        Convert indices back to tokens.
        
        Args:
            indices: List of token indices
            skip_special: Skip special tokens in output
            
        Returns:
            List of tokens
        """
        tokens = []
        special_indices = {self.pad_idx, self.sos_idx, self.eos_idx}
        
        for idx in indices:
            if skip_special and idx in special_indices:
                continue
            tokens.append(self.idx2token.get(idx, Tokenizer.UNK_TOKEN))
            
        return tokens
    
    def __len__(self) -> int:
        """Return vocabulary size"""
        return len(self.token2idx)
    
    def save(self, filepath: str):
        """Save vocabulary to file"""
        vocab_data = {
            'token2idx': self.token2idx,
            'idx2token': self.idx2token,
            'token_freqs': dict(self.token_freqs),
            'max_vocab_size': self.max_vocab_size,
            'min_freq': self.min_freq
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"Vocabulary saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """Load vocabulary from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        vocab = cls(
            max_vocab_size=vocab_data['max_vocab_size'],
            min_freq=vocab_data['min_freq']
        )
        
        # Restore mappings (convert string keys back to int for idx2token)
        vocab.token2idx = vocab_data['token2idx']
        vocab.idx2token = {int(k): v for k, v in vocab_data['idx2token'].items()}
        vocab.token_freqs = Counter(vocab_data['token_freqs'])
        
        print(f"Vocabulary loaded from {filepath} with {len(vocab)} tokens")
        return vocab


class Tokenizer:
    """Base tokenizer class with support for special tokens"""
    
    # Special tokens
    PAD_TOKEN = '[pad]'
    SOS_TOKEN = '[sos]'  # Start of sequence
    EOS_TOKEN = '[eos]'  # End of sequence
    UNK_TOKEN = '[unk]'  # Unknown token
    
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        
    def __len__(self) -> int:
        """Return vocabulary size."""
        raise NotImplementedError
    
    def build_from_texts(self, texts: List[str]):
        """
        Build vocabulary from a list of tokenized texts.
        
        Args:
            texts: List of tokenized texts (list of token lists)
        """
        raise NotImplementedError

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of tokens"""
        raise NotImplementedError
        
    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text"""
        raise NotImplementedError
    
    def encode(self, tokens: List[str], add_sos: bool = False, add_eos: bool = False) -> List[int]:
        """Convert tokens to indices"""
        raise NotImplementedError

    def encode_batch(self, texts: List[str], add_sos: bool = False, add_eos: bool = False) -> List[List[int]]:
        """Encode a batch of texts into lists of indices"""
        raise NotImplementedError

    def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
        """
        Convert indices back to tokens.
        """
        raise NotImplementedError

    def decode_to_text(self, indices: List[int], skip_special: bool = True) -> str:
        """Convert indices back to text string."""
        raise NotImplementedError

class HFTokenizerWrapper(Tokenizer):
    """
    Wrapper to make Hugging Face tokenizer compatible with your existing code.
    """
    
    def __init__(self, hf_tokenizer: HFTokenizer):
        """
        Initialize wrapper.
        
        Args:
            hf_tokenizer: Hugging Face Tokenizer instance
        """
        self.tokenizer = hf_tokenizer
        
        # Get special token IDs
        self.pad_idx = self.tokenizer.token_to_id(self.PAD_TOKEN)
        self.sos_idx = self.tokenizer.token_to_id(self.SOS_TOKEN)
        self.eos_idx = self.tokenizer.token_to_id(self.EOS_TOKEN)
        self.unk_idx = self.tokenizer.token_to_id(self.UNK_TOKEN)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into subword tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of subword tokens
        """
        encoding = self.tokenizer.encode(text)
        return encoding.tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """
        Convert tokens back to text.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Detokenized text string
        """
        return self.tokenizer.decode(self.tokenizer.encode(tokens).ids)

    def encode(self, tokens: List[str], add_sos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Convert tokens to indices.
        
        Args:
            tokens: List of tokens (already tokenized)
            add_sos: Add start-of-sequence token
            add_eos: Add end-of-sequence token
            
        Returns:
            List of token indices
        """
        indices = []
        
        if add_sos:
            indices.append(self.sos_idx)
        
        for token in tokens:
            idx = self.tokenizer.token_to_id(token)
            if idx is None:
                idx = self.unk_idx
            indices.append(idx)
        
        if add_eos:
            indices.append(self.eos_idx)
        
        return indices
    
    def encode_batch(self, texts: List[str], add_sos: bool = False, add_eos: bool = False) -> List[List[int]]:
        """
        Encode a batch of texts into lists of indices.
        
        Args:
            texts: List of input texts
            add_sos: Add start-of-sequence token
            add_eos: Add end-of-sequence token
            
        Returns:
            List of lists of token indices
        """
        # Use fast batch encoding - add_special_tokens already includes BOS/EOS
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=True)
        return [enc.ids for enc in encodings]
    
    def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
        """
        Convert indices back to tokens.
        
        Args:
            indices: List of token indices
            skip_special: Skip special tokens
            
        Returns:
            List of tokens
        """
        # Decode to string first
        text = self.tokenizer.decode(indices, skip_special_tokens=skip_special)
        # Split back into tokens for compatibility
        return text.split()
    
    def decode_to_text(self, indices: List[int], skip_special: bool = True) -> str:
        """
        Convert indices directly to text string.
        
        Args:
            indices: List of token indices
            skip_special: Skip special tokens
            
        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(indices, skip_special_tokens=skip_special)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.tokenizer.get_vocab_size()


class WordTokenizer(Tokenizer):
    """Simple word-level tokenizer"""

    def __init__(
        self,
        lowercase: bool = True,
        vocab: Optional[Vocabulary] = None,
        max_vocab_size: Optional[int] = None,
        min_freq: int = 1,
    ):
        super().__init__(lowercase=lowercase)
        # allow sharing an existing vocab; otherwise create empty one
        self.vocab = vocab if vocab is not None else Vocabulary(
            max_vocab_size=max_vocab_size,
            min_freq=min_freq,
        )

    def __len__(self) -> int:
        return len(self.vocab)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if self.lowercase:
            text = text.lower()
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text"""
        # Simple joining with spaces, could be improved
        result = []
        for i, token in enumerate(tokens):
            if token in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]:
                continue
            if i > 0 and token not in '.,!?;:)\']}"' and tokens[i-1] not in '([{':
                result.append(' ')
            result.append(token)
        return ''.join(result).strip()
    
    def build_from_texts(self, texts: List[str]):
        """Build the vocabulary from texts."""

        tokenized_texts = [self.tokenize(text) for text in texts]
        self.vocab.build_from_texts(tokenized_texts)

    def encode(self, tokens: List[str], add_sos: bool = False, add_eos: bool = False) -> List[int]:
        """Convert tokens to indices using the internal vocabulary."""
        return self.vocab.encode(tokens, add_sos=add_sos, add_eos=add_eos)

    def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
        """Convert indices back to tokens using the internal vocabulary."""
        return self.vocab.decode(indices, skip_special=skip_special)
    
    def decode_to_text(self, indices: List[int], skip_special: bool = True) -> str:
        """Convert indices back to text string."""
        tokens = self.decode(indices, skip_special=skip_special)
        return self.detokenize(tokens)
