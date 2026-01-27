"""
Positional Encoding for Transformer Models

Implements sinusoidal positional encodings to inject sequence order information
into the Transformer architecture, which otherwise has no notion of token position.
"""

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10,000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10,000^(2i/d_model))

    where:
        pos = position in sequence (0, 1, 2, ...)
        i = dimension index (0, 1, 2, ..., d_model/2)
        d_model = embedding dimension
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Dimension of model embeddings (must be even)
            max_len: Maximum sequence length to precompute encodings for
            dropout: Dropout probability applied after adding positional encoding
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Create a matrix to hold positional encodings
        # Shape: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # Create position indices: [0, 1, 2, ..., max_len-1]
        # Shape: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Create the division term for the sinusoidal functions
        # This creates: [1, 10000^(2/d_model), 10000^(4/d_model), ...]
        # Shape: [d_model/2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but part of module state)
        # This means it will be saved with the model but not updated during training
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input embeddings of shape [batch_size, seq_len, d_model]

        Returns:
            Embeddings with positional encoding added, same shape as input
        """
        # x shape: [batch_size, seq_len, d_model]
        # self.pe shape: [1, max_len, d_model]

        seq_len = x.size(1)

        # Extend positional encodings if sequence length exceeds precomputed max_len
        if seq_len > self.pe.size(1):
            self._extend_pe(seq_len, device=x.device)

        # Add positional encoding to embeddings
        # Broadcasting handles the batch dimension
        # We only use the first seq_len positions
        x = x + self.pe[:, :seq_len, :].to(x.device)

        # Apply dropout
        return self.dropout(x)


    def _extend_pe(self, new_max_len: int, device: torch.device) -> None:
        """Recompute positional encodings to cover longer sequences."""
        pe = torch.zeros(new_max_len, self.d_model, device=device)
        position = torch.arange(0, new_max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float, device=device)
            * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe


class LearnedPositionalEncoding(nn.Module):
    """
    Alternative: Learned positional embeddings instead of fixed sinusoidal.
    The model learns the best positional representations during training.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize learned positional encoding.

        Args:
            d_model: Dimension of model embeddings
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(LearnedPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Create learnable positional embeddings
        # These are parameters that will be updated during training
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input embeddings.

        Args:
            x: Input embeddings of shape [batch_size, seq_len, d_model]

        Returns:
            Embeddings with positional encoding added
        """
        batch_size, seq_len, d_model = x.size()

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )

        # Get positional embeddings
        pos_embeddings = self.pe(positions)

        # Add to input embeddings
        x = x + pos_embeddings

        return self.dropout(x)


def visualize_positional_encoding(d_model: int = 512, max_len: int = 100):
    """
    Visualize the positional encoding patterns.
    Helps understand how different dimensions encode position information.

    Args:
        d_model: Model dimension
        max_len: Sequence length to visualize
    """
    # Create positional encoding
    pe_module = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)

    # Get the positional encodings (remove batch dimension)
    pe = pe_module.pe.squeeze(0).numpy()  # Shape: [max_len, d_model]

    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Heatmap of all positional encodings
    im1 = axes[0].imshow(pe.T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Dimension")
    axes[0].set_title("Positional Encoding Heatmap (All Dimensions)")
    plt.colorbar(im1, ax=axes[0])

    # Plot 2: First 8 dimensions over positions
    axes[1].plot(pe[:, :8])
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Encoding Value")
    axes[1].set_title("First 8 Dimensions of Positional Encoding")
    axes[1].legend([f"Dim {i}" for i in range(8)], loc="right")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Show how wavelength increases with dimension
    sample_positions = [0, 10, 20, 30, 40]
    for pos in sample_positions:
        axes[2].plot(pe[pos, :50], label=f"Position {pos}")
    axes[2].set_xlabel("Dimension")
    axes[2].set_ylabel("Encoding Value")
    axes[2].set_title("Positional Encoding at Different Positions (First 50 Dims)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("positional_encoding_visualization.png", dpi=150)
    print("Visualization saved as 'positional_encoding_visualization.png'")
    plt.close()


def compare_encoding_methods(d_model: int = 128, seq_len: int = 50):
    """
    Compare sinusoidal vs learned positional encodings.
    Shows how they differ in their patterns.
    """
    # Create both types
    sinusoidal = PositionalEncoding(d_model=d_model, max_len=seq_len, dropout=0.0)
    learned = LearnedPositionalEncoding(d_model=d_model, max_len=seq_len, dropout=0.0)

    # Create dummy input
    dummy_input = torch.zeros(1, seq_len, d_model)

    # Get encodings
    sin_pe = sinusoidal.pe.squeeze(0)[:seq_len, :].detach().numpy()

    # For learned, we need to do a forward pass
    learned_output = learned(dummy_input)
    learned_pe = learned_output.squeeze(0).detach().numpy()

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sinusoidal
    im1 = axes[0].imshow(sin_pe.T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Dimension")
    axes[0].set_title("Sinusoidal Positional Encoding")
    plt.colorbar(im1, ax=axes[0])

    # Learned (random initialization)
    im2 = axes[1].imshow(learned_pe.T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Dimension")
    axes[1].set_title("Learned Positional Encoding (Random Init)")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig("encoding_comparison.png", dpi=150)
    print("Comparison saved as 'encoding_comparison.png'")
    plt.close()


def demonstrate_wavelength_pattern():
    """
    Demonstrates how the wavelength of sinusoidal functions increases
    with dimension index, allowing the model to attend to different scales.
    """
    d_model = 512
    max_len = 100

    # Get positional encoding
    pe_module = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
    pe = pe_module.pe.squeeze(0).numpy()

    # Select specific dimensions to show wavelength progression
    dimensions = [0, 10, 50, 100, 200, 400]

    fig, axes = plt.subplots(len(dimensions), 1, figsize=(12, 10))

    for idx, dim in enumerate(dimensions):
        axes[idx].plot(pe[:, dim])
        axes[idx].set_ylabel(f"Dim {dim}")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0, max_len)

        # Calculate and display wavelength
        # Wavelength = 2π / frequency
        frequency = 1.0 / (10000.0 ** (dim / d_model))
        wavelength = 2 * math.pi / frequency
        axes[idx].set_title(f"Wavelength ≈ {wavelength:.1f} positions", fontsize=10)

    axes[-1].set_xlabel("Position")
    fig.suptitle("Wavelength Increases with Dimension Index", fontsize=14)

    plt.tight_layout()
    plt.savefig("wavelength_pattern.png", dpi=150)
    print("Wavelength pattern saved as 'wavelength_pattern.png'")
    plt.close()


# Example usage and testing
if __name__ == "__main__":
    print("=== Positional Encoding Demo ===\n")

    # Configuration
    d_model = 512  # Standard Transformer dimension
    max_len = 100
    batch_size = 2
    seq_len = 20

    # Create positional encoding module
    pe_module = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.1)

    print("Positional Encoding Configuration:")
    print(f"  Model dimension (d_model): {d_model}")
    print(f"  Maximum sequence length: {max_len}")
    print(f"  Precomputed PE shape: {pe_module.pe.shape}")

    # Create sample input embeddings (typically from embedding layer)
    # Shape: [batch_size, seq_len, d_model]
    sample_embeddings = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput embeddings shape: {sample_embeddings.shape}")

    # Apply positional encoding
    output = pe_module(sample_embeddings)
    print(f"Output shape (after PE): {output.shape}")

    # Show that positional encoding is deterministic (same for all batches)
    print("\nPositional encoding for position 0 (first few dims):")
    print(pe_module.pe[0, 0, :10])

    print("\nPositional encoding for position 10 (first few dims):")
    print(pe_module.pe[0, 10, :10])

    # Demonstrate properties
    print("\n=== Key Properties ===")
    print(
        f"1. Same positional encoding for all batches: {torch.allclose(output[0], output[1])}... No! Because embeddings differ"
    )
    print("   But PE component is identical across batches")

    # Show the actual PE values (not embeddings)
    pe_values_pos0 = pe_module.pe[0, 0, :]
    pe_values_pos1 = pe_module.pe[0, 1, :]

    print("\n2. Different positions have different encodings:")
    print(
        f"   Position 0 != Position 1: {not torch.allclose(pe_values_pos0, pe_values_pos1)}"
    )

    print("\n3. All values are in range [-1, 1]:")
    print(f"   Min value: {pe_module.pe.min().item():.4f}")
    print(f"   Max value: {pe_module.pe.max().item():.4f}")

    # Test with different sequence lengths
    print("\n=== Testing Variable Sequence Lengths ===")
    for test_len in [5, 10, 20, 50]:
        test_input = torch.randn(1, test_len, d_model)
        test_output = pe_module(test_input)
        print(f"Input length {test_len:2d} -> Output shape: {test_output.shape}")

    # Visualizations
    print("\n=== Creating Visualizations ===")
    visualize_positional_encoding(d_model=512, max_len=100)
    compare_encoding_methods(d_model=128, seq_len=50)
    demonstrate_wavelength_pattern()

    print("\n=== Comparing Sinusoidal vs Learned ===")
    learned_pe = LearnedPositionalEncoding(
        d_model=d_model, max_len=max_len, dropout=0.1
    )
    learned_output = learned_pe(sample_embeddings)
    print(f"Learned PE output shape: {learned_output.shape}")
    print("Learned PE are parameters that will be optimized during training")
    print(
        f"Number of learnable parameters: {sum(p.numel() for p in learned_pe.parameters())}"
    )

    print("\n✓ Positional Encoding demo complete!")
