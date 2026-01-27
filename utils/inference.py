from utils.tokenization_vocab import HFTokenizerWrapper
from torch import nn
from torch.nn import functional as F
import torch


@torch.no_grad()
def evalute_model(
    model: nn.Module,
    input_sequence: torch.Tensor,
    tgt_sequence: torch.Tensor,
    device: torch.device,
):
    model.eval()
    src_key_padding_mask = torch.zeros(
        input_sequence.size(0), input_sequence.size(1), dtype=torch.bool, device=device
    )
    tgt_key_padding_mask = torch.zeros(
        tgt_sequence.size(0), tgt_sequence.size(1), dtype=torch.bool, device=device
    )

    return model(
        input_sequence, tgt_sequence, src_key_padding_mask, tgt_key_padding_mask
    )


@torch.no_grad()
def greedy_translate(
    model: nn.Module,
    input_sequence: torch.Tensor,
    src_tokenizer: HFTokenizerWrapper,
    tgt_tokenizer: HFTokenizerWrapper,
    max_len=100,
    device=torch.device("cpu"),
) -> torch.Tensor:
    """
    Translate a single sentence using the model with autoregressive generation.

    Args:
        input_sequence: Input sequence tensor to translate
        model: nn.Module model for translation
        src_tokenizer: Tokenizer for the source language
        tgt_tokenizer: Tokenizer for the target language
        max_len: Maximum sequence length to generate
        device: Device to run on

    Returns:
        Translated sentence and token indices
    """
    model.eval()

    input_sequence = input_sequence.to(device) # [1, src_len]
    tgt_sequence = [tgt_tokenizer.sos_idx]  # Start with SOS token
    tgt_sequence = torch.tensor(
        [tgt_sequence], dtype=torch.long, device=device
    )  # [1, 1]

    # Autoregressive generation loop
    for _ in range(max_len):
        output = evalute_model(
            model,
            input_sequence=input_sequence,
            tgt_sequence=tgt_sequence,
            device=device,
        )  # [1, seq_len, vocab_size]

        # Get prediction for the last token
        next_token_logits = output[0, -1, :]  # [vocab_size]
        next_token = torch.argmax(next_token_logits).item()

        # Append predicted token
        tgt_sequence = torch.cat(
            [tgt_sequence, torch.tensor([[next_token]], device=device)], dim=1
        )

        # Stop if we predict EOS token
        if next_token == tgt_tokenizer.eos_idx:
            break

    return tgt_sequence.tolist()[0]


@torch.no_grad()
def beam_search(
    model: nn.Module,
    input_sequence: torch.Tensor,
    sos: int,
    eos: int,
    beam_width: int = 5,
    max_length: int = 100,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.0,
    device: torch.device = torch.device("cpu"),
):
    """
    Beam search decoding - simpler is better for translation.

    Args:
        model: The trained model used for inference.
        input_sequence: The input sequence [1, src_len].
        sos: Start of sequence token ID.
        eos: End of sequence token ID.
        beam_width: Number of beams to keep.
        max_length: Maximum generation length.
        length_penalty: Length normalization (1.0 = no penalty, <1.0 favors shorter).
        repetition_penalty: Penalty for repeated tokens (1.0 = no penalty).
        device: Device to run on.

    Returns:
        Best decoded sequence (list of token IDs).
    """
    model.eval()

    beam = [([sos], 0.0)]  # (sequence, cumulative_log_prob)
    finished_sequences = []

    for _ in range(max_length):
        if not beam:
            break

        # Batch forward pass for all active beams
        tgt_sequence = torch.tensor([seq for seq, _ in beam], device=device)
        output = evalute_model(
            model,
            input_sequence=input_sequence.repeat(len(beam), 1).to(device),
            tgt_sequence=tgt_sequence,
            device=device,
        )

        next_token_logits = output[:, -1, :]  # [beam_width, vocab_size]
        log_probs = F.log_softmax(next_token_logits, dim=-1)

        # Apply repetition penalty if needed
        if repetition_penalty != 1.0:
            for i, (seq, _) in enumerate(beam):
                for token_id in set(seq[1:]):  # Skip SOS
                    log_probs[i, token_id] /= repetition_penalty

        candidates = []
        # Consider top-k tokens per beam
        top_k = min(beam_width, log_probs.size(-1))
        top_log_probs, top_indices = log_probs.topk(top_k, dim=-1)

        for (seq, score), beam_log_probs, beam_indices in zip(
            beam, top_log_probs, top_indices
        ):
            for token_id, prob in zip(beam_indices, beam_log_probs):
                new_score = score + prob.item()
                new_seq = seq + [token_id.item()]

                if token_id.item() == eos:
                    finished_sequences.append((new_seq, new_score))
                else:
                    candidates.append((new_seq, new_score))

        # Keep top beam_width candidates by RAW score (no premature normalization)
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_width]
        else:
            beam = []

        # Stop if we have enough finished sequences and they're better than active beams
        if len(finished_sequences) >= beam_width and not beam:
            break

    # Add any remaining active beams as finished (append EOS)
    for seq, score in beam:
        finished_sequences.append((seq + [eos], score))

    if finished_sequences:
        # NOW apply length normalization only for final selection
        def get_normalized_score(seq, score):
            # Length = tokens generated (exclude SOS, include EOS)
            length = len(seq) - 1
            return score / (length**length_penalty) if length > 0 else score

        best_seq, _ = max(
            finished_sequences, key=lambda x: get_normalized_score(x[0], x[1])
        )
        return best_seq
    else:
        return [sos, eos]
