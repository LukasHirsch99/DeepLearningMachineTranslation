import csv
import torch
import sacrebleu
from tqdm import tqdm
from pathlib import Path

from translation_lstm import TranslationLSTM
from tokenizers import Tokenizer as HFTokenizer
from tokenization_vocab import HFTokenizerWrapper
from parallel_corpus import TranslationDataset, DataLoaderFactory
from translate import translate_sample

# ----------------- Paths & Hyperparameters -----------------
VOCAB_PATH = "./data/bpe_tokenizer.json"
TEST_CSV = "./datasets/wmt14_translate_de-en_test.csv"
MODEL_PATH = "./models/best_model.pt"
MAX_TEST_SAMPLES = 20
MAX_LEN = 100
BATCH_SIZE = 64

# ----------------- Load CSV -----------------
def load_csv(csv_path):
    src, tgt = [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            s, t = row.get('de'), row.get('en')
            if not s or not t:
                continue
            src.append(str(s))
            tgt.append(str(t))
    return src, tgt

# ----------------- BLEU evaluation -----------------
def evaluate_bleu(model, tokenizer: HFTokenizerWrapper, test_loader, max_len=100, device='cpu', print_enabled=False):
    model.eval()
    total_bleu = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            src_batch, tgt_batch, src_mask, tgt_mask, src_lengths, tgt_lengths = batch
            batch_size = src_batch.size(0)

            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            # Generate translations batch-wise
            for i in range(batch_size):
                src_tokens = src_batch[i]
                tgt_tokens = tgt_batch[i]

                src_sentence = tokenizer.decode_to_text(src_tokens.tolist())
                ref_sentence = tokenizer.decode_to_text(tgt_tokens.tolist())

                translation, _ = translate_sample(
                    src_sentence,
                    model,
                    src_tokenizer=tokenizer,
                    tgt_tokenizer=tokenizer,
                    max_len=max_len,
                    device=device
                )

                bleu_score = sacrebleu.corpus_bleu([translation], [[ref_sentence]]).score
                total_bleu += bleu_score
                total_samples += 1

                if print_enabled:
                    print(f"DE: {src_sentence}")
                    print(f"EN (model): {translation}")
                    print(f"EN (ref): {ref_sentence}")
                    print(f"BLEU: {bleu_score:.2f}\n")

    avg_bleu = total_bleu / total_samples
    print(f"\nAverage BLEU over {total_samples} samples: {avg_bleu:.2f}")
    return avg_bleu

# ----------------- Main -----------------
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        
    print(f"Using device: {DEVICE}")

    # Load tokenizer
    tokenizer = HFTokenizerWrapper(HFTokenizer.from_file(VOCAB_PATH))
    print(f"Vocab size: {len(tokenizer)} | PAD idx: {tokenizer.pad_idx}")

    # Load test CSV
    test_src, test_tgt = load_csv(TEST_CSV)
    if MAX_TEST_SAMPLES:
        test_src = test_src[:MAX_TEST_SAMPLES]
        test_tgt = test_tgt[:MAX_TEST_SAMPLES]
    print(f"Loaded {len(test_src)} test samples")

    # Build dataset and dataloader
    test_ds = TranslationDataset(
        source_sentences=test_src,
        target_sentences=test_tgt,
        source_tokenizer=tokenizer,
        target_tokenizer=tokenizer,
        max_length=MAX_LEN,
        lazy=True
    )

    test_loader = DataLoaderFactory.create_dataloader(
        dataset=test_ds,
        batch_size=BATCH_SIZE,
        pad_idx=tokenizer.pad_idx,
        num_workers=0,
        shuffle=False,
        persistent_workers=True,
        prefetch_factor=4
    )

    # Load model
    model = TranslationLSTM(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        padding_idx=tokenizer.pad_idx
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    print("✅ Model loaded. Starting BLEU evaluation...")

    # Evaluate
    evaluate_bleu(model, tokenizer, test_loader, max_len=MAX_LEN, device=DEVICE, print_enabled=True)


if __name__ == "__main__":
    main()


