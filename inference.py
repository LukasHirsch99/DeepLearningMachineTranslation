from datasets import load_dataset, DatasetDict
import torch
import time
from utils.translation_transformer import TransformerConfig
from torch.utils.data import DataLoader
from utils.parallel_corpus import collate_fn
from tokenizers import Tokenizer as HFTokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing
from utils.tokenization_vocab import HFTokenizerWrapper, Tokenizer
from pathlib import Path
from functools import partial
from utils.translation_transformer import TranslationTransformer
from utils.train import estimate_loss
import torch.nn as nn


vocab_size = 30_000
vocab_path = "./data/bpe_tokenizer.json"
checkpoint_path = "./models/aiayn_base_100k.pt"

batch_size = 64
dataset_max_sample_len = 100

sharedVocab = True

configBig = TransformerConfig(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    max_len=150,
)

label_smoothing = 0.1


def get_tokenizer() -> HFTokenizer:
    bpe_tokenizer = HFTokenizer(BPE(unk_token=Tokenizer.UNK_TOKEN))
    trainer = BpeTrainer(
        special_tokens=[
            Tokenizer.PAD_TOKEN,
            Tokenizer.SOS_TOKEN,
            Tokenizer.EOS_TOKEN,
            Tokenizer.UNK_TOKEN,
        ],
        vocab_size=vocab_size,
        show_progress=True,
    )

    bpe_tokenizer.pre_tokenizer = Metaspace()
    bpe_tokenizer.decoder = decoders.Metaspace()

    pretrained = True  # Set to True if you want to load a previously saved tokenizer

    Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(vocab_path).is_file():
        pretrained = True

    if pretrained:
        bpe_tokenizer = HFTokenizer.from_file(vocab_path)

        bpe_tokenizer.post_processor = TemplateProcessing(
            single=f"{Tokenizer.SOS_TOKEN} $A {Tokenizer.EOS_TOKEN}",
            special_tokens=[
                (Tokenizer.SOS_TOKEN, bpe_tokenizer.token_to_id(Tokenizer.SOS_TOKEN)),
                (Tokenizer.EOS_TOKEN, bpe_tokenizer.token_to_id(Tokenizer.EOS_TOKEN)),
            ],
        )
    else:
        bpe_tokenizer.train(
            [
                "./datasets/wmt14_translate_de-en_test.csv",
                "./datasets/wmt14_translate_de-en_train.csv",
                "./datasets/wmt14_translate_de-en_validation.csv",
            ],
            trainer=trainer,
        )

        bpe_tokenizer.save(vocab_path)

    print(f"Vocab size: {bpe_tokenizer.get_vocab_size():,}")

    return bpe_tokenizer


def tokenizer_decode_batch(tokenizer: HFTokenizerWrapper, datasets: DatasetDict):

    def tokenize_batch(examples):
        inputs = [e["de"] for e in examples["translation"]]
        targets = [e["en"] for e in examples["translation"]]
        input_encodings = tokenizer.tokenizer.encode_batch_fast(inputs)
        target_encodings = tokenizer.tokenizer.encode_batch_fast(targets)
        return {
            "src": [enc.ids[1:] for enc in input_encodings],  # remove sos token
            "tgt": [enc.ids for enc in target_encodings],  # keep sos token
        }

    tokenized_ds = datasets.map(
        tokenize_batch,
        batched=True,
        num_proc=8,
        remove_columns=["translation"],
        load_from_cache_file=True,
        cache_file_names={
            "train": "./data/tokenized_dataset/tokenized_wmt14_deen_train.arrow",
            "test": "./data/tokenized_dataset/tokenized_wmt14_deen_test.arrow",
            "validation": "./data/tokenized_dataset/tokenized_wmt14_deen_validation.arrow",
        },
    )

    collate = partial(collate_fn, pad_idx=tokenizer.pad_idx)
    dl_train = DataLoader(
        tokenized_ds["train"].with_format("torch"),
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate,
        shuffle=True,
    )
    dl_test = DataLoader(
        tokenized_ds["test"].with_format("torch"),
        num_workers=4,
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=True,
    )

    print(
        f"Train samples: {len(tokenized_ds['train']):,}, Test samples: {len(tokenized_ds['test']):,}"
    )
    print(f"Train batches: {len(dl_train):,}, Test batches: {len(dl_test):,}")
    return dl_train, dl_test


def load_model(model: torch.nn.Module, path: str, device: torch.device):
    state_dict = torch.load(path, map_location=device)["model_state_dict"]
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)


def move_to_device(model, device):
    # 1. Enable TF32 for faster matmul on Ampere+ GPUs (A100, RTX 3090, etc.)
    # This provides ~2x speedup for matrix multiplications with minimal accuracy loss
    # torch.set_float32_matmul_precision('high')  # Options: 'highest', 'high', 'medium'
    # torch.backends.fp32_precision = 'tf32'

    # 2. For MPS (Apple Silicon), ensure we're using optimal settings
    if device.type == "mps":
        # MPS backend is already optimized, but we can ensure memory efficiency
        torch.mps.empty_cache()  # Clear any cached memory
    elif device.type == "cuda":
        # Enable TF32 for cuDNN convolutions as well
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("✓ Running on CPU (no GPU optimizations)")

    # Move model to device (GPU if available)
    return model.to(device)


if __name__ == "__main__":
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {DEVICE}")

    ds = load_dataset("wmt/wmt14", "de-en")

    tokenizer = HFTokenizerWrapper(get_tokenizer())
    dl_train, dl_test = tokenizer_decode_batch(tokenizer, ds)

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_idx, label_smoothing=label_smoothing
    )

    # Initialize the model with larger max_len to handle max_length + special tokens
    model = TranslationTransformer(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
        config=configBig,
        padding_idx=tokenizer.pad_idx,
        sharedVocab=sharedVocab,
    )

    print(f"Model initialized!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    load_model(model, checkpoint_path, DEVICE)
    model = move_to_device(model, DEVICE)
    # model = torch.compile(
    #    model, mode="default"
    # )  # Options: 'default', 'reduce-overhead', 'max-autotune'

    start_time = time.time()
    loss = estimate_loss(
        model=model,
        test_loader=dl_test,
        criterion=criterion,
        device=DEVICE,
        eval_iters=len(dl_test),
        print_enabled=True,
    )
    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
    print(f"Estimated loss on test set: {loss:.4f}")

# No compile: Evaluation completed in 20.07 seconds. Estimated loss on test set: 3.3188
# With compile: Evaluation completed in 85.24 seconds. Estimated loss on test set: 3.3171
