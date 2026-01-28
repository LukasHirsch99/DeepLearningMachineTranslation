from datasets import load_dataset, DatasetDict
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from utils.translation_transformer import TransformerConfig, TranslationTransformer
from utils.parallel_corpus import collate_fn, TranslationDataset, LazyTranslationPairs
from utils.tokenization_vocab import HFTokenizerWrapper, Tokenizer, BPETokenizer
from utils.train import train
from tokenizers import Tokenizer as HFTokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing
from pathlib import Path
from functools import partial


vocab_size = 40_000
vocab_path = "./data/bpe_tokenizer_40k.json"
checkpoint_path = None
# checkpoint_path = "./models/aiayn_base_100k.pt"

batch_size = 128
dataset_max_sample_len = 50  # Max tokens per sample (including special tokens)

compile_model = False  # Set to False to disable compilation

sharedVocab = True

transformerCfg = TransformerConfig(
    d_model=256,
    nhead=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=1024,
    dropout=0.1,
    max_len=dataset_max_sample_len + 2,  # +2 for special tokens
)

# training
num_steps = 100_000
warmup_steps = 4_000
eval_iters = 10

label_smoothing = 0.1

# optimizer
betas = (0.9, 0.98)
epsilon = 1e-9


def get_tokenizer() -> Tokenizer:
    bpe_tokenizer = HFTokenizer(BPE(unk_token=Tokenizer.UNK_TOKEN))

    bpe_tokenizer.pre_tokenizer = Metaspace()
    bpe_tokenizer.decoder = decoders.Metaspace()
    bpe_tokenizer.post_processor = TemplateProcessing(
        single=f"{Tokenizer.SOS_TOKEN} $A {Tokenizer.EOS_TOKEN}",
        special_tokens=[
            (Tokenizer.SOS_TOKEN, bpe_tokenizer.token_to_id(Tokenizer.SOS_TOKEN)),
            (Tokenizer.EOS_TOKEN, bpe_tokenizer.token_to_id(Tokenizer.EOS_TOKEN)),
        ],
    )

    pretrained = False  # Set to True if you want to load a previously saved tokenizer

    Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(vocab_path).is_file():
        pretrained = True

    if pretrained:
        bpe_tokenizer = HFTokenizer.from_file(vocab_path)

    else:
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

    return HFTokenizerWrapper(bpe_tokenizer)


def tokenizer_decode_batch(tokenizer: Tokenizer, datasets: DatasetDict):

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

    collate = partial(collate_fn, pad_idx=tokenizer.PAD_IDX)
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
    return dl_train, dl_test, len(tokenized_ds["train"]), len(tokenized_ds["test"])


def build_lazy_dataloaders(
    tokenizer: Tokenizer,
    datasets: DatasetDict,
    max_length: int,
    batch_size: int,
    num_workers: int = 0,
):
    """
    Build DataLoaders with on-the-fly tokenization to avoid materializing
    the entire dataset in memory.

    This uses `TranslationDataset` with `LazyTranslationPairs` so samples are
    read and tokenized per-batch from disk.
    """

    train_src = LazyTranslationPairs(
        datasets["train"], src_lang="de", tgt_lang="en", mode="src"
    )
    train_tgt = LazyTranslationPairs(
        datasets["train"], src_lang="de", tgt_lang="en", mode="tgt"
    )
    test_src = LazyTranslationPairs(
        datasets["test"], src_lang="de", tgt_lang="en", mode="src"
    )
    test_tgt = LazyTranslationPairs(
        datasets["test"], src_lang="de", tgt_lang="en", mode="tgt"
    )

    ds_train = TranslationDataset(
        source_sentences=train_src,
        target_sentences=train_tgt,
        source_tokenizer=tokenizer,
        target_tokenizer=tokenizer,
        max_length=max_length,
        lazy=True,
    )
    ds_test = TranslationDataset(
        source_sentences=test_src,
        target_sentences=test_tgt,
        source_tokenizer=tokenizer,
        target_tokenizer=tokenizer,
        max_length=max_length,
        lazy=True,
    )

    collate = partial(collate_fn, pad_idx=tokenizer.PAD_IDX)
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    return dl_train, dl_test, len(ds_train), len(ds_test)


def load_model(model: torch.nn.Module, path: str, device: torch.device):
    state_dict = torch.load(path, map_location=device)["model_state_dict"]
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)


def move_to_device(model: nn.Module, device) -> nn.Module:
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

    # tokenizer = get_tokenizer()
    tokenizer = BPETokenizer(vocab_path)
    
    # Prefer lazy, on-the-fly tokenization to reduce memory footprint
    dl_train, dl_test, train_size, test_size = build_lazy_dataloaders(
        tokenizer,
        ds,
        max_length=dataset_max_sample_len,
        batch_size=batch_size,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.PAD_IDX, label_smoothing=label_smoothing
    )

    # Initialize the model with larger max_len to handle max_length + special tokens
    model = TranslationTransformer(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
        config=transformerCfg,
        padding_idx=tokenizer.PAD_IDX,
        sharedVocab=sharedVocab,
    )

    print(f"Model initialized!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # load_model(model, checkpoint_path, DEVICE)
    # Move to device; disabling compile can save GPU memory
    model = move_to_device(model, DEVICE)

    # ========== TORCH.COMPILE INTEGRATION ==========
    # Compile the model for faster training (PyTorch 2.0+)

    if compile_model and torch.__version__ >= "2.0.0":
        if DEVICE.type == "cuda":
            print("Compiling model with torch.compile...")
            try:
                model = torch.compile(
                    model,
                    mode="default",  # "default", "reduce-overhead", "max-autotune"
                    fullgraph=False,
                )
                print("✓ Model compiled successfully!")
            except Exception as e:
                print(f"⚠ Compilation failed: {e}")
                print("Continuing without compilation...")
        elif DEVICE.type == "mps":
            print("⚠ torch.compile not fully supported on MPS yet")
        else:
            print("⚠ torch.compile only works on CUDA")
    else:
        if not compile_model:
            print("torch.compile disabled by user")
        else:
            print(f"torch.compile requires PyTorch 2.0+ (current: {torch.__version__})")
    # ===============================================

    def lr_lambda(step, warmup_steps=4000):
        step = max(step, 1)
        return transformerCfg.d_model ** (-0.5) * min(
            step**-0.5, step * warmup_steps**-1.5
        )

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.PAD_IDX, label_smoothing=label_smoothing
    )
    optimizer = optim.Adam(model.parameters(), lr=1, betas=betas, eps=epsilon)

    scheduler = LambdaLR(optimizer, lambda step: lr_lambda(step, warmup_steps))

    # Training
    train_losses, best_loss = train(
        model=model,
        config=transformerCfg,
        train_loader=dl_train,
        test_loader=dl_test,
        dataset_size=train_size,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        sos_idx=tokenizer.SOS_IDX,
        pad_idx=tokenizer.PAD_IDX,
        teacher_forcing_start=1.0,
        teacher_forcing_end=0.5,
        teacher_forcing_decay_steps=num_steps,
        num_steps=num_steps,
        eval_iters=eval_iters,
        checkpoint_path=checkpoint_path,
    )
