from datasets import load_dataset
from tokenizers import Tokenizer as HFTokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from utils.translation_transformer import TransformerConfig, TranslationTransformer
from utils.tokenization_vocab import HFTokenizerWrapper, Tokenizer
from utils.parallel_corpus import TranslationDataset, TranslationDataset2, DataLoaderFactory, LazyTranslationPairs
from utils.train import train
import os
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

ds = load_dataset("wmt/wmt14", "de-en")

vocab_size = 30_000
vocab_path = "./data/bpe_tokenizer.json"

training_samples = len(ds["train"])
batch_size = 64

dataset_max_sample_len = 100
sharedVocab = True

# training
num_steps = 100_000
warmup_steps = 2_000
eval_iters = 10
patience = 1_000

label_smoothing = 0.1

# optimizer
start_lr = 3e-4
betas = (0.9, 0.98)
epsilon = 1e-9

# bpe_v3_ep12
configSmall = TransformerConfig(
    d_model=256,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    max_len=150
)
# base model according to the paper 'Attention is all you need'
# big_3.8770loss
configBig = TransformerConfig(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    max_len=150
)

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    ds = load_dataset("wmt/wmt14", "de-en")

    # 1. Tokenizer setup

    bpe_tokenizer = HFTokenizer(BPE(unk_token=Tokenizer.UNK_TOKEN))
    trainer = BpeTrainer(
        special_tokens=[Tokenizer.PAD_TOKEN, Tokenizer.SOS_TOKEN, Tokenizer.EOS_TOKEN, Tokenizer.UNK_TOKEN],
        vocab_size=vocab_size,
        show_progress=True
    )

    bpe_tokenizer.pre_tokenizer = Metaspace()
    bpe_tokenizer.decoder = decoders.Metaspace()

    pretrained = True  # Set to True if you want to load a previously saved tokenizer

    Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(vocab_path).is_file():
        pretrained = True

    if pretrained:
        bpe_tokenizer = HFTokenizer.from_file(vocab_path)
    else:
        bpe_tokenizer.train(
            [
                './datasets/wmt14_translate_de-en_test.csv',
                './datasets/wmt14_translate_de-en_train.csv',
                './datasets/wmt14_translate_de-en_validation.csv',
            ],
            trainer=trainer
        )

        bpe_tokenizer.save(vocab_path)


    tokenizer = HFTokenizerWrapper(bpe_tokenizer)

    print(f"Vocab size: {bpe_tokenizer.get_vocab_size()}")

    # 2. Dataset and DataLoader setup

    # Create lazy wrappers - no materialization into lists!
    # train_src = LazyTranslationPairs(ds['train'], src_lang='de', tgt_lang='en', mode='src')
    # train_tgt = LazyTranslationPairs(ds['train'], src_lang='de', tgt_lang='en', mode='tgt')

    # test_src = LazyTranslationPairs(ds['test'], src_lang='de', tgt_lang='en', mode='src')
    # test_tgt = LazyTranslationPairs(ds['test'], src_lang='de', tgt_lang='en', mode='tgt')

    print("Tokenizing dataset (this may take a few minutes, will be cached for future runs)...")

    def tokenize_function(examples):
        return {
            'source_indices': tokenizer.encode_batch([e['de'] for e in examples['translation']]),
            'target_indices': tokenizer.encode_batch([e['en'] for e in examples['translation']])
        }

    indexed_ds = ds.map(
        tokenize_function,
        batched=True,
        batch_size=10000,
        num_proc=8,
        remove_columns=['translation'],
        load_from_cache_file=True,  # Use cached results if available
        desc="Tokenizing"
    )

    # Convert lists of ids to torch.Tensor on access (still stored compactly in Arrow)
    indexed_ds = indexed_ds.with_format(
        type='torch',
        columns=['source_indices', 'target_indices'],
        output_all_columns=False,
    )

    # Create datasets with lazy loading (processes on-the-fly, no upfront preprocessing)
    train_ds = TranslationDataset2(
        source_sentences=indexed_ds['train']['source_indices'],
        target_sentences=indexed_ds['train']['target_indices'],
        max_length=dataset_max_sample_len
    )

    test_ds = TranslationDataset2(
        source_sentences=indexed_ds['test']['source_indices'],
        target_sentences=indexed_ds['test']['target_indices'],
        max_length=dataset_max_sample_len
    )

    # Optimize num_workers based on CPU cores
    optimal_workers = min(8, os.cpu_count() or 4)

    train_loader = DataLoaderFactory.create_dataloader(
        dataset=train_ds,
        batch_size=batch_size,
        pad_idx=tokenizer.pad_idx,
        num_workers=optimal_workers,
        shuffle=True,  # Shuffle for training
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4  # Prefetch more batches
    )

    test_loader = DataLoaderFactory.create_dataloader(
        dataset=test_ds,
        batch_size=batch_size,
        pad_idx=tokenizer.pad_idx,
        num_workers=optimal_workers,
        shuffle=False,  # No shuffle for testing
        persistent_workers=True,
        prefetch_factor=4
    )

    print(f"✓ Using {optimal_workers} workers for parallel processing")
    print(f"Train samples: {len(train_ds):,}, Test samples: {len(test_ds):,}")
    print(f"Train batches: {len(train_loader):,}, Test batches: {len(test_loader):,}")

    # Initialize the model with larger max_len to handle max_length + special tokens
    model = TranslationTransformer(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
        config=configBig,
        padding_idx=tokenizer.pad_idx,
        sharedVocab=sharedVocab
    )

    print(f"Model initialized!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. For MPS (Apple Silicon), ensure we're using optimal settings
    if DEVICE.type == "mps":
        torch.mps.empty_cache()  # Clear any cached memory
    elif DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("✓ Running on CPU (no GPU optimizations)")

    model_compiled = torch.compile(model, mode='default')

    # Move model to device (GPU if available)
    model = model.to(DEVICE)
    model.train()

    print(f"Using device: {DEVICE}")
    print(f"Model moved to {DEVICE}")

    def lr_lambda(step, warmup_steps=4000):
        step = max(step, 1)
        return configBig.d_model**(-0.5) * min(
            step ** -0.5,
            step * warmup_steps ** -1.5
        )

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx, label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=1, betas=betas, eps=epsilon)

    scheduler = LambdaLR(optimizer, lambda step: lr_lambda(step, warmup_steps))

    # Training
    train_losses, best_loss = train(
        model=model_compiled,
        config=configBig,
        train_loader=train_loader,
        test_loader=test_loader,
        dataset_size=len(train_ds),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        num_steps=num_steps,
        eval_iters=eval_iters,
        patience=patience,
        checkpoint_path="./models/aiayn_base_100k.pt"
    )


if __name__ == "__main__":
    main()