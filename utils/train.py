# Improved Training loop with learning rate scheduling and validation
import time
import torch
from pathlib import Path
from utils.translation_transformer import TransformerConfig
import csv
import os
from datetime import datetime


def load_from_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device=torch.device("cpu"),
):
    checkpoint = torch.load(path, map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint.get("step", 0)
    return model, optimizer, step


def log_training_step(
    csv_path: str,
    step: int,
    epoch: int,
    loss: float,
    optimizer: torch.optim.Optimizer,
    extra_metrics: dict | None = None,
):
    """
    Append the current training state to a CSV file.

    Args:
        csv_path (str): Path to CSV file
        step (int): Global training step
        epoch (int): Current epoch
        loss (float): Training loss
        optimizer (torch.optim.Optimizer): Optimizer (for LR logging)
        extra_metrics (dict, optional): Any extra scalar metrics to log
    """
    file_exists = os.path.isfile(csv_path)

    # Get current learning rate (first param group)
    lr = optimizer.param_groups[0]["lr"]

    row = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "epoch": epoch,
        "loss": float(loss),
        "lr": lr,
    }

    if extra_metrics:
        for k, v in extra_metrics.items():
            row[k] = float(v)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_model(
    model: torch.nn.Module,
    config: TransformerConfig,
    path: str,
    name: str,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
):
    # Create models directory
    model_dir = Path(path)
    model_dir.mkdir(exist_ok=True)

    # Save model state
    model_path = model_dir / name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": num_epochs,
            "model_config": {
                "d_model": config.d_model,
                "nhead": config.nhead,
                "num_encoder_layers": config.num_encoder_layers,
                "num_decoder_layers": config.num_decoder_layers,
                "dim_feedforward": config.dim_feedforward,
                "dropout": config.dropout,
                "max_len": config.max_len,
            },
        },
        model_path,
    )


@torch.no_grad()
def estimate_loss(
    model, test_loader, criterion, device, eval_iters, print_enabled=False
):
    model.eval()
    losses = []

    for k, batch in enumerate(test_loader):
        src, tgt, src_key_padding_mask, tgt_key_padding_mask = batch
        src_key_padding_mask = src_key_padding_mask.to(device)
        tgt_key_padding_mask = tgt_key_padding_mask.to(device)

        src = src.to(device)  # [batch, src_len]
        tgt = tgt.to(device)  # [batch, tgt_len]
        tgt_input = tgt[:, :-1]  # [batch, tgt_len-1]
        tgt_output = tgt[:, 1:]  # [batch, tgt_len-1]

        # Also shift the target padding mask
        tgt_input_mask = tgt_key_padding_mask[:, :-1]  # [batch, tgt_len-1]

        output = model(
            src,
            tgt_input,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_input_mask,
        )
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        # Calculate loss
        loss = criterion(output, tgt_output)

        losses.append(loss.item())
        if print_enabled:
            print(f"Eval batch {k+1}/{eval_iters}, Loss: {loss.item():.4f}\r", end="")
        if k + 1 >= eval_iters:
            break
    print()
    return sum(losses) / len(losses)


@torch.no_grad()
def greedy_decode_batch(
    model, src, src_key_padding_mask, max_len, device, sos_idx, pad_idx
):
    model.eval()
    batch_size = src.size(0)
    # start with SOS = 1 (adjust if your vocab differs)

    generated = torch.full((batch_size, 1), sos_idx, device=device, dtype=torch.long)
    pad_mask = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)

    for _ in range(max_len - 1):
        logits = model(
            src,
            generated,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=pad_mask,
        )
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        pad_mask = torch.cat([pad_mask, next_token.eq(pad_idx)], dim=1)

    model.train()
    return generated, pad_mask


def train(
    model,
    config: TransformerConfig,
    dataset_size: int,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device,
    sos_idx: int,
    pad_idx: int,
    teacher_forcing_start: float = 1.0,
    teacher_forcing_end: float = 0.1,
    teacher_forcing_decay_steps: int = 50_000,
    num_steps=15,
    eval_iters=10,  # Number of batches for loss estimation
    checkpoint_path: str | None = None,
):
    def teacher_forcing_ratio(step: int) -> float:
        # Linear decay; clamp at teacher_forcing_end
        slope = (teacher_forcing_start - teacher_forcing_end) / max(
            1, teacher_forcing_decay_steps
        )
        return max(teacher_forcing_end, teacher_forcing_start - slope * step)

    model.train()

    # Track training history
    train_losses = []
    validation_losses = []
    best_val_loss = float("inf")

    print(f"Starting training for {num_steps:,} steps...")
    print(f"Total batches per epoch: {len(train_loader):,}")
    print(f"Dataset size: {dataset_size:,} samples")
    print(
        f"Learning rate: {optimizer.param_groups[0]['lr']:.6f} (with warmup and decay)"
    )
    print("=" * 60)

    step = 0
    nstep_evaluation = 500
    break_training = False

    if checkpoint_path is not None:
        print(f"Loading model and optimizer state from checkpoint: {checkpoint_path}")
        model, optimizer, step = load_from_checkpoint(
            model, optimizer, checkpoint_path, device=device
        )
        scheduler.last_epoch = step
        print(f"Resuming training from step {step:,}")

    # Training loop
    while step < num_steps:
        if break_training:
            break

        start_time = time.time()

        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            if step >= num_steps:
                break

            src, tgt, src_key_padding_mask, tgt_key_padding_mask = batch

            src = src.to(device)  # [batch, src_len]
            tgt = tgt.to(device)  # [batch, tgt_len]
            tgt_output = tgt[:, 1:]  # [batch, tgt_len-1]
            src_input_mask = src_key_padding_mask.to(device)  # [batch, src_len]

            tf_ratio = teacher_forcing_ratio(step)
            use_teacher = torch.rand(1).item() < tf_ratio

            if use_teacher:
                tgt_input = tgt[:, :-1]
                tgt_input_mask = tgt_key_padding_mask[:, :-1]  # [batch, tgt_len-1]
            else:
                # Greedy decode to build decoder input without ground truth
                tgt_input, tgt_input_mask = greedy_decode_batch(
                    model,
                    src,
                    src_key_padding_mask,
                    max_len=tgt.size(1) - 1,
                    device=device,
                    sos_idx=sos_idx,
                    pad_idx=pad_idx,
                )

            tgt_input = tgt_input.to(device)
            tgt_input_mask = tgt_input_mask.to(device)  # [batch, tgt_len-1]

            # Zero gradients (set_to_none=True is faster than zero_grad())
            optimizer.zero_grad(set_to_none=True)

            # Forward pass with masks
            output = model(
                src,
                tgt_input,
                src_key_padding_mask=src_input_mask,
                tgt_key_padding_mask=tgt_input_mask,
            )  # [batch, tgt_len-1, vocab_size]

            # Reshape for loss calculation
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            # Calculate loss
            loss = criterion(output, tgt_output)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # Update weights
            optimizer.step()

            # Step the scheduler
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            step += 1
            print(
                f"[Step {step:,}/{num_steps:,}] - Training Loss: {loss.item():.4f}\r", end=""
            )
            
            # Print progress every n steps
            if step % nstep_evaluation == 0:
                nsteps_time = time.time() - start_time
                avg_loss = total_loss / num_batches
                train_losses.append(avg_loss)
                validation_loss = estimate_loss(
                    model, test_loader, criterion, device, eval_iters=eval_iters
                )
                validation_losses.append(validation_loss)
                current_lr = optimizer.param_groups[0]["lr"]
                
                print(
                    f"\n[Step {step:,}/{num_steps:,}] - Training Loss: {avg_loss:.4f}, Validation Loss: {validation_loss:.4f}, Time/step: {nsteps_time / nstep_evaluation:.2f}sec, lr: {current_lr:.8f}"
                )

                log_training_step(
                    csv_path="./training_log.csv",
                    step=step,
                    epoch=step // len(train_loader),
                    loss=avg_loss,
                    optimizer=optimizer,
                    extra_metrics={"tf_ratio": tf_ratio, "val_loss": validation_loss},
                )
                
                # Early stopping check
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    # Save best model
                    save_model(
                        model, config, "./models", "best_model.pt", optimizer, step
                    )

                total_loss = 0
                num_batches = 0

                start_time = time.time()

    print("\n✓ Training complete!")
    print(f"Final loss: {train_losses[-1]:.4f}")
    print(f"Best loss: {best_val_loss:.4f}")
    print(
        f"Loss improvement: {train_losses[0]:.4f} → {train_losses[-1]:.4f} ({(train_losses[0] - train_losses[-1]):.4f})"
    )
    return train_losses, validation_losses
