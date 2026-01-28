import torch
import torch.nn as nn
import os
import csv
from pathlib import Path
from datetime import datetime

# ---------- Checkpointing & Logging ----------

def save_model(model, path: str, optimizer=None, epoch=None, name="model.pt"):
    model_dir = Path(path)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_dict = {'model_state_dict': model.state_dict()}
    if optimizer:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        save_dict['epoch'] = epoch
    torch.save(save_dict, model_dir / name)
    print(f"Model saved at {model_dir / name}")


def load_from_checkpoint(model, optimizer=None, path=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {path}, starting from epoch {epoch}")
    return model, optimizer, epoch


def log_training_step(csv_path, step, epoch, loss, optimizer, extra_metrics=None):
    file_exists = os.path.isfile(csv_path)
    lr = optimizer.param_groups[0]['lr'] if optimizer else None

    row = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "epoch": epoch,
        "loss": float(loss),
        "lr": lr
    }

    if extra_metrics:
        for k, v in extra_metrics.items():
            row[k] = float(v)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------- Validation ----------

@torch.no_grad()
def estimate_loss(model, criterion, dataloader, device, max_batches=None):
    model.eval()
    losses = []
    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break
        src, tgt, src_mask, tgt_mask, src_lengths, tgt_lengths = batch
        src, tgt = src.to(device), tgt.to(device)
        tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
        output, _ = model(src, tgt_input, src_lengths)
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)
        loss = criterion(output, tgt_output)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ---------- LSTM Training Loop ----------

def train_lstm(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs=10,
    teacher_forcing_ratio=0.5,
    scheduler=None,
    checkpoint_path=None,
    csv_log_path="./logs/training_log.csv",
    early_stop_patience=3,
    max_eval_batches=10
):
    model.to(device)
    step = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    start_epoch = 0
    if checkpoint_path:
        model, optimizer, start_epoch = load_from_checkpoint(model, optimizer, checkpoint_path)

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            step += 1
            src, tgt, src_mask, tgt_mask, src_lengths, tgt_lengths = batch
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()

            # Teacher forcing
            if torch.rand(1).item() < teacher_forcing_ratio:
                tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
            else:
                tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]

            output, _ = model(src, tgt_input, src_lengths)
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

            epoch_loss += loss.item()
            log_training_step(csv_log_path, step, epoch, loss.item(), optimizer)

        avg_epoch_loss = epoch_loss / len(train_loader)
        val_loss = estimate_loss(model, criterion, val_loader, device, max_batches=max_eval_batches)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_model(model, "./models", optimizer, epoch, name="best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    print("✅ Training complete.")
    return model
