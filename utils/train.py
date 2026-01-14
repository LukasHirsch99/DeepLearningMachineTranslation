# Improved Training loop with learning rate scheduling and validation
import time
import torch
from pathlib import Path
from .translation_transformer import TransformerConfig
import csv
import os
from datetime import datetime


def log_training_step(
    csv_path: str,
    step: int,
    epoch: int,
    loss: float,
    optimizer,
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
    num_epochs: int
):
    # Create models directory
    model_dir = Path(path)
    model_dir.mkdir(exist_ok=True)

    # Save model state
    model_path = model_dir / name
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': num_epochs,
        'model_config': {
            'd_model': config.d_model,
            'nhead': config.nhead,
            'num_encoder_layers': config.num_encoder_layers,
            'num_decoder_layers': config.num_decoder_layers,
            'dim_feedforward': config.dim_feedforward,
            'dropout': config.dropout,
            'max_len': config.max_len
        }
    }, model_path)

def train_epoch(model, dataloader, test_loader, criterion, optimizer, device, scheduler):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for i, batch in enumerate(dataloader):

        src, tgt, src_key_padding_mask, tgt_key_padding_mask = batch
        src_key_padding_mask = src_key_padding_mask.to(device)
        tgt_key_padding_mask = tgt_key_padding_mask.to(device)
        
        # tokens... <eos>
        src = src.to(device)  # [batch, src_len]
        # <sos> tokens... <eos>
        tgt = tgt.to(device)  # [batch, tgt_len]
        
        # Shift target for decoder input and labels
        # <sos> tokens...
        tgt_input = tgt[:, :-1]  # [batch, tgt_len-1]
        # tokens... <eos>
        tgt_output = tgt[:, 1:]  # [batch, tgt_len-1]
        
        # Also shift the target padding mask
        tgt_input_mask = tgt_key_padding_mask[:, :-1]  # [batch, tgt_len-1]
        
        # Zero gradients (set_to_none=True is faster than zero_grad())
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with masks
        output = model(
            src,
            tgt_input,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_input_mask
        )  # [batch, tgt_len-1, vocab_size]
        
        # Reshape for loss calculation
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)
        
        # Calculate loss
        loss = criterion(output, tgt_output)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
     
        # Step the scheduler
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1

        # Print progress every 50 batches
        if (i + 1) % 50 == 0:
            avg_loss_so_far = total_loss / num_batches
            # validation_loss = estimate_loss(model, test_loader, criterion, device, eval_iters=1)
            # print(f"  [Batch {i+1}/{len(dataloader)}] - Training Loss: {avg_loss_so_far:.4f}, Validation Loss: {validation_loss:.4f}")
            print(f"  [Batch {i+1}/{len(dataloader)}] - Training Loss: {avg_loss_so_far:.4f}")
    
    return total_loss / num_batches

@torch.no_grad()
def estimate_loss(model, test_loader, criterion, device, eval_iters):
    model.eval()
    losses = []

    for k, batch in enumerate(test_loader):
        if k >= eval_iters:
            break
        
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
            tgt_key_padding_mask=tgt_input_mask
        )
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)
        
        # Calculate loss
        loss = criterion(output, tgt_output)

        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


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
    num_steps = 15,
    eval_iters = 10,  # Number of batches for loss estimation
    patience = 3  # Number of epochs to wait for improvement before stopping
):
    model.train()

    # Track training history
    train_losses = []
    validation_losses = []
    best_val_loss = float('inf')
    steps_without_improvement = 0
    
    print(f"Starting training for {num_steps:,} steps...")
    print(f"Total batches per epoch: {len(train_loader)}")
    print(f"Dataset size: {dataset_size} samples")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f} (with warmup and decay)")
    print(f"=" * 60)

    step = 0
    nstep_evaluation = 100
    break_training = False

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
            src_key_padding_mask = src_key_padding_mask.to(device)
            tgt_key_padding_mask = tgt_key_padding_mask.to(device)
            
            # tokens... <eos>
            src = src.to(device)  # [batch, src_len]
            # <sos> tokens... <eos>
            tgt = tgt.to(device)  # [batch, tgt_len]
            
            # Shift target for decoder input and labels
            # <sos> tokens...
            tgt_input = tgt[:, :-1]  # [batch, tgt_len-1]
            # tokens... <eos>
            tgt_output = tgt[:, 1:]  # [batch, tgt_len-1]
            
            # Also shift the target padding mask
            tgt_input_mask = tgt_key_padding_mask[:, :-1]  # [batch, tgt_len-1]
            
            # Zero gradients (set_to_none=True is faster than zero_grad())
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with masks
            output = model(
                src,
                tgt_input,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_input_mask
            )  # [batch, tgt_len-1, vocab_size]
            
            # Reshape for loss calculation
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            
            # Calculate loss
            loss = criterion(output, tgt_output)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
        
            # Step the scheduler
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1

            step += 1
            # Print progress every 100 steps
            log_training_step(
                csv_path="./training_log.csv",
                step=step,
                epoch=step // len(train_loader),
                loss=loss.item(),
                optimizer=optimizer
            )
            if step % nstep_evaluation == 0:
                nsteps_time = time.time() - start_time
                avg_loss = total_loss / num_batches
                train_losses.append(avg_loss)
                validation_loss = estimate_loss(model, test_loader, criterion, device, eval_iters=eval_iters)
                validation_losses.append(validation_loss)
                current_lr = optimizer.param_groups[0]['lr']

                print(f"[Step {step}/{len(train_loader)}] - Training Loss: {avg_loss:.4f}, Validation Loss: {validation_loss:.4f}, Time/step: {nsteps_time / nstep_evaluation:.2f}sec, lr: {current_lr:.8f}")
        
                # Early stopping check
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    steps_without_improvement = 0
                    # Save best model
                    save_model(model, config, './models', 'best_model.pt', optimizer, step)
                else:
                    steps_without_improvement += 1
                    print(f"No improvement for {steps_without_improvement} steps(s)")
                
                if steps_without_improvement >= patience:
                    print(f"\n⚠ Early stopping triggered after {step} steps")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    break_training = True
                    break

                total_loss = 0
                num_batches = 0

                start_time = time.time()
        
    print("\n✓ Training complete!")
    print(f"Final loss: {train_losses[-1]:.4f}")
    print(f"Best loss: {best_val_loss:.4f}")
    print(f"Loss improvement: {train_losses[0]:.4f} → {train_losses[-1]:.4f} ({(train_losses[0] - train_losses[-1]):.4f})")
    return train_losses, validation_losses