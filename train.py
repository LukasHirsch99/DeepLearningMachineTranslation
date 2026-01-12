# Improved Training loop with learning rate scheduling and validation
import time
import torch
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR


def save_model(
    model: torch.nn.Module,
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
        'epoch': num_epochs,
        'model_config': {
            'd_model': 256,
            'nhead': 8,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'max_len': 150
        }
    }, model_path)

    print(f"✓ Model saved to: {model_path}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  File size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")

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


def lr_lambda(step, warmup_steps=2000):
    """Learning rate schedule with warmup and decay."""
    step = max(step, 1)
    return min(
        step ** -0.5,
        step * warmup_steps ** -1.5
    )

def train(
    model, dataloader, dataset_size, train_loader, test_loader,
    criterion, optimizer, device,
    num_epochs = 15,
    warmup_steps = 2000,  # 2 epochs
    eval_iters = 10,  # Number of batches for loss estimation
    patience = 3  # Number of epochs to wait for improvement before stopping
):
    model.train()
    # scheduler = LambdaLR(optimizer, lambda step: lr_lambda(step, warmup_steps))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=2000
    )

    # Track training history
    train_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Total batches per epoch: {len(dataloader)}")
    print(f"Dataset size: {dataset_size} samples")
    print(f"Learning rate: {scheduler.get_last_lr()[0]} (with warmup and decay)")
    print(f"=" * 60)

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch (first epoch will be slower due to compilation)
        if epoch == 0 and 'model_compiled' in dir():
            print("  ⏳ Compiling model on first batch (this will take extra time)...")
        avg_loss = train_epoch(model, train_loader, test_loader, criterion, optimizer, device, scheduler)

        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - start_time
        train_losses.append(avg_loss)

        val_loss = estimate_loss(model, test_loader, criterion, device, eval_iters=eval_iters)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save best model
            save_model(model, './models', 'best_model.pt', optimizer, epoch+1)
            print(f"  ✓ New best validation loss! Model saved.")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
        
        if epochs_without_improvement >= patience:
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break

        print(f"\n  Average Loss: {avg_loss:.4f}")
        print(f"\n  Validation Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Samples/sec: ~{dataset_size / epoch_time:.0f}")
        print(f"  Best Loss So Far: {best_val_loss:.4f}")
        print(f"=" * 60)
        
    print("\n✓ Training complete!")
    print(f"Final loss: {train_losses[-1]:.4f}")
    print(f"Best loss: {best_val_loss:.4f}")
    print(f"Loss improvement: {train_losses[0]:.4f} → {train_losses[-1]:.4f} ({(train_losses[0] - train_losses[-1]):.4f})")
    return train_losses, best_val_loss