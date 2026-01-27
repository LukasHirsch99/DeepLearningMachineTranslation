import torch
import torch.nn as nn
from torch.optim import Adam
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def calculate_accuracy(predictions, targets, pad_id):
  mask = targets != pad_id
  correct = ((predictions.argmax(dim=-1) == targets) & mask).sum().item()
  total = mask.sum().item()
  return correct / total if total > 0 else 0.0

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast  # Add these imports
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def train_model(model, train_loader, val_loader, tokenizer, checkpoint_manager, 
                num_epochs=10, learning_rate=0.001, teacher_forcing_ratio=0.5, 
                clip_grad=1.0, patience=3, resume_from=None, checkpoint_every=1,
                use_amp=True):  # Add AMP flag
    pad_id = tokenizer.pad_id
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Initialize AMP scaler if using CUDA
    if use_amp and device.type == 'cuda':
        scaler = GradScaler()
        print("Using mixed precision training (AMP)")
    else:
        scaler = None
        print("Using standard precision training")

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    start_epoch = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'learning_rates': []
    }

    if resume_from is not None:
        try:
            if resume_from == 'latest':
                checkpoint_path = checkpoint_manager.find_latest_checkpoint()
                if checkpoint_path is None:
                    print("No latest checkpoint found. Starting from scratch.")
                else:
                    print(f"Resuming from latest checkpoint: {checkpoint_path}")
            elif resume_from == 'best':
                checkpoint_path = os.path.join(checkpoint_manager.checkpoint_dir, 'best.pth')
                if not os.path.exists(checkpoint_path):
                    print("No best checkpoint found. Starting from scratch.")
                else:
                    print(f"Resuming from best checkpoint: {checkpoint_path}")
            else:
                checkpoint_path = resume_from

            if os.path.exists(checkpoint_path):
                checkpoint = checkpoint_manager.load(checkpoint_path, model, optimizer)
                start_epoch = checkpoint['epoch'] + 1

                if 'history' in checkpoint:
                    history = checkpoint['history']
                    print(f"Restored training history from epoch {start_epoch}")

                # Restore scaler state if available
                if scaler and 'scaler_state' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state'])
                    print("Restored AMP scaler state")

                print(f"Resuming training from epoch {start_epoch}/{num_epochs}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting training from scratch...")

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}" +
              (f" (Resumed from epoch {start_epoch})" if epoch == start_epoch and start_epoch > 0 else ""))
        
        # Training phase
        model.train()
        epoch_train_loss = 0
        epoch_train_correct = 0
        epoch_train_total = 0
        batch_count = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training",
                          unit="batch", leave=True)

        for batch_idx, batch in enumerate(train_pbar):
            src_tokens = batch['src_tokens'].to(device)
            src_lens = batch['src_lens'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)

            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if scaler:
                with autocast():
                    logits = model(src_tokens, src_lens, tgt_input, teacher_forcing_ratio)
                    loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.view(-1))
                
                # Scale loss and backward
                scaler.scale(loss).backward()
                
                # Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training
                logits = model(src_tokens, src_lens, tgt_input, teacher_forcing_ratio)
                loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.view(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

            accuracy = calculate_accuracy(logits, tgt_output, pad_id)

            batch_size = src_tokens.size(0)
            epoch_train_loss += loss.item() * batch_size
            epoch_train_correct += accuracy * batch_size * tgt_output.size(1)  
            epoch_train_total += batch_size
            batch_count += 1

            # Get current scale for monitoring
            current_scale = scaler.get_scale() if scaler else 1.0
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.2%}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'scale': f'{current_scale:.1f}' if scaler else ''
            })

        train_pbar.close()

        # Validation phase (always use full precision)
        model.eval()
        epoch_val_loss = 0
        epoch_val_correct = 0
        epoch_val_total = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation",
                        unit="batch", leave=True)

        with torch.no_grad():
            for batch in val_pbar:
                src_tokens = batch['src_tokens'].to(device)
                src_lens = batch['src_lens'].to(device)
                tgt_input = batch['tgt_input'].to(device)
                tgt_output = batch['tgt_output'].to(device)

                # Always use full precision for validation
                logits = model(src_tokens, src_lens, tgt_input, teacher_forcing_ratio=0)
                loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.view(-1))

                accuracy = calculate_accuracy(logits, tgt_output, pad_id)

                batch_size = src_tokens.size(0)
                epoch_val_loss += loss.item() * batch_size
                epoch_val_correct += accuracy * batch_size * tgt_output.size(1)  
                epoch_val_total += batch_size

                val_pbar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'val_acc': f'{accuracy:.2%}'
                })

        val_pbar.close()

        # Calculate averages
        avg_train_loss = epoch_train_loss / epoch_train_total if epoch_train_total > 0 else 0
        train_accuracy = epoch_train_correct / epoch_train_total if epoch_train_total > 0 else 0
        avg_val_loss = epoch_val_loss / epoch_val_total if epoch_val_total > 0 else 0
        val_accuracy = epoch_val_correct / epoch_val_total if epoch_val_total > 0 else 0

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Training - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2%}")
        print(f"  Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2%}")
        
        if scaler:
            print(f"  AMP Scale: {scaler.get_scale():.1f}")

        is_best = avg_val_loss < best_val_loss

        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0 or is_best:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss,
                'config': {
                    'vocab_size': model.vocab_size,
                    'hidden_size': model.hidden_size,
                    'embedding_dim': model.embedding_dim,
                    'num_layers': model.num_layers
                }
            }
            
            # Add scaler state if using AMP
            if scaler:
                checkpoint_data['scaler_state'] = scaler.state_dict()
            
            checkpoint_path = checkpoint_manager.save(
                checkpoint_data=checkpoint_data,
                is_best=is_best
            )
            print(f"Checkpoint saved: {checkpoint_path}")

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break

    print("Training Complete!")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

    # Save final checkpoint
    final_checkpoint_data = {
        'epoch': num_epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'best_val_loss': best_val_loss,
        'config': {
            'vocab_size': model.vocab_size,
            'hidden_size': model.hidden_size,
            'embedding_dim': model.embedding_dim,
            'num_layers': model.num_layers
        }
    }
    
    if scaler:
        final_checkpoint_data['scaler_state'] = scaler.state_dict()
    
    final_checkpoint = checkpoint_manager.save(
        checkpoint_data=final_checkpoint_data,
        is_best=False
    )
    print(f"Final checkpoint saved: {final_checkpoint}")

    print(f"\nFinal Statistics:")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final Training Accuracy: {history['train_accuracy'][-1]:.2%}")
    print(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.2%}")

    plot_training_history(history)

    return model, history

def plot_training_history(history):
  fig, axes = plt.subplots(1, 3, figsize=(15, 4))
  epochs = range(1, len(history['train_loss']) + 1)
  axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
  axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
  axes[0].set_xlabel('Epochs')
  axes[0].set_ylabel('Loss')
  axes[0].set_title('Training and Validation Loss')
  axes[0].legend()
  axes[0].grid(True, alpha=0.3)

  axes[1].plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy')
  axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
  axes[1].set_xlabel('Epochs')

  axes[1].set_ylabel('Accuracy')
  axes[1].set_title('Training and Validation Accuracy')
  axes[1].legend()
  axes[1].grid(True, alpha=0.3)

  axes[2].plot(epochs, history['learning_rates'], 'g-', label='Learning Rate')
  axes[2].set_xlabel('Epochs')
  axes[2].set_ylabel('Learning Rate')
  axes[2].set_title('Learning Rate Schedule')
  axes[2].set_yscale('log')
  axes[2].legend()
  axes[2].grid(True, alpha=0.3)

  plt.tight_layout()
  plt.show()

def translate_examples(model, dataloader, tokenizer, num_examples=5):
  device = next(model.parameters()).device
  model.eval()

  print("Sample Translations:")

  examples_translated = 0
  with torch.no_grad():
      for batch in dataloader:
          src = batch['src_tokens'].to(device)
          src_lens = batch['src_lens'].to(device)
          tgt_output = batch['tgt_output']

          predictions = model.inference(src[:num_examples], src_lens[:num_examples], max_len=50)

          for i in range(min(num_examples, len(src))):
              source_tokens = src[i].cpu().tolist()
              source_text = tokenizer.decode(source_tokens)

              target_tokens = tgt_output[i].cpu().tolist()
              target_text = tokenizer.decode(target_tokens)

              pred_tokens = predictions[i].cpu().tolist()
              pred_text = tokenizer.decode(pred_tokens)

              print(f"\nExample {examples_translated + 1}:")
              print(f"Source: {source_text}")
              print(f"Target: {target_text}")
              print(f"Prediction: {pred_text}")

              examples_translated += 1
              if examples_translated >= num_examples:
                  break

          if examples_translated >= num_examples:
              break