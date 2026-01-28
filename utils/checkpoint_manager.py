import torch
import os
import glob

class CheckpointManager:
  def __init__(self, checkpoint_dir='./checkpoints'):
    """
    Initialize CheckpointManager.
    
    Args:
        checkpoint_dir (str): Directory to save checkpoints
    """
    self.checkpoint_dir = checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
  
  def save(self, epoch, model, optimizer, history, is_best=False):
    """
    Save a checkpoint.
    
    Args:
        epoch (int): Current epoch number
        model (nn.Module): PyTorch model
        optimizer (torch.optim.Optimizer): Optimizer
        history (dict): Training history
        is_best (bool): Whether this is the best model so far
        
    Returns:
        str: Path to the saved checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }
    
    if history and 'val_loss' in history and history['val_loss']:
        checkpoint['val_loss'] = history['val_loss'][-1]
    
    checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1:03d}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {os.path.basename(checkpoint_path)}")
    
    if is_best:
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {os.path.basename(best_path)}")
        
        model_path = os.path.join(self.checkpoint_dir, 'best_model_weights.pth')
        torch.save(model.state_dict(), model_path)
    
    self._cleanup_old_checkpoints(max_keep=5)
    
    return checkpoint_path

  def load(self, checkpoint_path, model, optimizer=None):
    """
    Load a checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model (nn.Module): Model to load weights into
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into
        
    Returns:
        dict: Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    
    return checkpoint
  
  def find_latest_checkpoint(self):
    """
    Find the latest checkpoint based on epoch number.
    
    Returns:
        str or None: Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoint_pattern = os.path.join(self.checkpoint_dir, 'epoch_*.pth')
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    def extract_epoch(path):
        filename = os.path.basename(path)
        return int(filename.split('_')[-1].split('.')[0])
    
    latest_checkpoint = max(checkpoints, key=extract_epoch)
    return latest_checkpoint
  
  def load_best_model(self, model):
    """
    Load the best model weights.
    
    Args:
        model (nn.Module): Model to load weights into
        
    Returns:
        dict: Checkpoint dictionary
    """
    best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
    
    if not os.path.exists(best_path):
        weights_path = os.path.join(self.checkpoint_dir, 'best_model_weights.pth')
        if os.path.exists(weights_path):
          model.load_state_dict(torch.load(weights_path, map_location='cpu'))
          print("Loaded best model weights")
          return {'epoch': -1} 
        else:
          raise FileNotFoundError(f"No best model found in {self.checkpoint_dir}")
    
    checkpoint = torch.load(best_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
    return checkpoint
  
  def get_checkpoint_info(self):
    """
    Get information about available checkpoints.
    
    Returns:
        dict: Information about checkpoints
    """
    checkpoints = {}
    
    checkpoint_pattern = os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pth')
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    for filepath in checkpoint_files:
      filename = os.path.basename(filepath)
      try:
        epoch_num = int(filename.split('_')[-1].split('.')[0])
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        checkpoints[epoch_num] = {
            'path': filepath,
            'epoch': checkpoint.get('epoch', epoch_num - 1),
            'val_loss': checkpoint.get('val_loss', float('inf')),
            'size_mb': os.path.getsize(filepath) / (1024 * 1024)
        }
      except:
        continue
    
    best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_path):
      checkpoint = torch.load(best_path, map_location='cpu')
      checkpoints['best'] = {
        'path': best_path,
        'epoch': checkpoint.get('epoch', -1),
        'val_loss': checkpoint.get('val_loss', float('inf')),
        'size_mb': os.path.getsize(best_path) / (1024 * 1024)
      }
    
    return checkpoints
  
  def _cleanup_old_checkpoints(self, max_keep=5):
    """
    Keep only the last N checkpoints to save disk space.
    
    Args:
        max_keep (int): Maximum number of checkpoints to keep
    """
    checkpoint_pattern = os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pth')
    checkpoints = glob.glob(checkpoint_pattern)
    
    if len(checkpoints) <= max_keep:
      return
    
    def extract_epoch(path):
      filename = os.path.basename(path)
      return int(filename.split('_')[-1].split('.')[0])
    
    checkpoints.sort(key=extract_epoch, reverse=True)
    
    for checkpoint in checkpoints[max_keep:]:
      try:
        os.remove(checkpoint)
        print(f"Removed old checkpoint: {os.path.basename(checkpoint)}")
      except:
        pass