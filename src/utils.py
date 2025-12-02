"""
Utility Functions for Fast CIFAR-10 Training

This module implements:
- Patch whitening initialization
- Model evaluation with test-time augmentation
- Logging utilities
- Device detection
"""

import time

import torch
import torch.nn.functional as F


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


# ============================================
# Patch Whitening Initialization
# ============================================

def get_patches(x, patch_shape):
    """
    Extract all patches of given shape from a batch of images.
    
    Args:
        x: Tensor of shape (N, C, H, W)
        patch_shape: Tuple (h, w) for patch size
    
    Returns:
        Tensor of shape (N * num_patches, C, h, w)
    """
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()


def get_whitening_parameters(patches):
    """
    Compute whitening transformation from patches.
    
    Args:
        patches: Tensor of shape (N, C, H, W)
    
    Returns:
        eigenvalues: Sorted in descending order
        eigenvectors: Corresponding eigenvectors as filters
    """
    n, c, h, w = patches.shape
    original_device = patches.device
    
    # Move to CPU for eigendecomposition (not supported on MPS)
    patches = patches.cpu()
    patches_flat = patches.view(n, -1)
    
    # Compute covariance matrix
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    
    # Eigendecomposition (on CPU)
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    
    # Sort in descending order and reshape to filter format
    eigenvalues = eigenvalues.flip(0).view(-1, 1, 1, 1)
    eigenvectors = eigenvectors.T.reshape(c * h * w, c, h, w).flip(0)
    
    # Move back to original device
    return eigenvalues.to(original_device), eigenvectors.to(original_device)


def init_whitening_conv(layer, train_set, eps=5e-4):
    """
    Initialize a convolutional layer as a patch-whitening transformation.
    
    The layer will whiten 2x2 patches of the input images. The first half
    of filters are the whitening eigenvectors, the second half are their
    negations (to preserve information through the following activation).
    
    Args:
        layer: nn.Conv2d to initialize (should have 24 output channels for 2x2 patches)
        train_set: Tensor of training images (N, C, H, W) for computing statistics
        eps: Small constant for numerical stability
    """
    # Extract patches matching the layer's kernel size
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    
    # Compute whitening parameters
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    
    # Scale eigenvectors to whiten (divide by sqrt of eigenvalues)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    
    # Set weights: first 12 filters are whitening, next 12 are negated
    layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))


# ============================================
# Evaluation with Test-Time Augmentation
# ============================================

def infer(model, loader, tta_level=0):
    """
    Run inference with optional test-time augmentation.
    
    TTA levels:
    - 0: No augmentation
    - 1: Horizontal flip (average original and flipped)
    - 2: Horizontal flip + translations (6 views total)
    
    Args:
        model: Trained model
        loader: CifarLoader with test data
        tta_level: Level of test-time augmentation (0, 1, or 2)
    
    Returns:
        Tensor of logits for all test examples
    """
    def infer_basic(inputs, net):
        return net(inputs).clone()
    
    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))
    
    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        
        # Pad and translate
        pad = 1
        padded_inputs = F.pad(inputs, (pad,) * 4, 'reflect')
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],   # Up-left
            padded_inputs[:, :, 2:34, 2:34],  # Down-right
        ]
        
        logits_translate_list = [
            infer_mirror(inputs_translate, net)
            for inputs_translate in inputs_translate_list
        ]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        
        return 0.5 * logits + 0.5 * logits_translate
    
    model.eval()
    test_images = loader.normalize(loader.images)
    
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])


def evaluate(model, loader, tta_level=0):
    """
    Evaluate model accuracy on a dataset.
    
    Args:
        model: Trained model
        loader: CifarLoader with test data
        tta_level: Level of test-time augmentation
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()


def evaluate_standard(model, loader, device=None):
    """
    Evaluate model using standard dataloader (for baseline).
    
    Args:
        model: Trained model
        loader: Standard PyTorch DataLoader
        device: Device to run on
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    if device is None:
        device = get_device()
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return correct / total


# ============================================
# Timing Utilities
# ============================================

class Timer:
    """
    Cross-platform timer that works with CUDA, MPS, and CPU.
    
    Uses proper synchronization for accurate GPU timing.
    """
    
    def __init__(self, device=None):
        self.device = device or get_device()
        self.total_time = 0.0
        self._start_time = None
    
    def start(self):
        """Start timing."""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        elif self.device == 'mps':
            torch.mps.synchronize()
        self._start_time = time.perf_counter()
    
    def stop(self):
        """Stop timing and accumulate."""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        elif self.device == 'mps':
            torch.mps.synchronize()
        self.total_time += time.perf_counter() - self._start_time
    
    def reset(self):
        """Reset accumulated time."""
        self.total_time = 0.0


# ============================================
# Logging Utilities
# ============================================

LOGGING_COLUMNS = ['run', 'epoch', 'train_loss', 'train_acc', 'val_acc', 'tta_val_acc', 'time']


def print_columns(columns_list, is_head=False, is_final_entry=False):
    """Print a formatted table row."""
    print_string = ''
    for col in columns_list:
        print_string += '| %s ' % col
    print_string += '|'
    
    if is_head:
        print('-' * len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print('-' * len(print_string))


def print_training_details(variables, is_final_entry=False):
    """Print training metrics in table format."""
    formatted = []
    for col in LOGGING_COLUMNS:
        var = variables.get(col.strip(), None)
        if isinstance(var, int):
            res = str(var)
        elif isinstance(var, str):
            res = var
        elif isinstance(var, float):
            res = '{:0.4f}'.format(var)
        else:
            res = ''
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)


def print_header():
    """Print the table header."""
    print_columns(LOGGING_COLUMNS, is_head=True)

