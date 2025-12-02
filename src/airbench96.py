#!/usr/bin/env python3
"""
airbench96: 96% Accuracy on CIFAR-10

This script implements the highest accuracy training method from the paper:
"94% on CIFAR-10 in 3.29 Seconds on a Single GPU" (arXiv:2404.00498)

Modifications from airbench94:
- Increased training epochs: 9.9 -> 40
- Much wider network with residual connections:
  - Block 1: 128 channels (3 convs)
  - Block 2-3: 512 channels (3 convs each)
  - Residual connections across last 2 convs of each block
- Added 12-pixel Cutout augmentation
- Reduced learning rate by factor of 0.78

Target: 96.05% accuracy in ~40 epochs (~46.3s on A100)

Usage:
    python -m src.airbench96 [--runs 5]
"""

import argparse
from math import ceil

import torch
from torch import nn

from .data import CifarLoader
from .models import make_net_96
from .optim import LookaheadState, triangle_schedule, alpha_schedule, get_optimizer
from .utils import (
    init_whitening_conv, evaluate, Timer,
    print_header, print_training_details, get_device
)


# Hyperparameters for 96% target
HYP = {
    'opt': {
        'train_epochs': 40,
        'batch_size': 1024,
        'lr': 11.5 * 0.78,  # reduced by factor of 0.78
        'momentum': 0.85,
        'weight_decay': 0.0153,
        'bias_scaler': 64.0,
        'label_smoothing': 0.2,
        'whiten_bias_epochs': 3,
    },
    'aug': {
        'flip': True,
        'translate': 2,
        'cutout': 12,  # 12-pixel cutout
    },
    'net': {
        'batchnorm_momentum': 0.6,
        'scaling_factor': 1/9,
        'tta_level': 2,
    },
}


def apply_cutout(images, cutout_size):
    """
    Apply cutout augmentation to a batch of images.
    
    Cutout randomly masks out a square region of each image.
    
    Args:
        images: Tensor of shape (N, C, H, W)
        cutout_size: Size of the square mask
    
    Returns:
        Images with cutout applied
    """
    n, c, h, w = images.shape
    
    # Random center for each image
    cy = torch.randint(0, h, (n,), device=images.device)
    cx = torch.randint(0, w, (n,), device=images.device)
    
    # Create mask
    y = torch.arange(h, device=images.device).view(1, 1, h, 1)
    x = torch.arange(w, device=images.device).view(1, 1, 1, w)
    
    cy = cy.view(n, 1, 1, 1)
    cx = cx.view(n, 1, 1, 1)
    
    mask = (
        (y >= cy - cutout_size // 2) & (y < cy + cutout_size // 2) &
        (x >= cx - cutout_size // 2) & (x < cx + cutout_size // 2)
    )
    
    return images.masked_fill(mask, 0)


class CifarLoaderWithCutout(CifarLoader):
    """CifarLoader extended with Cutout augmentation."""
    
    def __init__(self, *args, cutout_size=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cutout_size = cutout_size
    
    def __iter__(self):
        for images, labels in super().__iter__():
            if self.cutout_size > 0:
                images = apply_cutout(images, self.cutout_size)
            yield images, labels


def train96(run_id=None, device=None, verbose=True):
    """
    Train the airbench96 model.
    
    Args:
        run_id: Run identifier for logging
        device: Device to train on
        verbose: Whether to print training progress
    
    Returns:
        dict with 'accuracy' and 'time' keys
    """
    device = device or get_device()
    
    batch_size = HYP['opt']['batch_size']
    epochs = HYP['opt']['train_epochs']
    momentum = HYP['opt']['momentum']
    
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = HYP['opt']['lr'] / kilostep_scale
    wd = HYP['opt']['weight_decay'] * batch_size / kilostep_scale
    
    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=HYP['opt']['label_smoothing'],
        reduction='none'
    )
    
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000, device=device)
    
    # Use loader with cutout augmentation
    train_loader = CifarLoaderWithCutout(
        'cifar10', train=True, batch_size=batch_size,
        aug={'flip': HYP['aug']['flip'], 'translate': HYP['aug']['translate']},
        cutout_size=HYP['aug']['cutout'],
        device=device
    )
    
    total_train_steps = ceil(len(train_loader) * epochs)
    
    # Create widest network with residual connections
    model = make_net_96(
        batchnorm_momentum=HYP['net']['batchnorm_momentum'],
        scaling_factor=HYP['net']['scaling_factor'],
        device=device
    )
    
    optimizer = get_optimizer(
        model, lr, momentum, wd,
        bias_scaler=HYP['opt']['bias_scaler']
    )
    
    lr_schedule = triangle_schedule(total_train_steps, start=0.2, end=0.07, peak=0.23)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: lr_schedule[i])
    
    lookahead_alpha = alpha_schedule(total_train_steps)
    lookahead_state = LookaheadState(model)
    
    timer = Timer(device)
    current_steps = 0
    
    if verbose:
        print_header()
    
    # Initialize whitening layer
    timer.start()
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model[0], train_images)
    timer.stop()
    
    for epoch in range(ceil(epochs)):
        model[0].bias.requires_grad = (epoch < HYP['opt']['whiten_bias_epochs'])
        
        timer.start()
        model.train()
        
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels).sum()
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            current_steps += 1
            
            if current_steps % 5 == 0:
                lookahead_state.update(model, decay=lookahead_alpha[current_steps].item())
            
            if current_steps >= total_train_steps:
                lookahead_state.update(model, decay=1.0)
                break
        
        timer.stop()
        
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        train_loss = loss.item() / batch_size
        val_acc = evaluate(model, test_loader, tta_level=0)
        
        if verbose:
            print_training_details({
                'run': run_id if epoch == 0 else None,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'time': timer.total_time
            }, is_final_entry=False)
        
        if current_steps >= total_train_steps:
            break
    
    timer.start()
    tta_val_acc = evaluate(model, test_loader, tta_level=HYP['net']['tta_level'])
    timer.stop()
    
    if verbose:
        print_training_details({
            'epoch': 'eval',
            'tta_val_acc': tta_val_acc,
            'time': timer.total_time
        }, is_final_entry=True)
    
    return {
        'accuracy': tta_val_acc,
        'time': timer.total_time
    }


def main():
    parser = argparse.ArgumentParser(description='airbench96: CIFAR-10 96% target')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs')
    args = parser.parse_args()
    
    device = get_device()
    print(f"Training airbench96 on {device}")
    print(f"Target: 96% accuracy in ~40 epochs")
    print()
    
    results = []
    for run in range(args.runs):
        result = train96(run_id=run + 1, device=device, verbose=True)
        results.append(result)
        print()
    
    accs = torch.tensor([r['accuracy'] for r in results])
    times = torch.tensor([r['time'] for r in results])
    
    print(f"Results over {args.runs} run(s):")
    print(f"  Accuracy: {accs.mean():.4f} +/- {accs.std():.4f}")
    print(f"  Time: {times.mean():.2f}s +/- {times.std():.2f}s")


if __name__ == '__main__':
    main()

