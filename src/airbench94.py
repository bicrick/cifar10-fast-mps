#!/usr/bin/env python3
"""
airbench94: 94% Accuracy on CIFAR-10

This script implements the fastest training method from the paper:
"94% on CIFAR-10 in 3.29 Seconds on a Single GPU" (arXiv:2404.00498)

Key techniques:
- VGG-like network with 1.97M parameters
- Frozen patch-whitening initialization for first layer
- Dirac (identity) initialization for later convolutions
- 64x scaled learning rate for BatchNorm biases
- Lookahead optimizer
- Alternating flip augmentation
- Multi-crop test-time augmentation
- Triangular learning rate schedule

Target: 94.01% accuracy in ~10 epochs (~3.8s on A100, longer on M3)

Usage:
    python -m src.airbench94 [--runs 25]
"""

import argparse
from math import ceil

import torch
from torch import nn

from .data import CifarLoader
from .models import make_net
from .optim import LookaheadState, triangle_schedule, alpha_schedule, get_optimizer
from .utils import (
    init_whitening_conv, evaluate, Timer,
    print_header, print_training_details, get_device
)


# Hyperparameters from the paper
HYP = {
    'opt': {
        'train_epochs': 9.9,
        'batch_size': 1024,
        'lr': 11.5,  # learning rate per 1024 examples
        'momentum': 0.85,
        'weight_decay': 0.0153,  # weight decay per 1024 examples
        'bias_scaler': 64.0,  # scales up LR for BatchNorm biases
        'label_smoothing': 0.2,
        'whiten_bias_epochs': 3,  # epochs to train whitening bias before freezing
    },
    'aug': {
        'flip': True,
        'translate': 2,
    },
    'net': {
        'widths': {
            'block1': 64,
            'block2': 256,
            'block3': 256,
        },
        'batchnorm_momentum': 0.6,
        'scaling_factor': 1/9,
        'tta_level': 2,  # 0=none, 1=mirror, 2=mirror+translate
    },
}


def train94(run_id=None, device=None, verbose=True):
    """
    Train the airbench94 model.
    
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
    
    # Decoupled learning rate formulation
    # The kilostep_scale accounts for momentum's effect on step size
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = HYP['opt']['lr'] / kilostep_scale
    wd = HYP['opt']['weight_decay'] * batch_size / kilostep_scale
    
    # Loss function with label smoothing
    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=HYP['opt']['label_smoothing'],
        reduction='none'
    )
    
    # Data loaders
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000, device=device)
    train_loader = CifarLoader(
        'cifar10', train=True, batch_size=batch_size,
        aug=HYP['aug'], device=device
    )
    
    total_train_steps = ceil(len(train_loader) * epochs)
    
    # Create model
    model = make_net(
        widths=HYP['net']['widths'],
        batchnorm_momentum=HYP['net']['batchnorm_momentum'],
        scaling_factor=HYP['net']['scaling_factor'],
        device=device
    )
    
    # Optimizer with scaled BatchNorm bias learning rate
    optimizer = get_optimizer(
        model, lr, momentum, wd,
        bias_scaler=HYP['opt']['bias_scaler']
    )
    
    # Learning rate schedule
    lr_schedule = triangle_schedule(total_train_steps, start=0.2, end=0.07, peak=0.23)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: lr_schedule[i])
    
    # Lookahead
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
        # Freeze whitening bias after initial epochs
        model[0].bias.requires_grad = (epoch < HYP['opt']['whiten_bias_epochs'])
        
        # Training
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
            
            # Lookahead update every 5 steps
            if current_steps % 5 == 0:
                lookahead_state.update(model, decay=lookahead_alpha[current_steps].item())
            
            if current_steps >= total_train_steps:
                # Final lookahead sync
                lookahead_state.update(model, decay=1.0)
                break
        
        timer.stop()
        
        # Evaluation
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
    
    # Final evaluation with TTA
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


def warmup(device=None):
    """
    Warmup run with dummy data to initialize GPU.
    
    This ensures accurate timing for subsequent runs.
    """
    device = device or get_device()
    
    # Create a quick training run with dummy labels
    train_loader = CifarLoader(
        'cifar10', train=True, batch_size=HYP['opt']['batch_size'],
        aug=HYP['aug'], device=device
    )
    train_loader.labels = torch.randint(
        0, 10, size=(len(train_loader.labels),),
        device=train_loader.labels.device
    )
    
    # Just run a few steps
    model = make_net(device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        if i >= 5:
            break
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    parser = argparse.ArgumentParser(description='airbench94: Fast CIFAR-10 training')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs')
    parser.add_argument('--no-warmup', action='store_true', help='Skip warmup')
    args = parser.parse_args()
    
    device = get_device()
    print(f"Training airbench94 on {device}")
    print(f"Target: 94% accuracy in ~10 epochs")
    print()
    
    # Warmup
    if not args.no_warmup:
        print("Warming up...")
        warmup(device)
        print()
    
    results = []
    for run in range(args.runs):
        result = train94(run_id=run + 1, device=device, verbose=True)
        results.append(result)
        print()
    
    # Summary
    accs = torch.tensor([r['accuracy'] for r in results])
    times = torch.tensor([r['time'] for r in results])
    
    print(f"Results over {args.runs} run(s):")
    print(f"  Accuracy: {accs.mean():.4f} +/- {accs.std():.4f}")
    print(f"  Time: {times.mean():.2f}s +/- {times.std():.2f}s")


if __name__ == '__main__':
    main()

