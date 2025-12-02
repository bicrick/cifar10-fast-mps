#!/usr/bin/env python3
"""
airbench95: 95% Accuracy on CIFAR-10

This script implements the medium training method from the paper:
"94% on CIFAR-10 in 3.29 Seconds on a Single GPU" (arXiv:2404.00498)

Modifications from airbench94:
- Increased training epochs: 9.9 -> 15
- Wider network:
  - Block 1: 64 -> 128 channels
  - Block 2-3: 256 -> 384 channels
- Reduced learning rate by factor of 0.87

Target: 95.01% accuracy in ~15 epochs (~10.4s on A100)

Usage:
    python -m src.airbench95 [--runs 10]
"""

import argparse
from math import ceil

import torch
from torch import nn

from .data import CifarLoader
from .models import make_net_95
from .optim import LookaheadState, triangle_schedule, alpha_schedule, get_optimizer
from .utils import (
    init_whitening_conv, evaluate, Timer,
    print_header, print_training_details, get_device
)


# Hyperparameters for 95% target
HYP = {
    'opt': {
        'train_epochs': 15,
        'batch_size': 1024,
        'lr': 11.5 * 0.87,  # reduced by factor of 0.87
        'momentum': 0.85,
        'weight_decay': 0.0153,
        'bias_scaler': 64.0,
        'label_smoothing': 0.2,
        'whiten_bias_epochs': 3,
    },
    'aug': {
        'flip': True,
        'translate': 2,
    },
    'net': {
        'batchnorm_momentum': 0.6,
        'scaling_factor': 1/9,
        'tta_level': 2,
    },
}


def train95(run_id=None, device=None, verbose=True):
    """
    Train the airbench95 model.
    
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
    train_loader = CifarLoader(
        'cifar10', train=True, batch_size=batch_size,
        aug=HYP['aug'], device=device
    )
    
    total_train_steps = ceil(len(train_loader) * epochs)
    
    # Create wider network for 95% target
    model = make_net_95(
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
    parser = argparse.ArgumentParser(description='airbench95: CIFAR-10 95% target')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs')
    args = parser.parse_args()
    
    device = get_device()
    print(f"Training airbench95 on {device}")
    print(f"Target: 95% accuracy in ~15 epochs")
    print()
    
    results = []
    for run in range(args.runs):
        result = train95(run_id=run + 1, device=device, verbose=True)
        results.append(result)
        print()
    
    accs = torch.tensor([r['accuracy'] for r in results])
    times = torch.tensor([r['time'] for r in results])
    
    print(f"Results over {args.runs} run(s):")
    print(f"  Accuracy: {accs.mean():.4f} +/- {accs.std():.4f}")
    print(f"  Time: {times.mean():.2f}s +/- {times.std():.2f}s")


if __name__ == '__main__':
    main()

