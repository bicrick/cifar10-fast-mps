#!/usr/bin/env python3
"""
Baseline Training: Standard ResNet-18 on CIFAR-10

This script implements a standard training procedure for comparison:
- ResNet-18 architecture (modified for CIFAR-10)
- Standard SGD with momentum
- Random horizontal flip augmentation
- No special initialization tricks
- No test-time augmentation

This serves as a baseline to demonstrate the speedups achieved by
the airbench methods.

Usage:
    python -m src.baseline [--epochs 50] [--runs 3]
"""

import argparse
from math import ceil

import torch
from torch import nn

from .data import StandardCifarLoader, CIFAR_MEAN, CIFAR_STD
from .models import make_resnet18
from .utils import Timer, print_header, print_training_details, get_device


# Hyperparameters for baseline training
BASELINE_HYP = {
    'epochs': 50,
    'batch_size': 128,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'lr_schedule': 'cosine',  # 'cosine' or 'step'
}


def train_baseline(
    epochs=None,
    batch_size=None,
    lr=None,
    run_id=None,
    device=None,
    verbose=True
):
    """
    Train ResNet-18 on CIFAR-10 using standard methods.
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Initial learning rate
        run_id: Run identifier for logging
        device: Device to train on
        verbose: Whether to print training progress
    
    Returns:
        dict with 'accuracy' and 'time' keys
    """
    # Set defaults
    epochs = epochs or BASELINE_HYP['epochs']
    batch_size = batch_size or BASELINE_HYP['batch_size']
    lr = lr or BASELINE_HYP['lr']
    device = device or get_device()
    
    # Data loaders with standard augmentation
    train_aug = {'flip': True, 'translate': 4}
    train_loader = StandardCifarLoader(
        'cifar10', train=True, batch_size=batch_size, aug=train_aug, device=device
    )
    test_loader = StandardCifarLoader(
        'cifar10', train=False, batch_size=500, device=device
    )
    
    # Model
    model = make_resnet18(num_classes=10, device=device)
    
    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=BASELINE_HYP['momentum'],
        weight_decay=BASELINE_HYP['weight_decay']
    )
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    
    timer = Timer(device)
    
    if verbose:
        print_header()
    
    timer.start()
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        val_acc = test_correct / test_total
        
        if verbose:
            print_training_details({
                'run': run_id if epoch == 0 else None,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'time': timer.total_time + (timer._start_time and 
                        (torch.mps.synchronize() if device == 'mps' else None) or 0)
            }, is_final_entry=(epoch == epochs - 1))
    
    timer.stop()
    
    return {
        'accuracy': val_acc,
        'time': timer.total_time
    }


def main():
    parser = argparse.ArgumentParser(description='Baseline ResNet-18 training on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    args = parser.parse_args()
    
    device = get_device()
    print(f"Training baseline on {device}")
    print(f"Epochs: {args.epochs}, Runs: {args.runs}")
    print()
    
    results = []
    for run in range(args.runs):
        result = train_baseline(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            run_id=run + 1,
            device=device,
            verbose=True
        )
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

