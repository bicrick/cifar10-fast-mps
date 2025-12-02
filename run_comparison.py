#!/usr/bin/env python3
"""
Comparison Script: Benchmark All Training Methods

This script runs all training methods and compares their performance:
1. Baseline: Standard ResNet-18 training
2. airbench94: 94% target (~10 epochs)
3. airbench95: 95% target (~15 epochs)
4. airbench96: 96% target (~40 epochs)

Usage:
    python run_comparison.py [--quick]
    
    --quick: Run fewer epochs for quick testing
"""

import argparse
import time

import torch
from tabulate import tabulate

from src.utils import get_device
from src.baseline import train_baseline
from src.airbench94 import train94, warmup
from src.airbench95 import train95
from src.airbench96 import train96


def run_comparison(quick_mode=False):
    """Run all training methods and compare results."""
    
    device = get_device()
    print("=" * 60)
    print("CIFAR-10 Fast Training Comparison")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Mode: {'Quick (reduced epochs)' if quick_mode else 'Full'}")
    print()
    
    results = []
    
    # Warmup GPU
    print("Warming up GPU...")
    warmup(device)
    print()
    
    # 1. Baseline (ResNet-18)
    print("-" * 60)
    print("1. BASELINE: ResNet-18 with standard training")
    print("-" * 60)
    baseline_epochs = 10 if quick_mode else 50
    print(f"   Epochs: {baseline_epochs}")
    print()
    
    result = train_baseline(epochs=baseline_epochs, run_id=1, device=device, verbose=True)
    results.append({
        'method': 'Baseline (ResNet-18)',
        'accuracy': result['accuracy'] * 100,
        'time': result['time'],
        'epochs': baseline_epochs,
    })
    print()
    
    # 2. airbench94
    print("-" * 60)
    print("2. AIRBENCH94: Fast training for 94% target")
    print("-" * 60)
    print("   Epochs: ~10 (9.9)")
    print()
    
    result = train94(run_id=1, device=device, verbose=True)
    results.append({
        'method': 'airbench94',
        'accuracy': result['accuracy'] * 100,
        'time': result['time'],
        'epochs': 10,
    })
    print()
    
    # 3. airbench95
    print("-" * 60)
    print("3. AIRBENCH95: Medium training for 95% target")
    print("-" * 60)
    print("   Epochs: 15")
    print()
    
    result = train95(run_id=1, device=device, verbose=True)
    results.append({
        'method': 'airbench95',
        'accuracy': result['accuracy'] * 100,
        'time': result['time'],
        'epochs': 15,
    })
    print()
    
    # 4. airbench96 (skip in quick mode due to long training)
    if not quick_mode:
        print("-" * 60)
        print("4. AIRBENCH96: Full training for 96% target")
        print("-" * 60)
        print("   Epochs: 40")
        print()
        
        result = train96(run_id=1, device=device, verbose=True)
        results.append({
            'method': 'airbench96',
            'accuracy': result['accuracy'] * 100,
            'time': result['time'],
            'epochs': 40,
        })
        print()
    else:
        print("-" * 60)
        print("4. AIRBENCH96: Skipped in quick mode (40 epochs)")
        print("-" * 60)
        print()
    
    # Summary table
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    
    table_data = []
    for r in results:
        table_data.append([
            r['method'],
            r['epochs'],
            f"{r['accuracy']:.2f}%",
            f"{r['time']:.2f}s",
            f"{r['time'] / r['epochs']:.2f}s"
        ])
    
    headers = ['Method', 'Epochs', 'Accuracy', 'Total Time', 'Time/Epoch']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print()
    
    # Speedup analysis
    if len(results) >= 2:
        baseline = results[0]
        print("Speedup Analysis (vs Baseline):")
        print("-" * 40)
        for r in results[1:]:
            # Speedup in terms of time to reach similar accuracy
            time_ratio = baseline['time'] / r['time']
            print(f"  {r['method']}:")
            print(f"    Time: {r['time']:.2f}s vs {baseline['time']:.2f}s ({time_ratio:.1f}x faster)")
            print(f"    Accuracy: {r['accuracy']:.2f}% vs {baseline['accuracy']:.2f}%")
            print()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare CIFAR-10 training methods')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick mode: fewer epochs for testing')
    args = parser.parse_args()
    
    run_comparison(quick_mode=args.quick)


if __name__ == '__main__':
    main()

