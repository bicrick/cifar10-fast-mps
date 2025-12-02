# CIFAR-10 Fast Training

A re-implementation of the paper ["94% on CIFAR-10 in 3.29 Seconds on a Single GPU"](https://arxiv.org/abs/2404.00498) by Keller Jordan.

## Overview

This repository implements the fast training methods described in the paper, which achieve remarkable training speeds on CIFAR-10:

| Method | Target Accuracy | Time (A100) | Epochs |
|--------|----------------|-------------|--------|
| airbench94 | 94.01% | 3.83s | ~10 |
| airbench95 | 95.01% | 10.4s | 15 |
| airbench96 | 96.05% | 46.3s | 40 |

**Note:** This implementation is adapted for Apple Silicon (M3) using MPS backend. Timings will differ from the paper's A100 benchmarks, but relative speedups should be observable.

## MPS Implementation Notes

This implementation uses Apple's Metal Performance Shaders (MPS) backend instead of CUDA. Key differences and tradeoffs:

### What Works Well
- **Unified Memory:** No CPU-GPU transfer overhead; data lives in shared memory
- **Development Speed:** Train locally without cloud GPU costs
- **Core Algorithms:** All the paper's key techniques (alternating flip, patch whitening, Lookahead) work identically

### Limitations
- **No `torch.compile`:** The paper's compiled variant (`airbench94_compiled.py`) achieves a 14% speedup via kernel fusion. MPS doesn't support `torch.compile`, so we use eager execution only
- **Slower Absolute Times:** An M3 is roughly 5-10x slower than an A100 for this workload. Expect ~30-40s for airbench94 instead of 3.8s
- **Some Ops Differ:** Certain operations (like `channels_last` memory format) may not be fully optimized on MPS
- **Timing:** CUDA uses `torch.cuda.Event` for precise GPU timing; we fall back to `time.perf_counter()` on MPS

### Relative Speedups Still Hold
Despite the absolute performance gap, the relative improvements from each technique remain valid. If alternating flip provides a 15% speedup on A100, you'll see a similar relative gain on MPS.

### Benchmark Results (M3 MPS)

| Method | Epochs | Accuracy | Total Time | Time/Epoch |
|--------|--------|----------|------------|------------|
| Baseline (ResNet-18) | 10 | 89.56% | 339.08s | 33.91s |
| airbench94 | 10 | 93.82% | 67.04s | 6.70s |
| airbench95 | 15 | 95.12% | 199.51s | 13.30s |

**Speedup vs Baseline:**
- **airbench94:** 5.1x faster (67s vs 339s) with +4.3% accuracy
- **airbench95:** 1.7x faster (200s vs 339s) with +5.6% accuracy

## Key Techniques

The paper introduces several optimizations that together achieve significant speedups:

### 1. Alternating Flip Augmentation (Novel Contribution)
Instead of randomly flipping each image with 50% probability per epoch, we deterministically alternate flips after the first epoch. This ensures every pair of consecutive epochs sees all 2N unique inputs (original + flipped), reducing redundancy.

### 2. Patch-Whitening Initialization
The first convolutional layer is initialized as a whitening transformation of 2x2 patches from the training set. This frozen layer normalizes inputs and speeds up convergence.

### 3. Dirac (Identity) Initialization
Later convolutions are partially initialized as identity transforms, easing gradient flow similar to residual connections but without the architectural overhead.

### 4. Scaled BatchNorm Biases
BatchNorm bias learning rates are scaled up by 64x, accelerating training significantly.

### 5. Lookahead Optimizer
An exponential moving average of weights is maintained and periodically synchronized with the fast weights, improving stability and final accuracy.

### 6. Multi-Crop Test-Time Augmentation
Evaluation uses 6 augmented views per image (original + flipped, with translations), improving accuracy by ~0.7%.

## Installation

### Using Conda (Recommended)

```bash
# Create and activate environment
conda create -n cifar-fast python=3.11 -y
conda activate cifar-fast

# Install dependencies
pip install -r requirements.txt
```

### Using pip

```bash
pip install -r requirements.txt
```

## Usage

### Run Individual Methods

```bash
# Fastest: 94% target
python -m src.airbench94 --runs 3

# Medium: 95% target  
python -m src.airbench95 --runs 3

# Highest accuracy: 96% target
python -m src.airbench96 --runs 3

# Baseline comparison (standard ResNet-18)
python -m src.baseline --epochs 50 --runs 1
```

### Run Full Comparison

```bash
# Full comparison (takes several minutes)
python run_comparison.py

# Quick test mode (reduced epochs)
python run_comparison.py --quick
```

## Project Structure

```
cifar-paper/
├── README.md
├── requirements.txt
├── environment.yml
├── run_comparison.py      # Benchmark all methods
└── src/
    ├── __init__.py
    ├── data.py            # CIFAR-10 loader with alternating flip
    ├── models.py          # Network architectures
    ├── optim.py           # Lookahead, LR schedules
    ├── utils.py           # Whitening init, evaluation, timing
    ├── baseline.py        # Standard ResNet-18 training
    ├── airbench94.py      # 94% target method
    ├── airbench95.py      # 95% target method
    └── airbench96.py      # 96% target method
```

## Architecture Details

The network is VGG-like with the following key features:

- **First layer:** 2x2 convolution with 24 channels (patch-whitening initialized)
- **Main body:** Three blocks, each with two 3x3 convolutions, max pooling, BatchNorm, and GELU
- **Classifier:** Global max pooling + linear layer with output scaling (1/9)

Channel configurations:
- airbench94: 64 -> 256 -> 256 (1.97M params)
- airbench95: 128 -> 384 -> 384 (wider)
- airbench96: 128 -> 512 -> 512 (3 convs per block + residual)

## References

- Paper: [arXiv:2404.00498](https://arxiv.org/abs/2404.00498)
- Official Code: [github.com/KellerJordan/cifar10-airbench](https://github.com/KellerJordan/cifar10-airbench)

## License

This is an educational re-implementation for learning purposes.

