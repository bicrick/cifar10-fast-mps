"""
CIFAR-10 Fast Training - Re-implementation of arXiv:2404.00498

This package implements the training methods from:
"94% on CIFAR-10 in 3.29 Seconds on a Single GPU" by Keller Jordan
"""

from .data import CifarLoader, CIFAR_MEAN, CIFAR_STD
from .models import make_net, make_net_95, make_net_96
from .optim import LookaheadState, triangle_schedule
from .utils import (
    init_whitening_conv,
    evaluate,
    infer,
    print_training_details,
    get_device,
)

__version__ = "0.1.0"

