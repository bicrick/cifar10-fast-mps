"""
Optimization Components for Fast CIFAR-10 Training

This module implements:
- Lookahead optimizer wrapper
- Triangular learning rate schedule
- Decoupled weight decay formulation

The paper uses a decoupled hyperparameter formulation where:
- Learning rate, momentum, and weight decay can be tuned independently
- The average step size is decoupled from momentum
- Weight decay size is decoupled from learning rate
"""

import torch


class LookaheadState:
    """
    Lookahead optimizer state (Zhang et al., 2019).
    
    Lookahead maintains an exponential moving average of the model weights
    and periodically synchronizes the fast weights back to this slow average.
    
    In this implementation, we update every 5 steps with a decay schedule
    that increases over training (starts at ~0.95^5 and grows).
    
    Args:
        net: The neural network model
    """
    
    def __init__(self, net):
        # Store EMA of all parameters
        self.net_ema = {k: v.clone() for k, v in net.state_dict().items()}
    
    def update(self, net, decay):
        """
        Update the EMA and synchronize network weights.
        
        Args:
            net: Current network
            decay: EMA decay factor (higher = slower update)
        """
        for ema_param, net_param in zip(
            self.net_ema.values(), 
            net.state_dict().values()
        ):
            if net_param.dtype in (torch.half, torch.float):
                # Update EMA: ema = decay * ema + (1 - decay) * current
                ema_param.lerp_(net_param, 1 - decay)
                # Synchronize network to EMA
                net_param.copy_(ema_param)


def triangle_schedule(total_steps, start=0.2, end=0.07, peak=0.23):
    """
    Create a triangular learning rate schedule.
    
    The learning rate:
    - Starts at `start` fraction of max
    - Linearly increases to max at `peak` fraction of training
    - Linearly decreases to `end` fraction of max
    
    Args:
        total_steps: Total number of training steps
        start: Initial LR as fraction of max (default 0.2)
        end: Final LR as fraction of max (default 0.07)
        peak: Fraction of training where LR peaks (default 0.23)
    
    Returns:
        Tensor of LR multipliers for each step
    """
    xp = torch.tensor([0, int(peak * total_steps), total_steps])
    fp = torch.tensor([start, 1.0, end])
    x = torch.arange(1 + total_steps)
    
    # Compute slopes and intercepts for each segment
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])
    
    # Find which segment each step belongs to
    indices = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indices = torch.clamp(indices, 0, len(m) - 1)
    
    return m[indices] * x + b[indices]


def alpha_schedule(total_steps):
    """
    Create the Lookahead alpha (decay) schedule.
    
    The decay factor increases over training as:
    alpha = 0.95^5 * (step / total_steps)^3
    
    This means early in training, synchronization is more aggressive
    (lower decay = faster update), and later it becomes more conservative.
    
    Args:
        total_steps: Total number of training steps
    
    Returns:
        Tensor of alpha values for each step
    """
    return 0.95 ** 5 * (torch.arange(total_steps + 1) / total_steps) ** 3


def get_optimizer(model, lr, momentum, weight_decay, bias_scaler=64.0):
    """
    Create SGD optimizer with scaled learning rates for BatchNorm biases.
    
    The paper uses 64x higher learning rate for BatchNorm biases,
    which speeds up training significantly.
    
    Args:
        model: Neural network model
        lr: Base learning rate
        momentum: SGD momentum
        weight_decay: Weight decay (already scaled by batch size)
        bias_scaler: Multiplier for BatchNorm bias learning rate
    
    Returns:
        torch.optim.SGD optimizer
    """
    # Separate parameters into norm biases and others
    norm_biases = [
        p for k, p in model.named_parameters() 
        if 'norm' in k and p.requires_grad
    ]
    other_params = [
        p for k, p in model.named_parameters() 
        if 'norm' not in k and p.requires_grad
    ]
    
    lr_biases = lr * bias_scaler
    
    param_configs = [
        dict(params=norm_biases, lr=lr_biases, weight_decay=weight_decay / lr_biases),
        dict(params=other_params, lr=lr, weight_decay=weight_decay / lr),
    ]
    
    return torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

