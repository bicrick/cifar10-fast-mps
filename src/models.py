"""
Network Architectures for Fast CIFAR-10 Training

This module implements the VGG-like network architectures from the paper:
- airbench94: 1.97M parameters, reaches 94% in ~10 epochs
- airbench95: Wider network for 95% target
- airbench96: Even wider with residual connections for 96% target

Key design choices:
- 2x2 first convolution creates 31x31 feature maps (slightly more favorable tradeoff)
- GELU activations
- BatchNorm without learnable scale (affine=True but weight.requires_grad=False)
- No biases in conv/linear layers except first conv
- Output scaled down by 1/9
"""

import torch
from torch import nn


class Flatten(nn.Module):
    """Flatten spatial dimensions for the classifier."""
    def forward(self, x):
        return x.view(x.size(0), -1)


class Mul(nn.Module):
    """Scale output by a constant factor."""
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        return x * self.scale


class BatchNorm(nn.BatchNorm2d):
    """
    BatchNorm with configurable learnable parameters.
    
    By default, scale (weight) is not learned, only bias is learned.
    This follows the paper's approach of disabling affine scale parameters.
    """
    def __init__(self, num_features, momentum=0.6, eps=1e-12, weight=False, bias=True):
        # PyTorch uses (1 - momentum) for its momentum parameter
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class Conv(nn.Conv2d):
    """
    Convolution with Dirac (identity) initialization.
    
    For layers where out_channels >= in_channels, the first in_channels
    filters are initialized as identity transforms. This helps gradient
    flow and speeds up training.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            bias=bias
        )
    
    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        # Initialize first in_channels filters as identity (Dirac)
        torch.nn.init.dirac_(w[:w.size(1)])


class ConvGroup(nn.Module):
    """
    A block of two convolutions with pooling, normalization, and activation.
    
    Structure: Conv -> MaxPool -> BN -> GELU -> Conv -> BN -> GELU
    """
    def __init__(self, channels_in, channels_out, batchnorm_momentum=0.6):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, batchnorm_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, batchnorm_momentum)
        self.activ = nn.GELU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


class ConvGroupResidual(nn.Module):
    """
    A block of three convolutions with residual connection (for airbench96).
    
    Structure: Conv -> Pool -> BN -> GELU -> [Conv -> BN -> GELU -> Conv -> BN + residual] -> GELU
    
    The residual connection spans the last two convolutions.
    """
    def __init__(self, channels_in, channels_out, batchnorm_momentum=0.6):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, batchnorm_momentum)
        
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, batchnorm_momentum)
        
        self.conv3 = Conv(channels_out, channels_out)
        self.norm3 = BatchNorm(channels_out, batchnorm_momentum)
        
        self.activ = nn.GELU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        
        # Residual block
        residual = x
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = x + residual
        x = self.activ(x)
        
        return x


def make_net(
    widths=None,
    batchnorm_momentum=0.6,
    scaling_factor=1/9,
    device=None
):
    """
    Create the airbench94 network architecture.
    
    Args:
        widths: Dict with channel counts for each block
        batchnorm_momentum: Momentum for batch normalization
        scaling_factor: Factor to scale final output (1/9 in paper)
        device: Device to create network on
    
    Returns:
        nn.Sequential model with 1.97M parameters
    """
    if widths is None:
        widths = {
            'block1': 64,
            'block2': 256,
            'block3': 256,
        }
    
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    # First layer: 2x2 conv with 24 channels (will be patch-whitening initialized)
    # 24 = 2 * 3 * 2^2 (doubled for positive/negative eigenvectors)
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size ** 2  # 24
    
    net = nn.Sequential(
        # Whitening layer: 2x2 conv, no padding, with bias
        nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        # Main body: three conv groups
        ConvGroup(whiten_width, widths['block1'], batchnorm_momentum),
        ConvGroup(widths['block1'], widths['block2'], batchnorm_momentum),
        ConvGroup(widths['block2'], widths['block3'], batchnorm_momentum),
        # Classifier
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        Mul(scaling_factor),
    )
    
    # Freeze whitening layer weights (will be initialized separately)
    net[0].weight.requires_grad = False
    
    # Move to device
    net = net.float().to(device)
    
    # Use channels_last memory format for better performance (not on MPS)
    if device != 'mps':
        net = net.to(memory_format=torch.channels_last)
    
    # Keep BatchNorm in float32 for stability
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    
    return net


def make_net_95(
    batchnorm_momentum=0.6,
    scaling_factor=1/9,
    device=None
):
    """
    Create the airbench95 network architecture.
    
    Wider than airbench94:
    - Block 1: 64 -> 128 channels
    - Block 2-3: 256 -> 384 channels
    """
    widths = {
        'block1': 128,
        'block2': 384,
        'block3': 384,
    }
    return make_net(widths, batchnorm_momentum, scaling_factor, device)


def make_net_96(
    batchnorm_momentum=0.6,
    scaling_factor=1/9,
    device=None
):
    """
    Create the airbench96 network architecture.
    
    Even wider with residual connections:
    - Block 1: 128 channels
    - Block 2-3: 512 channels
    - Three convolutions per block with residual
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size ** 2  # 24
    
    net = nn.Sequential(
        nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroupResidual(whiten_width, 128, batchnorm_momentum),
        ConvGroupResidual(128, 512, batchnorm_momentum),
        ConvGroupResidual(512, 512, batchnorm_momentum),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(512, 10, bias=False),
        Mul(scaling_factor),
    )
    
    net[0].weight.requires_grad = False
    net = net.float().to(device)
    
    if device != 'mps':
        net = net.to(memory_format=torch.channels_last)
    
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    
    return net


# Standard ResNet-18 for baseline comparison
def make_resnet18(num_classes=10, device=None):
    """
    Create a standard ResNet-18 for baseline comparison.
    
    This uses torchvision's implementation, modified for CIFAR-10
    (smaller input size than ImageNet).
    """
    import torchvision.models as models
    
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    # Create ResNet-18
    net = models.resnet18(weights=None, num_classes=num_classes)
    
    # Modify first conv for CIFAR-10 (32x32 instead of 224x224)
    # Use 3x3 kernel, stride 1, padding 1 instead of 7x7 kernel, stride 2, padding 3
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove the max pooling layer (not needed for small images)
    net.maxpool = nn.Identity()
    
    net = net.to(device)
    
    return net

