"""
CIFAR-10 Data Loading with Alternating Flip Augmentation

This module implements the GPU-accelerated CIFAR-10 loader from the paper,
including the novel "alternating flip" augmentation strategy.

The key insight is that standard random horizontal flipping causes some images
to be redundantly flipped the same way for many epochs in a row. Alternating
flip deterministically alternates flips after the first epoch, ensuring every
pair of consecutive epochs contains all 2N unique inputs (original + flipped).
"""

import os
from math import ceil

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


# CIFAR-10 normalization constants
CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))


def batch_flip_lr(inputs):
    """Apply random horizontal flipping to a batch of images."""
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    """
    Apply random cropping to a batch of images.
    
    This is equivalent to padding with `r` pixels and taking a random crop,
    but implemented efficiently for batched GPU operation.
    """
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    images_out = torch.empty(
        (len(images), 3, crop_size, crop_size),
        device=images.device,
        dtype=images.dtype
    )
    
    # Two cropping methods - the second is faster for r > 2
    if r <= 2:
        for sy in range(-r, r + 1):
            for sx in range(-r, r + 1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[
                    mask, :, r + sy:r + sy + crop_size, r + sx:r + sx + crop_size
                ]
    else:
        images_tmp = torch.empty(
            (len(images), 3, crop_size, crop_size + 2 * r),
            device=images.device,
            dtype=images.dtype
        )
        for s in range(-r, r + 1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r + s:r + s + crop_size, :]
        for s in range(-r, r + 1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r + s:r + s + crop_size]
    
    return images_out


class CifarLoader:
    """
    GPU-accelerated CIFAR-10 dataloader with alternating flip augmentation.
    
    Key features:
    - Loads data directly to GPU/MPS for faster processing
    - Implements alternating flip: on odd epochs, flip images that weren't
      flipped in epoch 0; on even epochs, flip those that were
    - Efficient batch cropping with reflection padding
    
    Args:
        path: Directory to store/load CIFAR-10 data
        train: Whether to load training set (True) or test set (False)
        batch_size: Batch size for iteration
        aug: Dict with augmentation options {'flip': bool, 'translate': int}
        drop_last: Whether to drop incomplete final batch
        shuffle: Whether to shuffle data each epoch
        device: Device to load data to ('cuda', 'mps', or 'cpu')
    """
    
    def __init__(
        self,
        path,
        train=True,
        batch_size=500,
        aug=None,
        drop_last=None,
        shuffle=None,
        device=None
    ):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        
        # Download and preprocess CIFAR-10 if needed
        if not os.path.exists(data_path):
            os.makedirs(path, exist_ok=True)
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save(
                {'images': images, 'labels': labels, 'classes': dset.classes},
                data_path
            )
        
        data = torch.load(data_path, map_location=torch.device(device), weights_only=True)
        self.images = data['images']
        self.labels = data['labels']
        self.classes = data['classes']
        
        # Convert to float and normalize to [0, 1]
        # Using float32 for MPS compatibility (MPS has limited float16 support)
        self.images = (self.images.float() / 255).permute(0, 3, 1, 2)
        if device != 'mps':
            self.images = self.images.to(memory_format=torch.channels_last)
        
        self.normalize = T.Normalize(CIFAR_MEAN.to(device), CIFAR_STD.to(device))
        self.proc_images = {}  # Cached processed images
        self.epoch = 0
        self.device = device
        
        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate'], f'Unrecognized augmentation key: {k}'
        
        self.batch_size = batch_size
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle
    
    def __len__(self):
        if self.drop_last:
            return len(self.images) // self.batch_size
        return ceil(len(self.images) / self.batch_size)
    
    def __iter__(self):
        """
        Iterate over batches with alternating flip augmentation.
        
        On epoch 0: Random 50% flip (standard augmentation)
        On odd epochs: Flip images that were NOT flipped in epoch 0
        On even epochs (>0): Flip images that WERE flipped in epoch 0
        
        This ensures consecutive epoch pairs see all 2N unique inputs.
        """
        if self.epoch == 0:
            # First epoch: normalize and apply random flip
            images = self.proc_images['norm'] = self.normalize(self.images)
            
            if self.aug.get('flip', False):
                images = self.proc_images['flip'] = batch_flip_lr(images)
            
            # Pre-pad images for translation augmentation
            pad = self.aug.get('translate', 0)
            if pad > 0:
                self.proc_images['pad'] = F.pad(images, (pad,) * 4, 'reflect')
        
        # Select appropriate processed images
        if self.aug.get('translate', 0) > 0:
            images = batch_crop(self.proc_images['pad'], self.images.shape[-2])
        elif self.aug.get('flip', False):
            images = self.proc_images['flip']
        else:
            images = self.proc_images['norm']
        
        # Alternating flip: on odd epochs, flip all images
        # Combined with initial random flip, this gives deterministic alternation
        if self.aug.get('flip', False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)
        
        self.epoch += 1
        
        # Generate batch indices
        indices = (torch.randperm if self.shuffle else torch.arange)(
            len(images), device=images.device
        )
        
        for i in range(len(self)):
            idxs = indices[i * self.batch_size:(i + 1) * self.batch_size]
            yield images[idxs], self.labels[idxs]


class StandardCifarLoader:
    """
    Standard CIFAR-10 dataloader with random flip augmentation.
    
    This is used for the baseline to demonstrate the improvement
    from alternating flip augmentation.
    """
    
    def __init__(
        self,
        path,
        train=True,
        batch_size=128,
        aug=None,
        device=None
    ):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        self.batch_size = batch_size
        self.train = train
        self.aug = aug or {}
        
        # Standard torchvision transforms
        transforms = [T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)]
        
        if train and self.aug.get('flip', False):
            transforms.insert(0, T.RandomHorizontalFlip())
        
        if train and self.aug.get('translate', 0) > 0:
            pad = self.aug['translate']
            transforms.insert(0, T.RandomCrop(32, padding=pad, padding_mode='reflect'))
        
        self.transform = T.Compose(transforms)
        
        # Download dataset
        os.makedirs(path, exist_ok=True)
        self.dataset = torchvision.datasets.CIFAR10(
            path, download=True, train=train, transform=self.transform
        )
        
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=0,  # MPS works better with num_workers=0
            pin_memory=False
        )
    
    def __len__(self):
        return len(self.loader)
    
    def __iter__(self):
        for images, labels in self.loader:
            yield images.to(self.device), labels.to(self.device)

