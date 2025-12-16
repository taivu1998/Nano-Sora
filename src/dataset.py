import os
import torch
import numpy as np
import requests
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Optional, Tuple

# Expected file size for verification (~819 MB)
EXPECTED_FILE_SIZE = 819200080
EXPECTED_SHAPE = (20, 10000, 64, 64)


class MovingMNIST(Dataset):
    """
    Moving MNIST Dataset.
    Source: http://www.cs.toronto.edu/~nitish/unsupervised_video/

    Each video contains two digits moving independently with constant velocity,
    bouncing off the walls. Videos are 20 frames of 64x64 grayscale images.

    Attributes:
        data: Tensor of shape (N, 1, T, H, W) where T is num_frames
    """
    URL = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
    BACKUP_URL = "https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy"

    def __init__(
        self,
        root: str,
        num_frames: int = 16,
        train: bool = True,
        split: float = 0.9,
        transform: Optional[callable] = None,
        random_temporal_crop: bool = False,
        horizontal_flip: bool = False,
    ):
        """
        Args:
            root: Directory to store/load the dataset
            num_frames: Number of frames to use (cropped from 20)
            train: If True, return training split; else return test split
            split: Fraction of data to use for training
            transform: Optional transform to apply to each sample
            random_temporal_crop: If True, randomly crop temporal window instead of from start
            horizontal_flip: If True, randomly flip videos horizontally (for training)
        """
        self.root = root
        self.num_frames = num_frames
        self.train = train
        self.split = split
        self.transform = transform
        self.random_temporal_crop = random_temporal_crop and train
        self.horizontal_flip = horizontal_flip and train

        self.path = os.path.join(root, "mnist_test_seq.npy")

        # Check if file exists and is valid
        if not os.path.exists(self.path) or not self._verify_file():
            self._download()

        # Load: (20, 10000, 64, 64) -> (Sequence, Batch, H, W)
        try:
            data = np.load(self.path, allow_pickle=True)
        except Exception as e:
            print(f"Failed to load dataset, attempting re-download...")
            os.remove(self.path)
            self._download()
            data = np.load(self.path, allow_pickle=True)

        # Validate shape
        if data.shape != EXPECTED_SHAPE:
            print(f"Unexpected data shape: {data.shape}, expected {EXPECTED_SHAPE}")
            print("Attempting re-download...")
            os.remove(self.path)
            self._download()
            data = np.load(self.path, allow_pickle=True)
            if data.shape != EXPECTED_SHAPE:
                raise ValueError(f"Dataset still invalid after re-download. Shape: {data.shape}")

        # Transpose to (Batch, Sequence, H, W) -> (10000, 20, 64, 64)
        data = data.transpose(1, 0, 2, 3)

        # For fixed temporal crop (when not using random)
        if not self.random_temporal_crop:
            data = data[:, :num_frames, :, :]

        # Normalize to [-1, 1]
        data = (data.astype(np.float32) / 127.5) - 1.0

        # Add Channel Dimension: (B, C, T, H, W)
        data = np.expand_dims(data, axis=1)

        # Split into train/test
        n_total = len(data)
        n_train = int(n_total * split)

        if train:
            self.data = data[:n_train]
        else:
            self.data = data[n_train:]

        # Store original num_frames for random cropping
        self.original_frames = data.shape[2]

    def _verify_file(self) -> bool:
        """Verify the downloaded file is valid."""
        if not os.path.exists(self.path):
            return False

        file_size = os.path.getsize(self.path)
        if file_size < EXPECTED_FILE_SIZE - 1000:  # Allow small tolerance
            print(f"File incomplete: {file_size:,} bytes, expected ~{EXPECTED_FILE_SIZE:,}")
            return False

        # Try to load and check shape
        try:
            data = np.load(self.path, allow_pickle=True)
            if data.shape != EXPECTED_SHAPE:
                print(f"Invalid shape: {data.shape}, expected {EXPECTED_SHAPE}")
                return False
            return True
        except Exception as e:
            print(f"File corrupted: {e}")
            return False

    def _download(self):
        """Download the Moving MNIST dataset."""
        print(f"Downloading Moving MNIST to {self.path}...")
        os.makedirs(self.root, exist_ok=True)

        # Remove existing incomplete file
        if os.path.exists(self.path):
            os.remove(self.path)

        # Try primary URL first, then backup
        for url in [self.URL, self.BACKUP_URL]:
            try:
                print(f"Trying: {url}")
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                print(f"Expected size: {total_size:,} bytes")

                downloaded = 0
                with open(self.path, 'wb') as f, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc="Downloading"
                ) as t:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            t.update(len(chunk))

                # Verify download size
                actual_size = os.path.getsize(self.path)
                if actual_size < total_size * 0.99:
                    print(f"Download incomplete: {actual_size:,}/{total_size:,} bytes")
                    os.remove(self.path)
                    continue

                print(f"Download complete: {self.path} ({actual_size:,} bytes)")
                return

            except requests.RequestException as e:
                print(f"Failed to download from {url}: {e}")
                if os.path.exists(self.path):
                    os.remove(self.path)
                continue

        raise RuntimeError(
            f"Failed to download Moving MNIST from all sources. "
            f"Please manually download from {self.URL} and save to {self.path}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            Video tensor of shape (1, T, H, W) normalized to [-1, 1]
        """
        video = self.data[idx].copy()  # (1, T, H, W)

        # Random temporal crop if enabled
        if self.random_temporal_crop and self.original_frames > self.num_frames:
            max_start = self.original_frames - self.num_frames
            start = np.random.randint(0, max_start + 1)
            video = video[:, start:start + self.num_frames, :, :]

        # Random horizontal flip
        if self.horizontal_flip and np.random.random() > 0.5:
            video = video[:, :, :, ::-1].copy()

        # Apply custom transform if provided
        if self.transform is not None:
            video = self.transform(video)

        return torch.from_numpy(video)

    def get_sample_shape(self) -> Tuple[int, int, int, int]:
        """Return the shape of a single sample (C, T, H, W)."""
        return (1, self.num_frames, 64, 64)

def create_dataloaders(
    root: str,
    batch_size: int,
    num_frames: int = 16,
    split: float = 0.9,
    num_workers: int = 4,
    augment: bool = False,
    pin_memory: bool = True,
):
    """
    Convenience function to create train and validation dataloaders.

    Args:
        root: Directory containing the dataset
        batch_size: Batch size for dataloaders
        num_frames: Number of frames per video
        split: Train/val split ratio
        num_workers: Number of dataloader workers
        augment: Whether to apply data augmentation
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    train_dataset = MovingMNIST(
        root=root,
        num_frames=num_frames,
        train=True,
        split=split,
        random_temporal_crop=augment,
        horizontal_flip=augment,
    )

    val_dataset = MovingMNIST(
        root=root,
        num_frames=num_frames,
        train=False,
        split=split,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
