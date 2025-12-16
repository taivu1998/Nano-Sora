"""Inference script for Nano-Sora with video output support."""

import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import NanoSora
from src.utils import unpatchify, patchify


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@torch.no_grad()
def sample_euler(model, shape, steps=50, device='cuda'):
    """
    Euler ODE Solver for Rectified Flow.

    The flow goes from noise (t=0) to data (t=1).
    At each step, we predict v = dx/dt and integrate: x_{t+dt} = x_t + v * dt
    """
    model.eval()
    x = torch.randn(shape, device=device)  # Start at x0 (noise)
    dt = 1.0 / steps

    for i in tqdm(range(steps), desc="Sampling (Euler)"):
        t = torch.ones(shape[0], device=device) * (i / steps)
        v_pred_patches = model(x, t)
        v_pred = unpatchify(v_pred_patches, model.patch_size, shape)
        x = x + v_pred * dt  # Euler step: x_{t+1} = x_t + v * dt

    return x


@torch.no_grad()
def sample_heun(model, shape, steps=50, device='cuda'):
    """
    Heun's method (improved Euler) for better accuracy.
    Uses a predictor-corrector approach.
    """
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps

    for i in tqdm(range(steps), desc="Sampling (Heun)"):
        t = torch.ones(shape[0], device=device) * (i / steps)
        t_next = torch.ones(shape[0], device=device) * ((i + 1) / steps)

        # Predictor (Euler)
        v1_patches = model(x, t)
        v1 = unpatchify(v1_patches, model.patch_size, shape)
        x_pred = x + v1 * dt

        # Corrector
        v2_patches = model(x_pred, t_next)
        v2 = unpatchify(v2_patches, model.patch_size, shape)

        # Average
        x = x + 0.5 * (v1 + v2) * dt

    return x


def video_to_numpy(video: torch.Tensor) -> np.ndarray:
    """Convert video tensor to numpy array for visualization.

    Args:
        video: (B, C, T, H, W) tensor in [-1, 1]

    Returns:
        (B, T, H, W) numpy array in [0, 255] uint8
    """
    video = video.cpu().numpy()
    video = (video + 1) / 2  # [-1, 1] -> [0, 1]
    video = np.clip(video, 0, 1)
    video = (video * 255).astype(np.uint8)
    return video[:, 0]  # Remove channel dim for grayscale


def save_video_as_gif(video: np.ndarray, save_path: str, fps: int = 8, loop: int = 0):
    """Save video as GIF.

    Args:
        video: (T, H, W) numpy array in [0, 255] uint8
        save_path: Output path
        fps: Frames per second
        loop: Number of loops (0 = infinite)
    """
    frames = [Image.fromarray(video[t]) for t in range(video.shape[0])]
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=loop
    )
    print(f"Saved GIF: {save_path}")


def save_video_as_mp4(video: np.ndarray, save_path: str, fps: int = 8):
    """Save video as MP4 using matplotlib animation."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('off')

    im = ax.imshow(video[0], cmap='gray', vmin=0, vmax=255)

    def update(frame):
        im.set_array(video[frame])
        return [im]

    anim = animation.FuncAnimation(
        fig, update, frames=video.shape[0], interval=1000//fps, blit=True
    )
    anim.save(save_path, writer='ffmpeg', fps=fps, dpi=150)
    plt.close()
    print(f"Saved MP4: {save_path}")


def save_video_grid_gif(videos: np.ndarray, save_path: str, fps: int = 8, ncols: int = 2):
    """Save multiple videos as a grid GIF.

    Args:
        videos: (N, T, H, W) numpy array
        save_path: Output path
        fps: Frames per second
        ncols: Number of columns in grid
    """
    N, T, H, W = videos.shape
    nrows = (N + ncols - 1) // ncols

    # Create grid frames
    grid_frames = []
    for t in range(T):
        grid = np.zeros((nrows * H, ncols * W), dtype=np.uint8)
        for i in range(N):
            row, col = i // ncols, i % ncols
            grid[row*H:(row+1)*H, col*W:(col+1)*W] = videos[i, t]
        grid_frames.append(Image.fromarray(grid))

    grid_frames[0].save(
        save_path,
        save_all=True,
        append_images=grid_frames[1:],
        duration=1000 // fps,
        loop=0
    )
    print(f"Saved grid GIF: {save_path}")


def visualize_samples_strip(samples, config, output_dir='.'):
    """Save generated video samples as image strips (legacy format)."""
    num_frames = config['data']['num_frames']

    for i, sample in enumerate(samples):
        # sample shape: (B, C, T, H, W) = (1, 1, 16, 64, 64)
        # Extract: (T, H, W)
        video = sample[0, 0].cpu().numpy()  # (T, H, W)
        video = (video + 1) / 2.0  # Denormalize to [0, 1]
        video = np.clip(video, 0, 1)

        # Create horizontal grid of frames
        grid = np.concatenate([video[t] for t in range(num_frames)], axis=1)

        output_path = os.path.join(output_dir, f"sample_{i}_strip.png")
        plt.imsave(output_path, grid, cmap='gray')
        print(f"Saved strip: {output_path}")


def visualize_attention(model, config, device, output_dir='.'):
    """
    The 'Gabriel Goh' Spacetime Attention Metric.
    Visualizes which spacetime regions a center token attends to.
    """
    print("Extracting Spacetime Attention Map...")
    attn_maps = []

    def hook_fn(module, input, output):
        # MultiheadAttention returns (output, attn_weights)
        # attn_weights shape: (B, N, N) when need_weights=True
        if output[1] is not None:
            attn_maps.append(output[1].detach().cpu())

    # Enable attention weights
    handle = model.blocks[-1].attn.register_forward_hook(hook_fn)

    # Get shape from config
    num_frames = config['data']['num_frames']
    image_size = config['data']['image_size']
    shape = (1, 1, num_frames, image_size, image_size)

    # Run dummy inference (1 step to get attention)
    sample_euler(model, shape, steps=1, device=device)
    handle.remove()

    if not attn_maps:
        print("Warning: No attention maps captured. Skipping visualization.")
        return

    # Plot: Center Token Attention
    pt, ph, pw = model.patch_size
    T_grid = num_frames // pt
    H_grid = image_size // ph
    W_grid = image_size // pw
    N = T_grid * H_grid * W_grid

    # Pick the token at the physical center of the volume
    center_idx = N // 2

    attn_row = attn_maps[0][0, center_idx, :]  # (N,)

    # Reshape back to (T_grid, H_grid, W_grid)
    attn_vol = attn_row.view(T_grid, H_grid, W_grid)

    # Plot heatmap strips
    fig, axs = plt.subplots(1, T_grid, figsize=(3 * T_grid, 3))
    if T_grid == 1:
        axs = [axs]
    for t in range(T_grid):
        axs[t].imshow(attn_vol[t].numpy(), cmap='magma')
        axs[t].axis('off')
        axs[t].set_title(f"t={t}")

    output_path = os.path.join(output_dir, "attention_spacetime.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention map: {output_path}")


def compute_test_mse(model, test_loader, config, device):
    """
    Compute MSE on test set for quantitative evaluation.
    This measures how well the model can predict the velocity field.
    """
    model.eval()
    total_mse = 0
    num_batches = 0

    with torch.no_grad():
        for x1 in test_loader:
            x1 = x1.to(device)
            b = x1.shape[0]

            # Sample noise and timesteps
            x0 = torch.randn_like(x1)
            t = torch.rand(b, device=device)

            # Interpolate
            t_broad = t.view(b, 1, 1, 1, 1)
            xt = t_broad * x1 + (1 - t_broad) * x0

            # Predict velocity
            v_pred = model(xt, t)
            v_target = patchify(x1 - x0, model.patch_size)

            mse = torch.mean((v_pred - v_target) ** 2).item()
            total_mse += mse
            num_batches += 1

    return total_mse / num_batches if num_batches > 0 else float('inf')


def main():
    parser = argparse.ArgumentParser(description="Nano-Sora Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of samples to generate')
    parser.add_argument('--steps', type=int, default=None, help='Number of Euler steps (default: from config)')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory for samples')
    parser.add_argument('--fps', type=int, default=8, help='Frames per second for video output')
    parser.add_argument('--use_heun', action='store_true', help='Use Heun method for better quality')
    parser.add_argument('--save_mp4', action='store_true', help='Also save as MP4 (requires ffmpeg)')
    parser.add_argument('--compute_mse', action='store_true', help='Compute test set MSE')
    parser.add_argument('--use_ema', action='store_true', default=True, help='Use EMA weights if available (default: True)')
    parser.add_argument('--no_ema', action='store_true', help='Disable EMA weights, use raw model weights')
    args = parser.parse_args()

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
    device = get_device()
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']

    print(f"Loaded checkpoint: {config['experiment']['name']}")
    if 'train_loss' in ckpt:
        print(f"  Train loss: {ckpt['train_loss']:.6f}")
    if 'val_loss' in ckpt and ckpt['val_loss'] is not None:
        print(f"  Val loss: {ckpt['val_loss']:.6f}")

    # Build model from config
    model = NanoSora(
        input_size=(config['data']['num_frames'], config['data']['image_size'], config['data']['image_size']),
        patch_size=tuple(config['model']['patch_size']),
        hidden_size=config['model']['hidden_size'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        mlp_ratio=config['model'].get('mlp_ratio', 4.0),
        dropout=config['model'].get('dropout', 0.0)
    ).to(device)

    # Determine which weights to load (EMA preferred if available)
    use_ema_weights = args.use_ema and not args.no_ema
    if use_ema_weights and 'model_ema' in ckpt:
        model.load_state_dict(ckpt['model_ema'])
        print("Loaded EMA weights (smoother, typically better quality)")
    else:
        model.load_state_dict(ckpt['model'])
        if use_ema_weights:
            print("EMA weights not found in checkpoint, using regular weights")
        else:
            print("Using regular model weights (EMA disabled)")
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Determine number of steps
    steps = args.steps if args.steps else config['training'].get('flow_steps', 50)
    print(f"Sampling with {steps} steps using {'Heun' if args.use_heun else 'Euler'} method")

    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    shape = (1, 1, config['data']['num_frames'], config['data']['image_size'], config['data']['image_size'])

    sampler = sample_heun if args.use_heun else sample_euler

    samples = []
    for i in range(args.num_samples):
        sample = sampler(model, shape, steps=steps, device=device)
        samples.append(sample)

    # Stack samples for batch processing
    all_samples = torch.cat(samples, dim=0)  # (N, 1, T, H, W)
    videos = video_to_numpy(all_samples)  # (N, T, H, W)

    # Save individual GIFs
    print("\nSaving outputs...")
    for i in range(args.num_samples):
        save_video_as_gif(videos[i], os.path.join(args.output_dir, f"sample_{i}.gif"), fps=args.fps)

    # Save grid GIF
    if args.num_samples > 1:
        save_video_grid_gif(videos, os.path.join(args.output_dir, "samples_grid.gif"), fps=args.fps)

    # Save strip images (legacy format)
    visualize_samples_strip(samples, config, args.output_dir)

    # Save MP4 if requested
    if args.save_mp4:
        try:
            for i in range(args.num_samples):
                save_video_as_mp4(videos[i], os.path.join(args.output_dir, f"sample_{i}.mp4"), fps=args.fps)
        except Exception as e:
            print(f"Could not save MP4 (ffmpeg may not be installed): {e}")

    # Visualize attention
    print("\nGenerating attention visualization...")
    visualize_attention(model, config, device, args.output_dir)

    # Compute test MSE if requested
    if args.compute_mse:
        print("\nComputing test set MSE...")
        from torch.utils.data import DataLoader
        from src.dataset import MovingMNIST

        # Setup data directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, '..', config['data'].get('data_dir', './data'))

        test_dataset = MovingMNIST(
            root=data_dir,
            num_frames=config['data']['num_frames'],
            train=False,
            split=config['data'].get('train_split', 0.9)
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=0
        )

        mse = compute_test_mse(model, test_loader, config, device)
        print(f"Test Set MSE: {mse:.6f}")

    print(f"\n{'='*50}")
    print(f"All outputs saved to: {args.output_dir}")
    print(f"  - GIFs: sample_*.gif, samples_grid.gif")
    print(f"  - Strips: sample_*_strip.png")
    print(f"  - Attention: attention_spacetime.png")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
