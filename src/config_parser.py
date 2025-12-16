import argparse
import yaml
import os
from typing import Dict, Any, List, Optional

# Required config keys for validation
REQUIRED_KEYS = {
    'experiment': ['name', 'seed', 'output_dir', 'save_every'],
    'data': ['batch_size', 'num_workers', 'num_frames', 'image_size'],
    'model': ['patch_size', 'hidden_size', 'depth', 'num_heads'],
    'training': ['epochs', 'lr', 'use_amp', 'grad_clip'],
}

def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate that all required config keys are present.

    Returns:
        List of missing keys (empty if valid)
    """
    missing = []
    for section, keys in REQUIRED_KEYS.items():
        if section not in config:
            missing.append(f"Section '{section}'")
            continue
        for key in keys:
            if key not in config[section]:
                missing.append(f"'{section}.{key}'")
    return missing

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate a YAML configuration file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Validated configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required keys are missing
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required keys
    missing = validate_config(config)
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    return config

def parse_args() -> argparse.Namespace:
    """
    Parses CLI arguments.
    Allows overriding key parameters for quick experimentation.
    """
    parser = argparse.ArgumentParser(
        description="Nano-Sora: Minimal Diffusion Transformer for Video Generation"
    )

    # Required arguments
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for resuming/inference')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (loads optimizer/scheduler state)')

    # Data overrides
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Override number of data workers')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='Override number of frames')

    # Model overrides
    parser.add_argument('--hidden_size', type=int, default=None,
                        help='Override model hidden size')
    parser.add_argument('--depth', type=int, default=None,
                        help='Override model depth (num layers)')
    parser.add_argument('--num_heads', type=int, default=None,
                        help='Override number of attention heads')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Override dropout rate')

    # Training overrides
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')

    # Experiment overrides
    parser.add_argument('--name', type=str, default=None,
                        help='Override experiment name')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory')

    return parser.parse_args()

def merge_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merges CLI args into the nested config dictionary.
    CLI arguments take precedence over config file values.

    Args:
        config: Base configuration from YAML file
        args: Parsed command line arguments

    Returns:
        Merged configuration dictionary
    """
    # Data overrides
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers
    if args.num_frames is not None:
        config['data']['num_frames'] = args.num_frames

    # Model overrides
    if args.hidden_size is not None:
        config['model']['hidden_size'] = args.hidden_size
    if args.depth is not None:
        config['model']['depth'] = args.depth
    if args.num_heads is not None:
        config['model']['num_heads'] = args.num_heads
    if args.dropout is not None:
        config['model']['dropout'] = args.dropout

    # Training overrides
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.no_amp:
        config['training']['use_amp'] = False

    # Experiment overrides
    if args.name is not None:
        config['experiment']['name'] = args.name
    if args.seed is not None:
        config['experiment']['seed'] = args.seed
    if args.output_dir is not None:
        config['experiment']['output_dir'] = args.output_dir

    return config

def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration to a YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def print_config(config: Dict[str, Any]) -> None:
    """Pretty print configuration."""
    print("\n" + "=" * 50)
    print("Configuration")
    print("=" * 50)
    for section, values in config.items():
        print(f"\n[{section}]")
        if isinstance(values, dict):
            for key, value in values.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {values}")
    print("=" * 50 + "\n")
