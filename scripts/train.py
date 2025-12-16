import sys
import os
import torch
from torch.utils.data import DataLoader

# Path hack to allow importing src without installation
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config_parser import load_config, parse_args, merge_config
from src.utils import seed_everything, get_logger
from src.dataset import MovingMNIST
from src.model import NanoSora
from src.trainer import Trainer

def main():
    args = parse_args()

    # Load and merge config
    config_path = args.config
    if not os.path.isabs(config_path):
        # Make path relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '..', config_path)

    config = load_config(config_path)
    config = merge_config(config, args)

    # Set seed for reproducibility
    seed_everything(config['experiment']['seed'])

    # Setup logging
    log_dir = os.path.join(config['experiment']['output_dir'], config['experiment']['name'])
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger("NanoSora", log_dir)

    # Device selection with CUDA availability check
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU (training will be slow)")

    logger.info(f"Config: {config}")

    # Setup data directory
    data_dir = config['data'].get('data_dir', './data')
    if not os.path.isabs(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, '..', data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Create datasets
    train_split = config['data'].get('train_split', 0.9)

    train_dataset = MovingMNIST(
        root=data_dir,
        num_frames=config['data']['num_frames'],
        train=True,
        split=train_split
    )

    val_dataset = MovingMNIST(
        root=data_dir,
        num_frames=config['data']['num_frames'],
        train=False,
        split=train_split
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True  # Ensures consistent batch sizes
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    # Create model
    model = NanoSora(
        input_size=(config['data']['num_frames'], config['data']['image_size'], config['data']['image_size']),
        patch_size=tuple(config['model']['patch_size']),
        hidden_size=config['model']['hidden_size'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        mlp_ratio=config['model'].get('mlp_ratio', 4.0),
        dropout=config['model'].get('dropout', 0.0)
    )

    # Get EMA settings from config (with sensible defaults)
    use_ema = config['training'].get('use_ema', True)
    ema_decay = config['training'].get('ema_decay', 0.9999)

    # Variables for resume
    start_epoch = 1
    best_loss = float('inf')
    resume_ckpt = None

    # Resume from checkpoint if provided (full training state)
    if args.resume:
        logger.info(f"Resuming training from {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume_ckpt['model'])
        start_epoch = resume_ckpt.get('epoch', 0) + 1
        best_loss = resume_ckpt.get('best_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch - 1}, best_loss={best_loss:.6f}")
    # Load model weights only (for inference or fine-tuning)
    elif args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        logger.info("Checkpoint loaded successfully")

    # Create trainer and start training
    trainer = Trainer(
        model,
        train_dataloader,
        config,
        device,
        logger,
        val_dataloader=val_dataloader,
        use_ema=use_ema,
        ema_decay=ema_decay,
        start_epoch=start_epoch,
        best_loss=best_loss
    )

    # Load optimizer/scheduler/EMA state if resuming
    if resume_ckpt is not None:
        if 'optimizer' in resume_ckpt:
            trainer.optimizer.load_state_dict(resume_ckpt['optimizer'])
            logger.info("Optimizer state restored")
        if 'scheduler' in resume_ckpt:
            trainer.scheduler.load_state_dict(resume_ckpt['scheduler'])
            logger.info("Scheduler state restored")
        if 'ema' in resume_ckpt and trainer.use_ema:
            trainer.ema.load_state_dict(resume_ckpt['ema'])
            logger.info("EMA state restored")

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint("interrupted", train_loss=0, val_loss=None, is_best=False)

if __name__ == "__main__":
    main()
