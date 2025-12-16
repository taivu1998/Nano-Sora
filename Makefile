.PHONY: install install-dev train inference test clean lint help

# Default target
help:
	@echo "Nano-Sora: Minimal Diffusion Transformer for Video Generation"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make install-dev  - Install with dev dependencies"
	@echo "  make train        - Start training"
	@echo "  make inference    - Run inference on trained model"
	@echo "  make test         - Run unit tests"
	@echo "  make clean        - Clean generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make train                                    # Train with default config"
	@echo "  make inference CKPT=logs/nano_sora_flow/best_model.pt"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Install with development dependencies
install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Training
train:
	python scripts/train.py --config configs/default.yaml

# Training with custom batch size
train-small:
	python scripts/train.py --config configs/default.yaml --batch_size 16

# Quick training for debugging (fewer epochs)
train-debug:
	python scripts/train.py --config configs/default.yaml --epochs 2 --batch_size 8

# Inference (requires CKPT variable or uses default path)
CKPT ?= logs/nano_sora_flow/best_model.pt
inference:
	@if [ -f "$(CKPT)" ]; then \
		python scripts/inference.py --checkpoint $(CKPT) --output_dir outputs; \
	else \
		echo "Error: Checkpoint not found at $(CKPT)"; \
		echo "Usage: make inference CKPT=path/to/checkpoint.pt"; \
		exit 1; \
	fi

# Inference with Heun sampler (better quality)
inference-heun:
	@if [ -f "$(CKPT)" ]; then \
		python scripts/inference.py --checkpoint $(CKPT) --output_dir outputs --use_heun; \
	else \
		echo "Error: Checkpoint not found at $(CKPT)"; \
		exit 1; \
	fi

# Run unit tests
test:
	python -m pytest tests/ -v

# Run quick tests (single test file)
test-quick:
	python -m unittest tests/test_model.py -v

# Clean generated files
clean:
	rm -rf logs/ outputs/ __pycache__ */__pycache__ */*/__pycache__ *.pyc
	rm -rf .pytest_cache/ *.egg-info/ build/ dist/
	rm -f *.png  # Clean generated images in root

# Clean everything including data
clean-all: clean
	rm -rf data/

# Lint code (requires dev dependencies)
lint:
	@echo "Running linters..."
	python -m flake8 src/ scripts/ tests/ --max-line-length 120 --ignore E501,W503

# Format code (requires black)
format:
	python -m black src/ scripts/ tests/ --line-length 120

# Check GPU availability
check-gpu:
	python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
