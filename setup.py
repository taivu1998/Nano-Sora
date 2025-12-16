from setuptools import setup, find_packages

setup(
    name="nano-sora",
    version="0.1.0",
    description="Nano-Sora: A minimal Diffusion Transformer for video generation using Rectified Flow",
    author="Research Engineer",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "numpy<2.0.0",
        "torch>=2.1.0",
        "torchvision",
        "pyyaml",
        "tqdm",
        "matplotlib",
        "requests",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov"],
    },
    entry_points={
        "console_scripts": [
            "nano-sora-train=scripts.train:main",
            "nano-sora-inference=scripts.inference:main",
        ],
    },
)
