import unittest
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import NanoSora, PatchEmbed3D, DiTBlock, TimestepEmbedder
from src.utils import patchify, unpatchify

class TestPatchifyUnpatchify(unittest.TestCase):
    """Test patchify and unpatchify operations."""

    def test_patch_consistency(self):
        """Test that patchify -> unpatchify is lossless."""
        B, C, T, H, W = 2, 1, 16, 64, 64
        patch_size = (2, 8, 8)

        x = torch.randn(B, C, T, H, W)
        x_p = patchify(x, patch_size)
        x_rec = unpatchify(x_p, patch_size, x.shape)

        self.assertTrue(torch.allclose(x, x_rec, atol=1e-6),
                        "Patchify -> Unpatchify should be lossless")

    def test_patch_shape(self):
        """Test that patchify produces correct output shape."""
        B, C, T, H, W = 4, 1, 16, 64, 64
        patch_size = (2, 8, 8)

        x = torch.randn(B, C, T, H, W)
        x_p = patchify(x, patch_size)

        # Expected: N = (16/2) * (64/8) * (64/8) = 8 * 8 * 8 = 512
        # Patch vol = 1 * 2 * 8 * 8 = 128
        expected_N = (T // patch_size[0]) * (H // patch_size[1]) * (W // patch_size[2])
        expected_vol = C * patch_size[0] * patch_size[1] * patch_size[2]

        self.assertEqual(x_p.shape, (B, expected_N, expected_vol))

    def test_patchify_invalid_shape(self):
        """Test that patchify raises error for invalid input dimensions."""
        x = torch.randn(2, 1, 15, 64, 64)  # 15 not divisible by 2
        patch_size = (2, 8, 8)

        with self.assertRaises(ValueError):
            patchify(x, patch_size)

    def test_unpatchify_invalid_shape(self):
        """Test that unpatchify raises error for invalid input."""
        x = torch.randn(2, 512, 64)  # Wrong patch volume (should be 128)
        patch_size = (2, 8, 8)
        out_shape = (2, 1, 16, 64, 64)

        with self.assertRaises(ValueError):
            unpatchify(x, patch_size, out_shape)


class TestPatchEmbed3D(unittest.TestCase):
    """Test 3D patch embedding."""

    def test_output_shape(self):
        """Test patch embedding produces correct output shape."""
        patch_size = (2, 8, 8)
        embed_dim = 384
        embed = PatchEmbed3D(patch_size, in_chans=1, embed_dim=embed_dim)

        x = torch.randn(2, 1, 16, 64, 64)
        out = embed(x)

        # N = (16/2) * (64/8) * (64/8) = 512
        expected_N = 512
        self.assertEqual(out.shape, (2, expected_N, embed_dim))


class TestDiTBlock(unittest.TestCase):
    """Test DiT block."""

    def test_output_shape(self):
        """Test DiT block preserves shape."""
        hidden_size = 384
        num_heads = 6
        block = DiTBlock(hidden_size, num_heads)

        x = torch.randn(2, 512, hidden_size)
        c = torch.randn(2, hidden_size)
        out = block(x, c)

        self.assertEqual(out.shape, x.shape)

    def test_zero_init(self):
        """Test that AdaLN modulation is zero-initialized."""
        block = DiTBlock(384, 6)

        # The last layer's weights and biases should be zero
        weight = block.adaLN_modulation[-1].weight
        bias = block.adaLN_modulation[-1].bias

        self.assertTrue(torch.allclose(weight, torch.zeros_like(weight)))
        self.assertTrue(torch.allclose(bias, torch.zeros_like(bias)))


class TestTimestepEmbedder(unittest.TestCase):
    """Test timestep embedder."""

    def test_output_shape(self):
        """Test timestep embedder produces correct output shape."""
        hidden_size = 384
        embedder = TimestepEmbedder(hidden_size)

        t = torch.rand(4)
        out = embedder(t)

        self.assertEqual(out.shape, (4, hidden_size))

    def test_sinusoidal_embedding(self):
        """Test sinusoidal embedding properties."""
        t = torch.tensor([0.0, 0.5, 1.0])
        emb = TimestepEmbedder.sinusoidal_embedding(t, 256)

        self.assertEqual(emb.shape, (3, 256))
        # Embedding should be bounded
        self.assertTrue(torch.all(emb >= -1))
        self.assertTrue(torch.all(emb <= 1))


class TestNanoSora(unittest.TestCase):
    """Test the full NanoSora model."""

    def setUp(self):
        """Create a small model for testing."""
        self.model = NanoSora(
            input_size=(16, 64, 64),
            patch_size=(2, 8, 8),
            hidden_size=384,
            depth=2,  # Fewer layers for faster tests
            num_heads=6
        )

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        x = torch.randn(2, 1, 16, 64, 64)
        t = torch.rand(2)
        out = self.model(x, t)

        # Expected: (B, N, patch_vol) = (2, 512, 128)
        expected_N = 512
        expected_patch_vol = 1 * 2 * 8 * 8
        self.assertEqual(out.shape, (2, expected_N, expected_patch_vol))

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        # Create a model without zero-init for testing gradient flow
        model = NanoSora(
            input_size=(16, 64, 64),
            patch_size=(2, 8, 8),
            hidden_size=384,
            depth=2,
            num_heads=6
        )
        # Re-initialize final layer with non-zero weights for gradient test
        torch.nn.init.xavier_uniform_(model.final_linear.weight)

        x = torch.randn(2, 1, 16, 64, 64, requires_grad=True)
        t = torch.rand(2)

        out = model(x, t)
        loss = out.mean()
        loss.backward()

        # Check that input has gradients
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))

    def test_zero_init_final_layer(self):
        """Test that final linear layer is zero-initialized."""
        weight = self.model.final_linear.weight
        bias = self.model.final_linear.bias

        self.assertTrue(torch.allclose(weight, torch.zeros_like(weight)))
        self.assertTrue(torch.allclose(bias, torch.zeros_like(bias)))

    def test_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 1, 16, 64, 64)
            t = torch.rand(batch_size)
            out = self.model(x, t)
            self.assertEqual(out.shape[0], batch_size)

    def test_num_params(self):
        """Test parameter counting."""
        num_params = self.model.get_num_params()
        self.assertGreater(num_params, 0)
        # For depth=2, should have reasonable number of params
        self.assertLess(num_params, 100_000_000)  # Less than 100M


class TestFlowMatchingLoss(unittest.TestCase):
    """Test flow matching loss computation."""

    def setUp(self):
        """Create model for testing."""
        self.model = NanoSora(
            input_size=(16, 64, 64),
            patch_size=(2, 8, 8),
            hidden_size=128,  # Small for testing
            depth=1,
            num_heads=4
        )

    def test_loss_computation(self):
        """Test that loss can be computed and is positive."""
        x1 = torch.randn(2, 1, 16, 64, 64)  # "Real" data
        x0 = torch.randn_like(x1)  # Noise
        t = torch.rand(2)

        # Interpolate
        t_broad = t.view(2, 1, 1, 1, 1)
        xt = t_broad * x1 + (1 - t_broad) * x0

        # Predict
        v_pred = self.model(xt, t)

        # Target
        v_target = patchify(x1 - x0, self.model.patch_size)

        # Loss
        loss = torch.mean((v_pred - v_target) ** 2)

        self.assertGreater(loss.item(), 0)
        self.assertTrue(torch.isfinite(loss))


class TestModelOnCPU(unittest.TestCase):
    """Test model runs on CPU."""

    def test_cpu_forward(self):
        """Test forward pass on CPU."""
        model = NanoSora(
            input_size=(16, 64, 64),
            patch_size=(2, 8, 8),
            hidden_size=128,
            depth=1,
            num_heads=4
        )
        model.eval()

        x = torch.randn(1, 1, 16, 64, 64)
        t = torch.rand(1)

        with torch.no_grad():
            out = model(x, t)

        self.assertEqual(out.shape, (1, 512, 128))


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation."""

    def test_valid_config(self):
        """Test that valid config passes validation."""
        from src.config_parser import validate_config

        config = {
            'experiment': {'name': 'test', 'seed': 42, 'output_dir': '.', 'save_every': 10},
            'data': {'batch_size': 32, 'num_workers': 4, 'num_frames': 16, 'image_size': 64},
            'model': {'patch_size': [2, 8, 8], 'hidden_size': 384, 'depth': 8, 'num_heads': 6},
            'training': {'epochs': 100, 'lr': 0.0003, 'use_amp': True, 'grad_clip': 1.0}
        }

        missing = validate_config(config)
        self.assertEqual(len(missing), 0)

    def test_missing_keys(self):
        """Test that missing keys are detected."""
        from src.config_parser import validate_config

        config = {
            'experiment': {'name': 'test'},  # Missing seed, output_dir, save_every
            'data': {},  # Missing all
        }

        missing = validate_config(config)
        self.assertGreater(len(missing), 0)


if __name__ == "__main__":
    unittest.main()
