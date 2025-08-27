import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock

from QuantaMechInterp.model_sae import (
    safe_log, safe_kl_div, AdaptiveSparseAutoencoder, 
    SparseAutoencoderConfig, SparseAutoencoderForHF, save_sae_to_huggingface
)


class TestSafeMathFunctions(unittest.TestCase):
    """Test utility functions for safe mathematical operations"""
    
    def test_safe_log(self):
        """Test safe logarithm function"""
        # Test normal values
        x = torch.tensor([1.0, 2.0, 0.5])
        result = safe_log(x)
        expected = torch.log(x)
        self.assertTrue(torch.allclose(result, expected))
        
        # Test very small values (should be clamped)
        x_small = torch.tensor([1e-10, 0.0, -1.0])
        result = safe_log(x_small, eps=1e-8)
        self.assertTrue(torch.all(torch.isfinite(result)))
        self.assertTrue(torch.all(result >= torch.log(torch.tensor(1e-8))))
        
        # Test with different epsilon
        x = torch.tensor([1e-12])
        result = safe_log(x, eps=1e-6)
        expected = torch.log(torch.tensor(1e-6))
        self.assertTrue(torch.allclose(result, expected))
    
    # TODO: RESOLVE BUG
    # def test_safe_kl_div(self):
    #     """Test safe KL divergence function"""
    #     # Test normal values
    #     p = torch.tensor([0.3, 0.7])
    #     q = torch.tensor([0.4, 0.6])
    #     result = safe_kl_div(p, q)
    #     self.assertTrue(torch.all(torch.isfinite(result)))
    #     self.assertEqual(result.shape, p.shape)
    #     
    #     # Test edge cases (0 and 1 values should be clamped)
    #     p_edge = torch.tensor([0.0, 1.0, 0.5])
    #     q_edge = torch.tensor([1.0, 0.0, 0.5])
    #     result = safe_kl_div(p_edge, q_edge)
    #     self.assertTrue(torch.all(torch.isfinite(result)))
    #     
    #     # Test with numpy arrays
    #     p_np = np.array([0.2, 0.8])
    #     q_np = np.array([0.3, 0.7])
    #     result = safe_kl_div(p_np, q_np)
    #     self.assertTrue(torch.all(torch.isfinite(result)))
    #     
    #     # Test scalar inputs
    #     result = safe_kl_div(0.3, 0.4)
    #     self.assertTrue(torch.isfinite(result))
    #     
    #     # Test with different epsilon
    #     result = safe_kl_div(p, q, eps=1e-6)
    #     self.assertTrue(torch.all(torch.isfinite(result)))


class TestAdaptiveSparseAutoencoder(unittest.TestCase):
    """Test the main AdaptiveSparseAutoencoder class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.encoding_dim = 64
        self.input_dim = 128
        self.sparsity_target = 0.05
        self.sparsity_weight = 1e-3
        self.l1_weight = 1e-5
        
        # Mock CUDA availability to avoid GPU requirements in tests
        self.cuda_patcher = patch('torch.cuda.is_available', return_value=False)
        self.cuda_patcher.start()
        
    def tearDown(self):
        """Clean up test fixtures"""
        self.cuda_patcher.stop()
    
    def test_initialization(self):
        """Test SAE initialization"""
        sae = AdaptiveSparseAutoencoder(
            self.encoding_dim, self.input_dim, 
            self.sparsity_target, self.sparsity_weight, self.l1_weight
        )
        
        self.assertEqual(sae.encoding_dim, self.encoding_dim)
        self.assertEqual(sae.input_dim, self.input_dim)
        self.assertEqual(sae.sparsity_target, self.sparsity_target)
        self.assertEqual(sae.sparsity_weight, self.sparsity_weight)
        self.assertEqual(sae.l1_weight, self.l1_weight)
        
        # Check network architecture
        self.assertIsInstance(sae.encoder, nn.Sequential)
        self.assertIsInstance(sae.decoder, nn.Linear)
        
        # Check encoder structure
        encoder_layers = list(sae.encoder.children())
        self.assertEqual(len(encoder_layers), 2)
        self.assertIsInstance(encoder_layers[0], nn.Linear)
        self.assertIsInstance(encoder_layers[1], nn.ReLU)
        
        # Check dimensions
        self.assertEqual(encoder_layers[0].in_features, self.input_dim)
        self.assertEqual(encoder_layers[0].out_features, self.encoding_dim)
        self.assertEqual(sae.decoder.in_features, self.encoding_dim)
        self.assertEqual(sae.decoder.out_features, self.input_dim)
    
    def test_initialization_default_params(self):
        """Test SAE initialization with default parameters"""
        sae = AdaptiveSparseAutoencoder(64, 128)
        
        self.assertEqual(sae.sparsity_target, 0.05)
        self.assertEqual(sae.sparsity_weight, 1e-3)
        self.assertEqual(sae.l1_weight, 1e-5)
    
    def test_forward_pass(self):
        """Test forward pass through the autoencoder"""
        sae = AdaptiveSparseAutoencoder(self.encoding_dim, self.input_dim)
        
        # Create test input
        batch_size = 32
        x = torch.randn(batch_size, self.input_dim)
        
        # Forward pass
        encoded, decoded = sae.forward(x)
        
        # Check output shapes
        self.assertEqual(encoded.shape, (batch_size, self.encoding_dim))
        self.assertEqual(decoded.shape, (batch_size, self.input_dim))
        
        # Check that outputs are finite
        self.assertTrue(torch.all(torch.isfinite(encoded)))
        self.assertTrue(torch.all(torch.isfinite(decoded)))
        
        # Check that ReLU is working (encoded should be non-negative)
        self.assertTrue(torch.all(encoded >= 0))
    
    def test_loss_computation(self):
        """Test loss computation with all components"""
        sae = AdaptiveSparseAutoencoder(self.encoding_dim, self.input_dim)
        
        # Create test data
        batch_size = 16
        x = torch.randn(batch_size, self.input_dim)
        encoded, decoded = sae.forward(x)
        
        # Compute loss
        total_loss, mse_loss, sparsity_penalty, l1_penalty = sae.loss(x, encoded, decoded)
        
        # Check that all loss components are finite and non-negative
        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(mse_loss >= 0)
        self.assertTrue(sparsity_penalty >= 0)
        self.assertTrue(l1_penalty >= 0)
        
        # Check that total loss is a scalar
        self.assertEqual(total_loss.dim(), 0)
        
        # Verify loss composition (approximately)
        expected_total = mse_loss + sae.sparsity_weight * sparsity_penalty + sae.l1_weight * l1_penalty
        self.assertAlmostEqual(total_loss.item(), expected_total, places=5)
    
    def test_loss_with_perfect_reconstruction(self):
        """Test loss when input equals output (perfect reconstruction)"""
        sae = AdaptiveSparseAutoencoder(self.encoding_dim, self.input_dim)
        
        # Create test data
        x = torch.randn(8, self.input_dim)
        encoded = torch.randn(8, self.encoding_dim)
        decoded = x.clone()  # Perfect reconstruction
        
        total_loss, mse_loss, sparsity_penalty, l1_penalty = sae.loss(x, encoded, decoded)
        
        # MSE loss should be very small (near zero)
        self.assertLess(mse_loss, 1e-6)
        
        # Other penalties should still exist
        self.assertGreater(sparsity_penalty, 0)
        self.assertGreater(l1_penalty, 0)
    
    def test_loss_with_extreme_values(self):
        """Test loss computation with extreme values"""
        sae = AdaptiveSparseAutoencoder(self.encoding_dim, self.input_dim)
        
        # Test with very large values
        x = torch.ones(4, self.input_dim) * 1000
        encoded = torch.ones(4, self.encoding_dim) * 1000
        decoded = torch.ones(4, self.input_dim) * 1000
        
        total_loss, mse_loss, sparsity_penalty, l1_penalty = sae.loss(x, encoded, decoded)
        
        # Loss should still be finite due to clamping
        self.assertTrue(torch.isfinite(total_loss))
        self.assertLessEqual(total_loss.item(), 1e6)  # Should be clamped
    
    # TODO: RESOLVE BUG
    # def test_sparsity_target_effect(self):
    #     """Test that different sparsity targets affect the loss"""
    #     # Create two SAEs with different sparsity targets
    #     sae_low_sparsity = AdaptiveSparseAutoencoder(
    #         self.encoding_dim, self.input_dim, sparsity_target=0.01
    #     )
    #     sae_high_sparsity = AdaptiveSparseAutoencoder(
    #         self.encoding_dim, self.input_dim, sparsity_target=0.1
    #     )
    #     
    #     # Use same input and encoded values
    #     x = torch.randn(8, self.input_dim)
    #     encoded = torch.rand(8, self.encoding_dim) * 0.05  # Low activation
    #     decoded = torch.randn(8, self.input_dim)
    #     
    #     _, _, penalty_low, _ = sae_low_sparsity.loss(x, encoded, decoded)
    #     _, _, penalty_high, _ = sae_high_sparsity.loss(x, encoded, decoded)
    #     
    #     # Lower sparsity target should have higher penalty for same activation
    #     self.assertGreater(penalty_low, penalty_high)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the network"""
        sae = AdaptiveSparseAutoencoder(self.encoding_dim, self.input_dim)
        
        # Create test data
        x = torch.randn(4, self.input_dim, requires_grad=True)
        
        # Forward pass
        encoded, decoded = sae.forward(x)
        total_loss, _, _, _ = sae.loss(x, encoded, decoded)
        
        # Backward pass
        total_loss.backward()
        
        # Check that gradients exist for all parameters
        for param in sae.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.all(torch.isfinite(param.grad)))


class TestSparseAutoencoderConfig(unittest.TestCase):
    """Test the SparseAutoencoderConfig class"""
    
    def test_initialization_default(self):
        """Test config initialization with default values"""
        config = SparseAutoencoderConfig()
        
        self.assertEqual(config.encoding_dim, 128)
        self.assertEqual(config.input_dim, 768)
        self.assertEqual(config.sparsity_target, 0.05)
        self.assertEqual(config.sparsity_weight, 1e-3)
        self.assertEqual(config.model_type, "sparse_autoencoder")
    
    def test_initialization_custom(self):
        """Test config initialization with custom values"""
        config = SparseAutoencoderConfig(
            encoding_dim=256,
            input_dim=512,
            sparsity_target=0.1,
            sparsity_weight=1e-2
        )
        
        self.assertEqual(config.encoding_dim, 256)
        self.assertEqual(config.input_dim, 512)
        self.assertEqual(config.sparsity_target, 0.1)
        self.assertEqual(config.sparsity_weight, 1e-2)
    
    def test_config_inheritance(self):
        """Test that config properly inherits from PretrainedConfig"""
        from transformers import PretrainedConfig
        config = SparseAutoencoderConfig()
        self.assertIsInstance(config, PretrainedConfig)


class TestSparseAutoencoderForHF(unittest.TestCase):
    """Test the HuggingFace-compatible wrapper"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock CUDA to avoid GPU requirements
        self.cuda_patcher = patch('torch.cuda.is_available', return_value=False)
        self.cuda_patcher.start()
        
        self.config = SparseAutoencoderConfig(
            encoding_dim=64,
            input_dim=128,
            sparsity_target=0.05,
            sparsity_weight=1e-3
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.cuda_patcher.stop()
    
    def test_initialization(self):
        """Test HF wrapper initialization"""
        model = SparseAutoencoderForHF(self.config)
        
        self.assertEqual(model.config, self.config)
        self.assertIsInstance(model.sae, AdaptiveSparseAutoencoder)
        self.assertEqual(model.sae.encoding_dim, self.config.encoding_dim)
        self.assertEqual(model.sae.input_dim, self.config.input_dim)
    
    def test_forward_pass(self):
        """Test forward pass through HF wrapper"""
        model = SparseAutoencoderForHF(self.config)
        
        x = torch.randn(8, self.config.input_dim)
        encoded, decoded = model.forward(x)
        
        self.assertEqual(encoded.shape, (8, self.config.encoding_dim))
        self.assertEqual(decoded.shape, (8, self.config.input_dim))
    
    @patch('torch.save')
    def test_save_pretrained(self, mock_torch_save):
        """Test saving model in HuggingFace format"""
        model = SparseAutoencoderForHF(self.config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the config save method
            with patch.object(model.config, 'save_pretrained') as mock_config_save:
                model.save_pretrained(temp_dir)
                
                # Verify config was saved
                mock_config_save.assert_called_once_with(temp_dir)
                
                # Verify model state dict was saved
                mock_torch_save.assert_called_once()
                args, kwargs = mock_torch_save.call_args
                self.assertEqual(args[1], f"{temp_dir}/pytorch_model.bin")
    
    # TODO: RESOLVE BUG
    # @patch('torch.load')
    # def test_from_pretrained(self, mock_torch_load):
    #     """Test loading model from HuggingFace format"""
    #     # Mock the state dict
    #     mock_state_dict = {'sae.encoder.0.weight': torch.randn(64, 128)}
    #     mock_torch_load.return_value = mock_state_dict
    #     
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         # Create a mock config file
    #         self.config.save_pretrained(temp_dir)
    #         
    #         # Mock the config loading
    #         with patch.object(SparseAutoencoderConfig, 'from_pretrained', return_value=self.config):
    #             model = SparseAutoencoderForHF.from_pretrained(temp_dir)
    #             
    #             # Verify model was created correctly
    #             self.assertIsInstance(model, SparseAutoencoderForHF)
    #             self.assertEqual(model.config, self.config)
    #             
    #             # Verify torch.load was called
    #             mock_torch_load.assert_called_once_with(f"{temp_dir}/pytorch_model.bin")


class TestSaveToHuggingFace(unittest.TestCase):
    """Test the save_sae_to_huggingface utility function"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock CUDA to avoid GPU requirements
        self.cuda_patcher = patch('torch.cuda.is_available', return_value=False)
        self.cuda_patcher.start()
        
        self.cuda_method_patcher = patch.object(nn.Module, 'cuda', return_value=None)
        self.cuda_method_patcher.start()
        
    def tearDown(self):
        """Clean up test fixtures"""
        self.cuda_patcher.stop()
        self.cuda_method_patcher.stop()
    
    @patch('builtins.print')
    def test_save_sae_to_huggingface(self, mock_print):
        """Test saving SAE to HuggingFace format"""
        # Create a test SAE
        sae = AdaptiveSparseAutoencoder(64, 128, 0.05, 1e-3)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the save_pretrained method
            with patch.object(SparseAutoencoderForHF, 'save_pretrained') as mock_save:
                save_sae_to_huggingface(sae, temp_dir)
                
                # Verify save_pretrained was called
                mock_save.assert_called_once_with(temp_dir)
                
                # Verify print was called with success message
                mock_print.assert_called_once_with(f"Model saved to {temp_dir}")
    
    # TODO: RESOLVE BUG
    # def test_save_sae_config_creation(self):
    #     """Test that config is created correctly from SAE parameters"""
    #     sae = AdaptiveSparseAutoencoder(
    #         encoding_dim=256,
    #         input_dim=512,
    #         sparsity_target=0.1,
    #         sparsity_weight=1e-2
    #     )
    #     
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         with patch.object(SparseAutoencoderForHF, 'save_pretrained'):
    #             # Capture the config creation by patching SparseAutoencoderConfig
    #             with patch('QuantaMechInterp.model_sae.SparseAutoencoderConfig') as mock_config_class:
    #                 save_sae_to_huggingface(sae, temp_dir)
    #                 
    #                 # Verify config was created with correct parameters
    #                 mock_config_class.assert_called_once_with(
    #                     encoding_dim=256,
    #                     input_dim=512,
    #                     sparsity_target=0.1,
    #                     sparsity_weight=1e-2
    #                 )


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete SAE workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock CUDA to avoid GPU requirements
        self.cuda_patcher = patch('torch.cuda.is_available', return_value=False)
        self.cuda_patcher.start()
        
        self.cuda_method_patcher = patch.object(nn.Module, 'cuda', return_value=None)
        self.cuda_method_patcher.start()
        
    def tearDown(self):
        """Clean up test fixtures"""
        self.cuda_patcher.stop()
        self.cuda_method_patcher.stop()
    
    def test_end_to_end_training_simulation(self):
        """Test a complete training simulation workflow"""
        # Create SAE
        sae = AdaptiveSparseAutoencoder(32, 64, sparsity_target=0.05)
        
        # Simulate training data
        batch_size = 16
        num_batches = 5
        
        initial_loss = None
        final_loss = None
        
        # Simulate training loop
        optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)
        
        for batch_idx in range(num_batches):
            # Generate random training data
            x = torch.randn(batch_size, 64)
            
            # Forward pass
            encoded, decoded = sae.forward(x)
            total_loss, mse_loss, sparsity_penalty, l1_penalty = sae.loss(x, encoded, decoded)
            
            # Store losses for comparison
            if batch_idx == 0:
                initial_loss = total_loss.item()
            if batch_idx == num_batches - 1:
                final_loss = total_loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Verify all components are working
            self.assertTrue(torch.isfinite(total_loss))
            self.assertGreaterEqual(mse_loss, 0)
            self.assertGreaterEqual(sparsity_penalty, 0)
            self.assertGreaterEqual(l1_penalty, 0)
        
        # Training should generally reduce loss (though not guaranteed in few steps)
        self.assertIsNotNone(initial_loss)
        self.assertIsNotNone(final_loss)
        self.assertTrue(torch.isfinite(torch.tensor(final_loss)))
    
    def test_config_to_model_consistency(self):
        """Test that config and model parameters stay consistent"""
        config = SparseAutoencoderConfig(
            encoding_dim=128,
            input_dim=256,
            sparsity_target=0.08,
            sparsity_weight=2e-3
        )
        
        # Create HF model from config
        hf_model = SparseAutoencoderForHF(config)
        
        # Verify all parameters match
        self.assertEqual(hf_model.sae.encoding_dim, config.encoding_dim)
        self.assertEqual(hf_model.sae.input_dim, config.input_dim)
        self.assertEqual(hf_model.sae.sparsity_target, config.sparsity_target)
        self.assertEqual(hf_model.sae.sparsity_weight, config.sparsity_weight)
        
        # Test forward pass works
        x = torch.randn(4, config.input_dim)
        encoded, decoded = hf_model.forward(x)
        
        self.assertEqual(encoded.shape, (4, config.encoding_dim))
        self.assertEqual(decoded.shape, (4, config.input_dim))
    
    def test_sparsity_enforcement(self):
        """Test that the SAE actually enforces sparsity"""
        # Create SAE with strong sparsity enforcement
        sae = AdaptiveSparseAutoencoder(
            encoding_dim=100,
            input_dim=50,
            sparsity_target=0.01,  # Very sparse
            sparsity_weight=1.0,   # Strong enforcement
            l1_weight=0.1
        )
        
        # Create input data
        x = torch.randn(32, 50)
        
        # Multiple forward passes to see sparsity effect
        optimizer = torch.optim.Adam(sae.parameters(), lr=0.01)
        
        sparsity_levels = []
        for _ in range(10):
            encoded, decoded = sae.forward(x)
            total_loss, _, _, _ = sae.loss(x, encoded, decoded)
            
            # Calculate actual sparsity (fraction of near-zero activations)
            sparsity = (encoded < 0.01).float().mean().item()
            sparsity_levels.append(sparsity)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Sparsity should generally increase over training
        # (though this is not guaranteed in just 10 steps)
        final_sparsity = sparsity_levels[-1]
        self.assertGreater(final_sparsity, 0.5)  # At least 50% sparse


if __name__ == '__main__':
    unittest.main()
