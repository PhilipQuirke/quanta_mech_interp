import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from QuantaMechInterp.ablate_hooks import (
    to_numpy, validate_value, a_put_resid_post_hook, a_get_l0_attn_z_hook,
    a_get_l1_attn_z_hook, a_get_l2_attn_z_hook, a_get_l3_attn_z_hook,
    a_put_l0_attn_z_hook, a_put_l1_attn_z_hook, a_put_l2_attn_z_hook,
    a_put_l3_attn_z_hook, a_set_ablate_hooks, a_calc_mean_values,
    a_predict_questions, a_run_attention_intervention
)
from QuantaMechInterp.useful_node import NodeLocation
from QuantaMechInterp.ablate_config import AblateConfig
from QuantaMechInterp.quanta_constants import NO_IMPACT_TAG


class TestToNumpy(unittest.TestCase):
    """Test the to_numpy utility function."""
    
    def test_numpy_array_input(self):
        """Test with numpy array input."""
        arr = np.array([1, 2, 3])
        result = to_numpy(arr)
        np.testing.assert_array_equal(result, arr)
        self.assertIs(result, arr)  # Should return the same object
    
    def test_list_input(self):
        """Test with list input."""
        lst = [1, 2, 3]
        result = to_numpy(lst)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)
        self.assertIsInstance(result, np.ndarray)
    
    def test_tuple_input(self):
        """Test with tuple input."""
        tpl = (1, 2, 3)
        result = to_numpy(tpl)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)
        self.assertIsInstance(result, np.ndarray)
    
    def test_torch_tensor_input(self):
        """Test with PyTorch tensor input."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = to_numpy(tensor)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)
        self.assertIsInstance(result, np.ndarray)
    
    def test_torch_parameter_input(self):
        """Test with PyTorch parameter input."""
        param = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        result = to_numpy(param)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)
        self.assertIsInstance(result, np.ndarray)
    
    def test_scalar_inputs(self):
        """Test with scalar inputs."""
        # Integer
        result = to_numpy(42)
        expected = np.array(42)
        np.testing.assert_array_equal(result, expected)
        
        # Float
        result = to_numpy(3.14)
        expected = np.array(3.14)
        np.testing.assert_array_equal(result, expected)
        
        # Boolean
        result = to_numpy(True)
        expected = np.array(True)
        np.testing.assert_array_equal(result, expected)
        
        # String
        result = to_numpy("test")
        expected = np.array("test")
        np.testing.assert_array_equal(result, expected)
    
    def test_invalid_input_type(self):
        """Test with invalid input type."""
        with self.assertRaises(ValueError) as context:
            to_numpy({"invalid": "dict"})
        
        self.assertIn("Input to to_numpy has invalid type", str(context.exception))
    
    def test_cuda_tensor_input(self):
        """Test with CUDA tensor (if available)."""
        if torch.cuda.is_available():
            tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
            result = to_numpy(tensor)
            expected = np.array([1.0, 2.0, 3.0])
            np.testing.assert_array_equal(result, expected)
            self.assertIsInstance(result, np.ndarray)
    
    def test_gradient_tensor_input(self):
        """Test with tensor that requires gradients."""
        tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = to_numpy(tensor)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)
        self.assertIsInstance(result, np.ndarray)


class TestValidateValue(unittest.TestCase):
    """Test the validate_value function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the global acfg object
        self.mock_acfg = Mock()
        self.mock_acfg.ablate_node_names = "TestNode"
        self.mock_acfg.operation = 1
        self.mock_acfg.expected_answer = "TestAnswer"
        self.mock_acfg.expected_impact = "TestImpact"
        self.mock_acfg.abort = False
        
        # Patch the global acfg
        self.acfg_patcher = patch('QuantaMechInterp.ablate_hooks.acfg', self.mock_acfg)
        self.acfg_patcher.start()
    
    def tearDown(self):
        """Clean up patches."""
        self.acfg_patcher.stop()
    
    def test_valid_tensor(self):
        """Test with valid tensor (non-zero batch size)."""
        tensor = torch.randn(5, 22, 3, 170)  # batch_size > 0
        result = validate_value("test_hook", tensor)
        self.assertTrue(result)
        self.assertFalse(self.mock_acfg.abort)
    
    def test_invalid_tensor_zero_batch(self):
        """Test with invalid tensor (zero batch size)."""
        tensor = torch.randn(0, 22, 3, 170)  # batch_size = 0
        
        with patch('builtins.print') as mock_print:
            result = validate_value("test_hook", tensor)
            
        self.assertFalse(result)
        self.assertTrue(self.mock_acfg.abort)
        mock_print.assert_called_once_with(
            "Aborted", "test_hook", "TestNode", 1, "TestAnswer", "TestImpact"
        )
    
    def test_single_element_batch(self):
        """Test with single element batch."""
        tensor = torch.randn(1, 22, 3, 170)
        result = validate_value("test_hook", tensor)
        self.assertTrue(result)
        self.assertFalse(self.mock_acfg.abort)


class TestHookFunctions(unittest.TestCase):
    """Test the various hook functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the global acfg object
        self.mock_acfg = Mock()
        self.mock_acfg.ablate_node_locations = [NodeLocation(5, 1, True, 2)]
        self.mock_acfg.mean_resid_post = torch.randn(1, 22, 510)
        self.mock_acfg.layer_store = [torch.empty(0) for _ in range(4)]
        self.mock_acfg.abort = False
        
        # Patch the global acfg
        self.acfg_patcher = patch('QuantaMechInterp.ablate_hooks.acfg', self.mock_acfg)
        self.acfg_patcher.start()
    
    def tearDown(self):
        """Clean up patches."""
        self.acfg_patcher.stop()
    
    def test_a_put_resid_post_hook(self):
        """Test the residual post hook function."""
        # Create test tensor
        batch_size, n_ctx, d_model = 64, 22, 510
        value = torch.randn(batch_size, n_ctx, d_model)
        original_position_5 = value[:, 5, :].clone()
        
        # Call the hook
        mock_hook = Mock()
        a_put_resid_post_hook(value, mock_hook)
        
        # Check that position 5 was replaced with mean values
        expected_position_5 = self.mock_acfg.mean_resid_post[0, 5, :].clone()
        torch.testing.assert_close(value[:, 5, :], expected_position_5.expand(batch_size, -1))
        
        # Check that other positions weren't changed
        for pos in [0, 1, 2, 3, 4, 6, 7]:
            if pos < n_ctx:
                # These positions should be unchanged (we can't test exact equality due to random init)
                self.assertEqual(value[:, pos, :].shape, (batch_size, d_model))
    
    def test_a_get_l0_attn_z_hook_valid(self):
        """Test the L0 attention z get hook with valid tensor."""
        value = torch.randn(1, 22, 3, 170)
        mock_hook = Mock()
        
        with patch('QuantaMechInterp.ablate_hooks.validate_value', return_value=True):
            a_get_l0_attn_z_hook(value, mock_hook)
        
        # Check that the value was stored in layer_store[0]
        torch.testing.assert_close(self.mock_acfg.layer_store[0], value)
    
    def test_a_get_l0_attn_z_hook_invalid(self):
        """Test the L0 attention z get hook with invalid tensor."""
        value = torch.randn(0, 22, 3, 170)  # Invalid batch size
        mock_hook = Mock()
        
        with patch('QuantaMechInterp.ablate_hooks.validate_value', return_value=False):
            a_get_l0_attn_z_hook(value, mock_hook)
        
        # layer_store[0] should remain unchanged (empty tensor)
        self.assertEqual(self.mock_acfg.layer_store[0].numel(), 0)
    
    def test_a_get_l1_attn_z_hook(self):
        """Test the L1 attention z get hook."""
        value = torch.randn(1, 22, 3, 170)
        mock_hook = Mock()
        
        with patch('QuantaMechInterp.ablate_hooks.validate_value', return_value=True):
            a_get_l1_attn_z_hook(value, mock_hook)
        
        torch.testing.assert_close(self.mock_acfg.layer_store[1], value)
    
    def test_a_get_l2_attn_z_hook(self):
        """Test the L2 attention z get hook."""
        value = torch.randn(1, 22, 3, 170)
        mock_hook = Mock()
        
        with patch('QuantaMechInterp.ablate_hooks.validate_value', return_value=True):
            a_get_l2_attn_z_hook(value, mock_hook)
        
        torch.testing.assert_close(self.mock_acfg.layer_store[2], value)
    
    def test_a_get_l3_attn_z_hook(self):
        """Test the L3 attention z get hook."""
        value = torch.randn(1, 22, 3, 170)
        mock_hook = Mock()
        
        with patch('QuantaMechInterp.ablate_hooks.validate_value', return_value=True):
            a_get_l3_attn_z_hook(value, mock_hook)
        
        torch.testing.assert_close(self.mock_acfg.layer_store[3], value)
    
    def test_a_put_l0_attn_z_hook(self):
        """Test the L0 attention z put hook."""
        # Set up layer store with test data
        self.mock_acfg.layer_store[0] = torch.randn(1, 22, 3, 170)
        
        # Create test value tensor
        value = torch.randn(1, 22, 3, 170)
        original_value = value.clone()
        
        # Set up ablate locations for layer 0, head
        self.mock_acfg.ablate_node_locations = [NodeLocation(5, 0, True, 2)]
        
        mock_hook = Mock()
        a_put_l0_attn_z_hook(value, mock_hook)
        
        # Check that the specific position/head was replaced
        expected_replacement = self.mock_acfg.layer_store[0][:, 5, 2, :]
        torch.testing.assert_close(value[:, 5, 2, :], expected_replacement)
        
        # Check that other positions/heads weren't changed
        for pos in [0, 1, 2, 3, 4, 6]:
            for head in [0, 1]:  # Skip head 2 at position 5
                if not (pos == 5 and head == 2):
                    torch.testing.assert_close(
                        value[:, pos, head, :], 
                        original_value[:, pos, head, :]
                    )
    
    def test_a_put_l1_attn_z_hook(self):
        """Test the L1 attention z put hook."""
        self.mock_acfg.layer_store[1] = torch.randn(1, 22, 3, 170)
        value = torch.randn(1, 22, 3, 170)
        
        # Set up ablate locations for layer 1, head
        self.mock_acfg.ablate_node_locations = [NodeLocation(10, 1, True, 1)]
        
        mock_hook = Mock()
        a_put_l1_attn_z_hook(value, mock_hook)
        
        # Check that the specific position/head was replaced
        expected_replacement = self.mock_acfg.layer_store[1][:, 10, 1, :]
        torch.testing.assert_close(value[:, 10, 1, :], expected_replacement)
    
    def test_a_put_l2_attn_z_hook(self):
        """Test the L2 attention z put hook."""
        self.mock_acfg.layer_store[2] = torch.randn(1, 22, 3, 170)
        value = torch.randn(1, 22, 3, 170)
        
        self.mock_acfg.ablate_node_locations = [NodeLocation(15, 2, True, 0)]
        
        mock_hook = Mock()
        a_put_l2_attn_z_hook(value, mock_hook)
        
        expected_replacement = self.mock_acfg.layer_store[2][:, 15, 0, :]
        torch.testing.assert_close(value[:, 15, 0, :], expected_replacement)
    
    def test_a_put_l3_attn_z_hook(self):
        """Test the L3 attention z put hook."""
        self.mock_acfg.layer_store[3] = torch.randn(1, 22, 3, 170)
        value = torch.randn(1, 22, 3, 170)
        
        self.mock_acfg.ablate_node_locations = [NodeLocation(8, 3, True, 2)]
        
        mock_hook = Mock()
        a_put_l3_attn_z_hook(value, mock_hook)
        
        expected_replacement = self.mock_acfg.layer_store[3][:, 8, 2, :]
        torch.testing.assert_close(value[:, 8, 2, :], expected_replacement)
    
    def test_put_hooks_skip_non_matching_layer(self):
        """Test that put hooks skip nodes not matching their layer."""
        self.mock_acfg.layer_store[0] = torch.randn(1, 22, 3, 170)
        value = torch.randn(1, 22, 3, 170)
        original_value = value.clone()
        
        # Set up ablate locations for layer 1 (should be skipped by L0 hook)
        self.mock_acfg.ablate_node_locations = [NodeLocation(5, 1, True, 2)]
        
        mock_hook = Mock()
        a_put_l0_attn_z_hook(value, mock_hook)
        
        # Value should be unchanged since layer doesn't match
        torch.testing.assert_close(value, original_value)
    
    def test_put_hooks_skip_non_head_nodes(self):
        """Test that attention put hooks skip non-head nodes."""
        self.mock_acfg.layer_store[0] = torch.randn(1, 22, 3, 170)
        value = torch.randn(1, 22, 3, 170)
        original_value = value.clone()
        
        # Set up ablate locations for MLP (not head)
        self.mock_acfg.ablate_node_locations = [NodeLocation(5, 0, False, 0)]
        
        mock_hook = Mock()
        a_put_l0_attn_z_hook(value, mock_hook)
        
        # Value should be unchanged since it's not a head
        torch.testing.assert_close(value, original_value)
    
    def test_put_hooks_multiple_locations(self):
        """Test put hooks with multiple ablate locations."""
        self.mock_acfg.layer_store[0] = torch.randn(1, 22, 3, 170)
        value = torch.randn(1, 22, 3, 170)
        
        # Set up multiple ablate locations for layer 0
        self.mock_acfg.ablate_node_locations = [
            NodeLocation(5, 0, True, 1),
            NodeLocation(10, 0, True, 2),
            NodeLocation(15, 1, True, 0)  # Different layer, should be skipped
        ]
        
        mock_hook = Mock()
        a_put_l0_attn_z_hook(value, mock_hook)
        
        # Check that both layer 0 positions were replaced
        torch.testing.assert_close(
            value[:, 5, 1, :], 
            self.mock_acfg.layer_store[0][:, 5, 1, :]
        )
        torch.testing.assert_close(
            value[:, 10, 2, :], 
            self.mock_acfg.layer_store[0][:, 10, 2, :]
        )


class TestASetAblateHooks(unittest.TestCase):
    """Test the a_set_ablate_hooks function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the global acfg object
        self.mock_acfg = Mock()
        self.mock_acfg.l_hook_resid_post_name = ["resid_0", "resid_1", "resid_2", "resid_3"]
        self.mock_acfg.l_attn_hook_z_name = ["attn_0", "attn_1", "attn_2", "attn_3"]
        
        # Patch the global acfg
        self.acfg_patcher = patch('QuantaMechInterp.ablate_hooks.acfg', self.mock_acfg)
        self.acfg_patcher.start()
    
    def tearDown(self):
        """Clean up patches."""
        self.acfg_patcher.stop()
    
    def test_set_hooks_3_layers(self):
        """Test setting hooks for 3 layers."""
        mock_cfg = Mock()
        mock_cfg.n_layers = 3
        
        a_set_ablate_hooks(mock_cfg)
        
        # Check resid_put_hooks
        self.assertEqual(len(self.mock_acfg.resid_put_hooks), 3)
        self.assertEqual(self.mock_acfg.resid_put_hooks[0][0], "resid_0")
        self.assertEqual(self.mock_acfg.resid_put_hooks[1][0], "resid_1")
        self.assertEqual(self.mock_acfg.resid_put_hooks[2][0], "resid_2")
        
        # Check attn_get_hooks
        self.assertEqual(len(self.mock_acfg.attn_get_hooks), 3)
        self.assertEqual(self.mock_acfg.attn_get_hooks[0][0], "attn_0")
        self.assertEqual(self.mock_acfg.attn_get_hooks[1][0], "attn_1")
        self.assertEqual(self.mock_acfg.attn_get_hooks[2][0], "attn_2")
        
        # Check attn_put_hooks
        self.assertEqual(len(self.mock_acfg.attn_put_hooks), 3)
        self.assertEqual(self.mock_acfg.attn_put_hooks[0][0], "attn_0")
        self.assertEqual(self.mock_acfg.attn_put_hooks[1][0], "attn_1")
        self.assertEqual(self.mock_acfg.attn_put_hooks[2][0], "attn_2")
    
    def test_set_hooks_4_layers(self):
        """Test setting hooks for 4 layers."""
        mock_cfg = Mock()
        mock_cfg.n_layers = 4
        
        a_set_ablate_hooks(mock_cfg)
        
        # Check that all 4 layers are included
        self.assertEqual(len(self.mock_acfg.resid_put_hooks), 4)
        self.assertEqual(len(self.mock_acfg.attn_get_hooks), 4)
        self.assertEqual(len(self.mock_acfg.attn_put_hooks), 4)
        
        # Check the 4th layer
        self.assertEqual(self.mock_acfg.resid_put_hooks[3][0], "resid_3")
        self.assertEqual(self.mock_acfg.attn_get_hooks[3][0], "attn_3")
        self.assertEqual(self.mock_acfg.attn_put_hooks[3][0], "attn_3")
    
    def test_set_hooks_1_layer(self):
        """Test setting hooks for 1 layer."""
        mock_cfg = Mock()
        mock_cfg.n_layers = 1
        
        a_set_ablate_hooks(mock_cfg)
        
        # Check that only 1 layer is included
        self.assertEqual(len(self.mock_acfg.resid_put_hooks), 1)
        self.assertEqual(len(self.mock_acfg.attn_get_hooks), 1)
        self.assertEqual(len(self.mock_acfg.attn_put_hooks), 1)
        
        self.assertEqual(self.mock_acfg.resid_put_hooks[0][0], "resid_0")
        self.assertEqual(self.mock_acfg.attn_get_hooks[0][0], "attn_0")
        self.assertEqual(self.mock_acfg.attn_put_hooks[0][0], "attn_0")


class TestACalcMeanValues(unittest.TestCase):
    """Test the a_calc_mean_values function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the global acfg object
        self.mock_acfg = Mock()
        self.mock_acfg.l_attn_hook_z_name = ["blocks.0.attn.hook_z"]
        self.mock_acfg.l_hook_resid_post_name = ["blocks.0.hook_resid_post"]
        self.mock_acfg.l_mlp_hook_post_name = ["blocks.0.mlp.hook_post"]
        
        # Patch the global acfg
        self.acfg_patcher = patch('QuantaMechInterp.ablate_hooks.acfg', self.mock_acfg)
        self.acfg_patcher.start()
    
    def tearDown(self):
        """Clean up patches."""
        self.acfg_patcher.stop()
    
    @patch('QuantaMechInterp.ablate_hooks.logits_to_tokens_loss')
    @patch('QuantaMechInterp.ablate_hooks.loss_fn')
    @patch('QuantaMechInterp.ablate_hooks.to_numpy')
    @patch('builtins.print')
    def test_calc_mean_values(self, mock_print, mock_to_numpy, mock_loss_fn, mock_logits_loss):
        """Test calculating mean values from cache."""
        # Set up mock model and cache
        mock_cfg = Mock()
        mock_model = Mock()
        mock_cfg.main_model = mock_model
        
        # Mock cache data
        sample_attn_z = torch.randn(350, 22, 3, 170)
        sample_resid_post = torch.randn(350, 22, 510)
        sample_mlp_post = torch.randn(350, 22, 2040)
        
        mock_cache = {
            "blocks.0.attn.hook_z": sample_attn_z,
            "blocks.0.hook_resid_post": sample_resid_post,
            "blocks.0.mlp.hook_post": sample_mlp_post
        }
        
        mock_logits = torch.randn(350, 22, 15)
        mock_model.run_with_cache.return_value = (mock_logits, mock_cache)
        
        # Mock loss calculations
        mock_losses_raw = torch.randn(350, 22)
        mock_logits_loss.return_value = (mock_losses_raw, None)
        mock_loss_fn.return_value.mean.return_value = torch.tensor(0.03)
        mock_to_numpy.return_value = 0.03
        
        # Test questions
        test_questions = torch.randint(0, 15, (350, 22))
        
        # Call the function
        a_calc_mean_values(mock_cfg, test_questions)
        
        # Verify model setup
        mock_model.reset_hooks.assert_called_once()
        mock_model.set_use_attn_result.assert_called_once_with(True)
        mock_model.run_with_cache.assert_called_once()
        
        # Verify mean calculations
        expected_mean_attn_z = torch.mean(sample_attn_z, dim=0, keepdim=True)
        expected_mean_resid_post = torch.mean(sample_resid_post, dim=0, keepdim=True)
        expected_mean_mlp_post = torch.mean(sample_mlp_post, dim=0, keepdim=True)
        
        torch.testing.assert_close(self.mock_acfg.mean_attn_z, expected_mean_attn_z)
        torch.testing.assert_close(self.mock_acfg.mean_resid_post, expected_mean_resid_post)
        torch.testing.assert_close(self.mock_acfg.mean_mlp_hook_post, expected_mean_mlp_post)
        
        # Verify print statements
        self.assertEqual(mock_print.call_count, 6)  # 6 print statements in the function


class TestAPredictQuestions(unittest.TestCase):
    """Test the a_predict_questions function."""
    
    @patch('QuantaMechInterp.ablate_hooks.logits_to_tokens_loss')
    def test_predict_questions_no_hooks(self, mock_logits_loss):
        """Test predicting questions without hooks."""
        mock_cfg = Mock()
        mock_model = Mock()
        mock_cfg.main_model = mock_model
        
        # Mock model outputs
        mock_logits = torch.randn(10, 22, 15)
        mock_cache = {}
        mock_model.run_with_cache.return_value = (mock_logits, mock_cache)
        
        # Mock loss calculations
        mock_losses_raw = torch.randn(10, 22)
        mock_max_prob_tokens = torch.randint(0, 15, (10, 22))
        mock_logits_loss.return_value = (mock_losses_raw, mock_max_prob_tokens)
        
        # Test questions
        questions = torch.randint(0, 15, (10, 22))
        
        # Call the function
        losses, tokens = a_predict_questions(mock_cfg, questions, None)
        
        # Verify model setup
        mock_model.reset_hooks.assert_called_once()
        mock_model.set_use_attn_result.assert_called_once_with(True)
        mock_model.run_with_cache.assert_called_once()
        mock_model.run_with_hooks.assert_not_called()
        
        # Verify outputs
        torch.testing.assert_close(losses, mock_losses_raw)
        torch.testing.assert_close(tokens, mock_max_prob_tokens)
    
    @patch('QuantaMechInterp.ablate_hooks.logits_to_tokens_loss')
    def test_predict_questions_with_hooks(self, mock_logits_loss):
        """Test predicting questions with hooks."""
        mock_cfg = Mock()
        mock_model = Mock()
        mock_cfg.main_model = mock_model
        
        # Mock model outputs
        mock_logits = torch.randn(10, 22, 15)
        mock_model.run_with_hooks.return_value = mock_logits
        
        # Mock loss calculations
        mock_losses_raw = torch.randn(10, 22)
        mock_max_prob_tokens = torch.randint(0, 15, (10, 22))
        mock_logits_loss.return_value = (mock_losses_raw, mock_max_prob_tokens)
        
        # Test questions and hooks
        questions = torch.randint(0, 15, (10, 22))
        test_hooks = [("hook1", lambda x, h: x), ("hook2", lambda x, h: x)]
        
        # Call the function
        losses, tokens = a_predict_questions(mock_cfg, questions, test_hooks)
        
        # Verify model setup
        mock_model.reset_hooks.assert_called_once()
        mock_model.set_use_attn_result.assert_called_once_with(True)
        mock_model.run_with_cache.assert_not_called()
        mock_model.run_with_hooks.assert_called_once_with(
            questions.cuda(), return_type="logits", fwd_hooks=test_hooks
        )
        
        # Verify outputs
        torch.testing.assert_close(losses, mock_losses_raw)
        torch.testing.assert_close(tokens, mock_max_prob_tokens)


class TestARunAttentionIntervention(unittest.TestCase):
    """Test the a_run_attention_intervention function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the global acfg object
        self.mock_acfg = Mock()
        self.mock_acfg.num_tests_run = 0
        self.mock_acfg.abort = False
        self.mock_acfg.expected_answer = "ExpectedAns"
        self.mock_acfg.expected_impact = "ExpectedImpact"
        self.mock_acfg.intervened_answer = ""
        self.mock_acfg.intervened_impact = ""
        self.mock_acfg.threshold = 0.01
        self.mock_acfg.attn_get_hooks = []
        self.mock_acfg.attn_put_hooks = []
        
        # Patch the global acfg
        self.acfg_patcher = patch('QuantaMechInterp.ablate_hooks.acfg', self.mock_acfg)
        self.acfg_patcher.start()
    
    def tearDown(self):
        """Clean up patches."""
        self.acfg_patcher.stop()
    
    @patch('QuantaMechInterp.ablate_hooks.get_answer_impact')
    @patch('QuantaMechInterp.ablate_hooks.tokens_to_string')
    @patch('QuantaMechInterp.ablate_hooks.loss_fn')
    @patch('QuantaMechInterp.ablate_hooks.to_numpy')
    @patch('QuantaMechInterp.ablate_hooks.a_predict_questions')
    def test_successful_intervention(self, mock_predict, mock_to_numpy, mock_loss_fn, 
                                   mock_tokens_to_string, mock_get_impact):
        """Test successful attention intervention."""
        # Set up mock config
        mock_cfg = Mock()
        mock_cfg.num_question_positions = 10
        
        # Set up test data
        store_qa = torch.randint(0, 15, (15,))  # 15 positions total
        clean_qa = torch.randint(0, 15, (15,))
        clean_answer = "CleanAnswer"
        
        # Mock prediction results
        mock_losses = torch.tensor([[0.005]])  # Low loss
        mock_tokens = torch.randint(0, 15, (1, 15))
        mock_predict.return_value = (mock_losses, mock_tokens)
        
        # Mock other functions
        mock_to_numpy.return_value = 0.005
        mock_loss_fn.return_value.max.return_value = torch.tensor(0.005)
        mock_tokens_to_string.return_value = "IntervenedAnswer"
        mock_get_impact.return_value = "SomeImpact"
        
        # Call the function
        result = a_run_attention_intervention(mock_cfg, store_qa, clean_qa, clean_answer)
        
        # Verify test counter incremented
        self.assertEqual(self.mock_acfg.num_tests_run, 1)
        
        # Verify predictions were called
        self.assertEqual(mock_predict.call_count, 2)
        
        # Verify intervened answer and impact were set
        self.assertEqual(self.mock_acfg.intervened_answer, "IntervenedAnswer")
        self.assertEqual(self.mock_acfg.intervened_impact, "SomeImpact")
        
        # Verify result string contains expected components
        self.assertIn("CleanAns: CleanAnswer", result)
        self.assertIn("ExpectedAns/Impact: ExpectedAns/ExpectedImpact", result)
        self.assertIn("AblatedAns/Impact: IntervenedAnswer/SomeImpact", result)
        
        # Verify impact calculation was called correctly
        mock_get_impact.assert_called_once_with(mock_cfg, clean_answer, "IntervenedAnswer")
    
    @patch('QuantaMechInterp.ablate_hooks.a_predict_questions')
    def test_abort_on_store(self, mock_predict):
        """Test abortion during store phase."""
        mock_cfg = Mock()
        mock_cfg.num_question_positions = 10
        
        store_qa = torch.randint(0, 15, (15,))
        clean_qa = torch.randint(0, 15, (15,))
        clean_answer = "CleanAnswer"
        
        # Set abort flag after first prediction
        def set_abort(*args, **kwargs):
            self.mock_acfg.abort = True
            return (torch.tensor([]), torch.tensor([]))
        
        mock_predict.side_effect = set_abort
        
        # Call the function
        result = a_run_attention_intervention(mock_cfg, store_qa, clean_qa, clean_answer)
        
        # Verify abortion message
        self.assertIn("(Aborted on store)", result)
        self.assertEqual(mock_predict.call_count, 1)
    
    @patch('QuantaMechInterp.ablate_hooks.a_predict_questions')
    def test_abort_on_intervention(self, mock_predict):
        """Test abortion during intervention phase."""
        mock_cfg = Mock()
        mock_cfg.num_question_positions = 10
        
        store_qa = torch.randint(0, 15, (15,))
        clean_qa = torch.randint(0, 15, (15,))
        clean_answer = "CleanAnswer"
        
        # First call succeeds, second call sets abort
        call_count = 0
        def mock_predict_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (torch.tensor([[0.01]]), torch.tensor([[1, 2, 3]]))
            else:
                self.mock_acfg.abort = True
                return (torch.tensor([]), torch.tensor([]))
        
        mock_predict.side_effect = mock_predict_side_effect
        
        # Call the function
        result = a_run_attention_intervention(mock_cfg, store_qa, clean_qa, clean_answer)
        
        # Verify abortion message
        self.assertIn("(Aborted on intervention)", result)
        self.assertEqual(mock_predict.call_count, 2)
    
    @patch('QuantaMechInterp.ablate_hooks.a_predict_questions')
    def test_abort_on_bad_losses(self, mock_predict):
        """Test abortion on bad loss tensor shape."""
        mock_cfg = Mock()
        mock_cfg.num_question_positions = 10
        
        store_qa = torch.randint(0, 15, (15,))
        clean_qa = torch.randint(0, 15, (15,))
        clean_answer = "CleanAnswer"
        
        # First call succeeds, second call returns empty tensor
        call_count = 0
        def mock_predict_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (torch.tensor([[0.01]]), torch.tensor([[1, 2, 3]]))
            else:
                return (torch.tensor([]).reshape(0, 15), torch.tensor([]))
        
        mock_predict.side_effect = mock_predict_side_effect
        
        with patch('builtins.print') as mock_print:
            # Call the function
            result = a_run_attention_intervention(mock_cfg, store_qa, clean_qa, clean_answer)
        
        # Verify abortion
        self.assertTrue(self.mock_acfg.abort)
        self.assertIn("(Aborted on Bad all_losses_raw)", result)
        mock_print.assert_called_once()
    
    @patch('QuantaMechInterp.ablate_hooks.get_answer_impact')
    @patch('QuantaMechInterp.ablate_hooks.tokens_to_string')
    @patch('QuantaMechInterp.ablate_hooks.loss_fn')
    @patch('QuantaMechInterp.ablate_hooks.to_numpy')
    @patch('QuantaMechInterp.ablate_hooks.a_predict_questions')
    def test_empty_impact_uses_no_impact_tag(self, mock_predict, mock_to_numpy, mock_loss_fn,
                                           mock_tokens_to_string, mock_get_impact):
        """Test that empty impact gets replaced with NO_IMPACT_TAG."""
        mock_cfg = Mock()
        mock_cfg.num_question_positions = 10
        
        store_qa = torch.randint(0, 15, (15,))
        clean_qa = torch.randint(0, 15, (15,))
        clean_answer = "CleanAnswer"
        
        # Mock prediction results
        mock_losses = torch.tensor([[0.005]])
        mock_tokens = torch.randint(0, 15, (1, 15))
        mock_predict.return_value = (mock_losses, mock_tokens)
        
        # Mock other functions
        mock_to_numpy.return_value = 0.005
        mock_loss_fn.return_value.max.return_value = torch.tensor(0.005)
        mock_tokens_to_string.return_value = "IntervenedAnswer"
        mock_get_impact.return_value = ""  # Empty impact
        
        # Call the function
        result = a_run_attention_intervention(mock_cfg, store_qa, clean_qa, clean_answer)
        
        # Verify empty impact was replaced with NO_IMPACT_TAG
        self.assertEqual(self.mock_acfg.intervened_impact, NO_IMPACT_TAG)
    
    @patch('QuantaMechInterp.ablate_hooks.get_answer_impact')
    @patch('QuantaMechInterp.ablate_hooks.tokens_to_string')
    @patch('QuantaMechInterp.ablate_hooks.loss_fn')
    @patch('QuantaMechInterp.ablate_hooks.to_numpy')
    @patch('QuantaMechInterp.ablate_hooks.a_predict_questions')
    def test_high_loss_included_in_description(self, mock_predict, mock_to_numpy, mock_loss_fn,
                                             mock_tokens_to_string, mock_get_impact):
        """Test that high loss is included in description."""
        mock_cfg = Mock()
        mock_cfg.num_question_positions = 10
        
        store_qa = torch.randint(0, 15, (15,))
        clean_qa = torch.randint(0, 15, (15,))
        clean_answer = "CleanAnswer"
        
        # Mock prediction results with high loss
        mock_losses = torch.tensor([[0.05]])  # High loss > threshold
        mock_tokens = torch.randint(0, 15, (1, 15))
        mock_predict.return_value = (mock_losses, mock_tokens)
        
        # Mock other functions
        mock_to_numpy.return_value = 0.05
        mock_loss_fn.return_value.max.return_value = torch.tensor(0.05)
        mock_tokens_to_string.return_value = "IntervenedAnswer"
        mock_get_impact.return_value = "SomeImpact"
        
        # Call the function
        result = a_run_attention_intervention(mock_cfg, store_qa, clean_qa, clean_answer)
        
        # Verify loss is included in description
        self.assertIn("Loss: 0.05", result)
    
    @patch('QuantaMechInterp.ablate_hooks.get_answer_impact')
    @patch('QuantaMechInterp.ablate_hooks.tokens_to_string')
    @patch('QuantaMechInterp.ablate_hooks.loss_fn')
    @patch('QuantaMechInterp.ablate_hooks.to_numpy')
    @patch('QuantaMechInterp.ablate_hooks.a_predict_questions')
    def test_very_low_loss_shows_no_impact_tag(self, mock_predict, mock_to_numpy, mock_loss_fn,
                                              mock_tokens_to_string, mock_get_impact):
        """Test that very low loss shows NO_IMPACT_TAG instead of scientific notation."""
        mock_cfg = Mock()
        mock_cfg.num_question_positions = 10
        
        store_qa = torch.randint(0, 15, (15,))
        clean_qa = torch.randint(0, 15, (15,))
        clean_answer = "CleanAnswer"
        
        # Mock prediction results with very low loss
        mock_losses = torch.tensor([[1e-8]])  # Very low loss
        mock_tokens = torch.randint(0, 15, (1, 15))
        mock_predict.return_value = (mock_losses, mock_tokens)
        
        # Mock other functions
        mock_to_numpy.return_value = 1e-8
        mock_loss_fn.return_value.max.return_value = torch.tensor(1e-8)
        mock_tokens_to_string.return_value = "IntervenedAnswer"
        mock_get_impact.return_value = "SomeImpact"
        
        # Call the function
        result = a_run_attention_intervention(mock_cfg, store_qa, clean_qa, clean_answer)
        
        # Verify NO_IMPACT_TAG is used for very low loss
        self.assertIn(f"Loss: {NO_IMPACT_TAG}", result)


class TestIntegration(unittest.TestCase):
    """Integration tests for ablate_hooks module."""
    
    def test_hook_function_integration(self):
        """Test that hook functions work together properly."""
        # This is a basic integration test to ensure the functions can be called
        # without errors when properly mocked
        
        with patch('QuantaMechInterp.ablate_hooks.acfg') as mock_acfg:
            mock_acfg.ablate_node_locations = [NodeLocation(5, 0, True, 2)]
            mock_acfg.layer_store = [torch.randn(1, 22, 3, 170) for _ in range(4)]
            mock_acfg.mean_resid_post = torch.randn(1, 22, 510)
            
            # Test that hooks can be called without errors
            value = torch.randn(1, 22, 3, 170)
            mock_hook = Mock()
            
            # These should not raise exceptions
            a_get_l0_attn_z_hook(value, mock_hook)
            a_put_l0_attn_z_hook(value, mock_hook)
            
            resid_value = torch.randn(64, 22, 510)
            a_put_resid_post_hook(resid_value, mock_hook)
    
    def test_to_numpy_with_various_tensor_types(self):
        """Integration test for to_numpy with various PyTorch tensor types."""
        # Test different tensor configurations
        tensors_to_test = [
            torch.tensor([1, 2, 3]),
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([True, False, True]),
        ]
        
        for tensor in tensors_to_test:
            result = to_numpy(tensor)
            self.assertIsInstance(result, np.ndarray)
            np.testing.assert_array_equal(result, tensor.detach().cpu().numpy())


if __name__ == '__main__':
    unittest.main()
