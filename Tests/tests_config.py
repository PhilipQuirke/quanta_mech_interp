import unittest
import json
import numpy as np
import torch as th

from QuantaMechInterp.model_config import ModelConfig
from QuantaMechInterp.useful_config import UsefulConfig
from QuantaMechInterp.algo_config import AlgoConfig
from QuantaMechInterp.ablate_config import AblateConfig
from QuantaMechInterp.useful_node import NodeLocation, UsefulNode
from QuantaMechInterp.quanta_constants import NO_IMPACT_TAG


class TestModelConfig(unittest.TestCase):
    
    def setUp(self):
        self.config = ModelConfig()
    
    def test_initialization(self):
        """Test that ModelConfig initializes with correct default values"""
        self.assertEqual(self.config.hf_repo, "")
        self.assertEqual(self.config.model_name, "")
        self.assertIsNone(self.config.main_model)
        self.assertEqual(self.config.n_layers, 3)
        self.assertEqual(self.config.n_heads, 4)
        self.assertEqual(self.config.d_vocab, 15)
        self.assertEqual(self.config.d_model, 510)
        self.assertEqual(self.config.d_mlp_multiplier, 4)
        self.assertEqual(self.config.d_head, 170)
        self.assertEqual(self.config.act_fn, 'relu')
        self.assertFalse(self.config.grokfast)
        self.assertEqual(self.config.grokfast_alpha, 0.98)
        self.assertEqual(self.config.grokfast_lamb, 2.0)
        self.assertEqual(self.config.batch_size, 512)
        self.assertEqual(self.config.n_training_steps, 15000)
        self.assertEqual(self.config.weight_decay, 0.1)
        self.assertEqual(self.config.lr, 0.00008)
        self.assertEqual(self.config.insert_mode, 0)
        self.assertFalse(self.config.insert_late)
        self.assertEqual(self.config.training_seed, 372001)
        self.assertEqual(self.config.analysis_seed, 673023)
        self.assertEqual(self.config.avg_final_loss, 0.0)
        self.assertEqual(self.config.final_loss, 0.0)
        self.assertTrue(self.config.use_cuda)
        self.assertEqual(self.config.graph_file_suffix, "pdf")
        self.assertEqual(self.config.graph_file_prefix, "")
        
    def test_initialize_token_positions(self):
        """Test token position initialization"""
        self.config.initialize_token_positions(10, 5, True)
        self.assertEqual(self.config.num_question_positions, 10)
        self.assertEqual(self.config.num_answer_positions, 5)
        self.assertTrue(self.config.answer_meanings_ascend)
        self.assertEqual(len(self.config.token_position_meanings), 15)
        
        # Test with descending answer meanings
        self.config.initialize_token_positions(8, 6, False)
        self.assertEqual(self.config.num_question_positions, 8)
        self.assertEqual(self.config.num_answer_positions, 6)
        self.assertFalse(self.config.answer_meanings_ascend)
        self.assertEqual(len(self.config.token_position_meanings), 14)
    
    def test_properties(self):
        """Test computed properties"""
        self.config.initialize_token_positions(10, 5, True)
        self.assertEqual(self.config.n_ctx, 15)
        self.assertEqual(self.config.d_mlp, 2040)  # 4 * 510
        
    def test_set_seed(self):
        """Test seed setting functionality"""
        test_seed = 12345
        self.config.set_seed(test_seed)
        # Verify that numpy and torch seeds are set (we can't directly test this,
        # but we can ensure the method runs without error)
        self.assertTrue(True)  # If we get here, no exception was raised
        
    def test_parse_model_name(self):
        """Test model name parsing"""
        # Test basic parsing
        self.config.model_name = "test_l5_h8_t25K_s123456"
        self.config.parse_model_name()
        self.assertEqual(self.config.n_layers, 5)
        self.assertEqual(self.config.n_heads, 8)
        self.assertEqual(self.config.n_training_steps, 25000)
        self.assertEqual(self.config.training_seed, 123456)
        self.assertFalse(self.config.grokfast)
        self.assertEqual(self.config.insert_mode, 0)
        
        # Test with grokfast
        self.config.model_name = "test_l3_h4_t15K_gf_s654321"
        self.config.parse_model_name()
        self.assertTrue(self.config.grokfast)
        
        # Test insert modes
        self.config.model_name = "ins1_test_l2_h6_t10K_s111111"
        self.config.parse_model_name()
        self.assertEqual(self.config.insert_mode, 1)
        
        self.config.model_name = "ins2_test_l2_h6_t10K_s111111"
        self.config.parse_model_name()
        self.assertEqual(self.config.insert_mode, 2)
        
        self.config.model_name = "ins3_test_l2_h6_t10K_s111111"
        self.config.parse_model_name()
        self.assertEqual(self.config.insert_mode, 3)
        
        self.config.model_name = "ins4_test_l2_h6_t10K_s111111"
        self.config.parse_model_name()
        self.assertEqual(self.config.insert_mode, 4)
        
    def test_parse_insert_model_name(self):
        """Test insert model name parsing"""
        self.config.insert_model_name = "insert_l2_h3_t5K_s987654"
        self.config.parse_insert_model_name()
        self.assertEqual(self.config.insert_n_layers, 2)
        self.assertEqual(self.config.insert_n_heads, 3)
        self.assertEqual(self.config.insert_n_training_steps, 5000)
        self.assertEqual(self.config.insert_training_seed, 987654)
        
    def test_set_model_names(self):
        """Test setting model names"""
        # Test single model name
        self.config.set_model_names("test_l4_h6_t20K_s555555")
        self.assertEqual(self.config.model_name, "test_l4_h6_t20K_s555555")
        self.assertEqual(self.config.n_layers, 4)
        self.assertEqual(self.config.n_heads, 6)
        
        # Test with insert model name
        self.config.set_model_names("main_l3_h4_t15K_s111111,insert_l2_h2_t10K_s222222")
        self.assertEqual(self.config.model_name, "main_l3_h4_t15K_s111111")
        self.assertEqual(self.config.insert_model_name, "insert_l2_h2_t10K_s222222")
        self.assertEqual(self.config.insert_n_layers, 2)
        self.assertEqual(self.config.insert_n_heads, 2)
        
    def test_config_descriptions(self):
        """Test configuration description properties"""
        self.config.n_layers = 3
        self.config.n_heads = 4
        self.config.n_training_steps = 15000
        self.config.training_seed = 372001
        self.config.grokfast = False
        self.config.insert_mode = 0
        
        self.assertEqual(self.config.short_config_description, "_l3_h4")
        self.assertEqual(self.config.long_config_description, "_l3_h4_t15K_s372001")
        self.assertEqual(self.config.insert_config_description, "")
        
        # Test with grokfast
        self.config.grokfast = True
        self.assertEqual(self.config.long_config_description, "_l3_h4_t15K_gf_s372001")
        
        # Test with insert mode
        self.config.insert_mode = 2
        self.assertEqual(self.config.insert_config_description, "ins2_")
        
    def test_to_dict(self):
        """Test dictionary conversion"""
        config_dict = self.config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn("hf_repo", config_dict)
        self.assertIn("model_name", config_dict)
        self.assertIn("n_layers", config_dict)
        self.assertIn("n_heads", config_dict)
        self.assertEqual(config_dict["n_layers"], self.config.n_layers)
        self.assertEqual(config_dict["n_heads"], self.config.n_heads)
        
    def test_init_from_json(self):
        """Test initialization from JSON data"""
        test_data = {
            "hf_repo": "test/repo",
            "model_name": "test_model",
            "n_layers": 5,
            "n_heads": 8,
            "d_vocab": 20,
            "batch_size": 256,
            "lr": 0.001,
            "training_seed": 999999
        }
        
        self.config.init_from_json(test_data)
        self.assertEqual(self.config.hf_repo, "test/repo")
        self.assertEqual(self.config.model_name, "test_model")
        self.assertEqual(self.config.n_layers, 5)
        self.assertEqual(self.config.n_heads, 8)
        self.assertEqual(self.config.d_vocab, 20)
        self.assertEqual(self.config.batch_size, 256)
        self.assertEqual(self.config.lr, 0.001)
        self.assertEqual(self.config.training_seed, 999999)
        
        # Test with missing keys (should use defaults)
        partial_data = {"n_layers": 7}
        original_heads = self.config.n_heads
        self.config.init_from_json(partial_data)
        self.assertEqual(self.config.n_layers, 7)
        self.assertEqual(self.config.n_heads, original_heads)  # Should remain unchanged
        
    def test_sanity_check(self):
        """Test sanity check validation"""
        # Should pass with default values
        self.config.sanity_check()
        
        # Test invalid values
        with self.assertRaises(AssertionError):
            self.config.n_layers = 0
            self.config.sanity_check()
            
        self.config.n_layers = 3  # Reset
        with self.assertRaises(AssertionError):
            self.config.lr = 0
            self.config.sanity_check()
            
        self.config.lr = 0.00008  # Reset
        with self.assertRaises(AssertionError):
            self.config.training_seed = 0
            self.config.sanity_check()


class TestUsefulConfig(unittest.TestCase):
    
    def setUp(self):
        self.config = UsefulConfig()
    
    def test_initialization(self):
        """Test that UsefulConfig initializes correctly"""
        # Should inherit from ModelConfig
        self.assertIsInstance(self.config, ModelConfig)
        self.assertEqual(self.config.n_layers, 3)
        self.assertEqual(self.config.n_heads, 4)
        
        # Should have useful-specific attributes
        self.assertEqual(len(self.config.useful_positions), 0)
        self.assertIsNotNone(self.config.useful_nodes)
        
    def test_reset_useful(self):
        """Test resetting useful data"""
        # Add some data first
        self.config.add_useful_position(5)
        node_location = NodeLocation(0, 1, True, 2)
        self.config.add_useful_node_tag(node_location, "test", "tag")
        
        # Reset and verify
        self.config.reset_useful()
        self.assertEqual(len(self.config.useful_positions), 0)
        self.assertEqual(len(self.config.useful_nodes.nodes), 0)
        
    def test_set_model_names(self):
        """Test that setting model names resets useful data"""
        # Add some useful data
        self.config.add_useful_position(3)
        
        # Set model names should reset useful data
        self.config.set_model_names("test_l2_h3_t10K_s123456")
        self.assertEqual(len(self.config.useful_positions), 0)
        self.assertEqual(self.config.n_layers, 2)
        self.assertEqual(self.config.n_heads, 3)
        
    def test_useful_position_properties(self):
        """Test min/max useful position properties"""
        # Empty case
        self.assertEqual(self.config.min_useful_position, -1)
        self.assertEqual(self.config.max_useful_position, -1)
        
        # Add positions
        self.config.add_useful_position(5)
        self.config.add_useful_position(2)
        self.config.add_useful_position(8)
        
        self.assertEqual(self.config.min_useful_position, 2)
        self.assertEqual(self.config.max_useful_position, 8)
        
    def test_add_useful_position(self):
        """Test adding useful positions"""
        self.config.add_useful_position(3)
        self.assertIn(3, self.config.useful_positions)
        self.assertEqual(len(self.config.useful_positions), 1)
        
        # Adding duplicate should not increase length
        self.config.add_useful_position(3)
        self.assertEqual(len(self.config.useful_positions), 1)
        
        # Adding different position should increase length
        self.config.add_useful_position(7)
        self.assertEqual(len(self.config.useful_positions), 2)
        self.assertIn(7, self.config.useful_positions)
        
    def test_add_useful_node_tag(self):
        """Test adding useful node tags"""
        node_location = NodeLocation(0, 1, True, 2)
        self.config.add_useful_node_tag(node_location, "major", "minor")
        
        # Verify node was added
        self.assertEqual(len(self.config.useful_nodes.nodes), 1)
        added_node = self.config.useful_nodes.nodes[0]
        self.assertEqual(added_node.position, 0)
        self.assertEqual(added_node.layer, 1)
        self.assertTrue(added_node.is_head)
        self.assertEqual(added_node.num, 2)
        self.assertTrue(added_node.contains_tag("major", "minor"))
        
    def test_add_useful_node_tag_validation(self):
        """Test validation in add_useful_node_tag"""
        # Test invalid position
        with self.assertRaises(AssertionError):
            node_location = NodeLocation(-1, 1, True, 2)
            self.config.add_useful_node_tag(node_location, "major", "minor")
            
        # Test invalid layer
        with self.assertRaises(AssertionError):
            node_location = NodeLocation(0, -1, True, 2)
            self.config.add_useful_node_tag(node_location, "major", "minor")
            
        # Test position >= n_ctx
        with self.assertRaises(AssertionError):
            node_location = NodeLocation(100, 1, True, 2)  # Way beyond n_ctx
            self.config.add_useful_node_tag(node_location, "major", "minor")
            
        # Test layer >= n_layers
        with self.assertRaises(AssertionError):
            node_location = NodeLocation(0, 10, True, 2)  # Beyond n_layers (3)
            self.config.add_useful_node_tag(node_location, "major", "minor")
            
        # Test head num >= n_heads
        with self.assertRaises(AssertionError):
            node_location = NodeLocation(0, 1, True, 10)  # Beyond n_heads (4)
            self.config.add_useful_node_tag(node_location, "major", "minor")


class TestAlgoConfig(unittest.TestCase):
    
    def setUp(self):
        self.config = AlgoConfig()
    
    def test_initialization(self):
        """Test that AlgoConfig initializes correctly"""
        # Should inherit from UsefulConfig
        self.assertIsInstance(self.config, UsefulConfig)
        self.assertIsInstance(self.config, ModelConfig)
        
        # Should have algo-specific attributes
        self.assertEqual(self.config.num_algo_valid_clauses, 0)
        self.assertEqual(self.config.num_algo_invalid_clauses, 0)
        
    def test_reset_algo(self):
        """Test resetting algorithm data"""
        # Set some values first
        self.config.num_algo_valid_clauses = 5
        self.config.num_algo_invalid_clauses = 3
        
        # Reset and verify
        self.config.reset_algo()
        self.assertEqual(self.config.num_algo_valid_clauses, 0)
        self.assertEqual(self.config.num_algo_invalid_clauses, 0)
        
    def test_set_model_names(self):
        """Test that setting model names resets algo data"""
        # Set some algo data
        self.config.num_algo_valid_clauses = 5
        
        # Set model names should reset algo data
        self.config.set_model_names("test_l2_h3_t10K_s123456")
        self.assertEqual(self.config.num_algo_valid_clauses, 0)
        self.assertEqual(self.config.num_algo_invalid_clauses, 0)
        
    def test_test_algo_logic(self):
        """Test algorithm logic testing"""
        # Test valid clause
        self.config.test_algo_logic("test_clause_valid", True)
        self.assertEqual(self.config.num_algo_valid_clauses, 1)
        self.assertEqual(self.config.num_algo_invalid_clauses, 0)
        
        # Test invalid clause
        self.config.test_algo_logic("test_clause_invalid", False)
        self.assertEqual(self.config.num_algo_valid_clauses, 1)
        self.assertEqual(self.config.num_algo_invalid_clauses, 1)
        
    def test_test_algo_clause(self):
        """Test algorithm clause testing with node filtering"""
        from QuantaMechInterp.quanta_filter import FilterTrue
        from QuantaMechInterp.useful_node import UsefulNodeList
        
        # Create a test node list
        node_list = UsefulNodeList()
        node_location = NodeLocation(0, 1, True, 2)
        node_list.add_node_tag(node_location, "test", "tag")
        
        # Test with filter that should match
        filter_true = FilterTrue()
        position = self.config.test_algo_clause(node_list, filter_true, True)
        self.assertEqual(position, 0)  # Should return the position of the matched node
        self.assertEqual(self.config.num_algo_valid_clauses, 1)
        
        # Test with empty node list (should fail)
        empty_list = UsefulNodeList()
        position = self.config.test_algo_clause(empty_list, filter_true, True)
        self.assertEqual(position, -1)
        self.assertEqual(self.config.num_algo_invalid_clauses, 1)


class TestAblateConfig(unittest.TestCase):
    
    def setUp(self):
        self.config = AblateConfig()
    
    def test_initialization(self):
        """Test that AblateConfig initializes correctly"""
        self.assertEqual(self.config.operation, 0)
        self.assertFalse(self.config.show_test_failures)
        self.assertFalse(self.config.show_test_successes)
        self.assertEqual(self.config.threshold, 0.01)
        self.assertEqual(self.config.num_varied_questions, 0)
        self.assertEqual(self.config.num_varied_successes, 0)
        self.assertEqual(self.config.expected_answer, "")
        self.assertEqual(self.config.expected_impact, NO_IMPACT_TAG)
        self.assertEqual(self.config.intervened_answer, "")
        self.assertEqual(self.config.intervened_impact, NO_IMPACT_TAG)
        self.assertEqual(self.config.ablate_description, "")
        self.assertFalse(self.config.abort)
        self.assertEqual(self.config.num_filtered_nodes, 0)
        self.assertEqual(self.config.num_tests_run, 0)
        self.assertEqual(self.config.num_tags_added, 0)
        
    def test_reset_ablate_layer_store(self):
        """Test resetting ablate layer store"""
        # Modify layer store
        self.config.layer_store[0] = [1, 2, 3]
        self.config.layer_store[1] = [4, 5, 6]
        
        # Reset and verify
        self.config.reset_ablate_layer_store()
        self.assertEqual(len(self.config.layer_store), 4)
        for layer in self.config.layer_store:
            self.assertEqual(len(layer), 0)
            
    def test_reset_ablate(self):
        """Test resetting ablate configuration"""
        # Set some values first
        self.config.threshold = 0.05
        self.config.num_varied_questions = 10
        self.config.ablate_node_locations = [NodeLocation(0, 1, True, 2)]
        
        # Reset and verify
        self.config.reset_ablate()
        self.assertEqual(self.config.threshold, 0.01)
        self.assertEqual(self.config.num_varied_questions, 0)
        self.assertEqual(len(self.config.ablate_node_locations), 0)
        self.assertEqual(len(self.config.l_attn_hook_z_name), 0)
        self.assertEqual(len(self.config.resid_put_hooks), 0)
        self.assertEqual(len(self.config.attn_get_hooks), 0)
        self.assertEqual(len(self.config.mean_attn_z), 0)
        
    def test_reset_intervention(self):
        """Test resetting intervention data"""
        # Set some values first
        self.config.expected_answer = "test_answer"
        self.config.expected_impact = "test_impact"
        self.config.intervened_answer = "intervened"
        self.config.abort = True
        
        # Reset with defaults
        self.config.reset_intervention()
        self.assertEqual(self.config.expected_answer, "")
        self.assertEqual(self.config.expected_impact, NO_IMPACT_TAG)
        self.assertEqual(self.config.intervened_answer, "")
        self.assertEqual(self.config.intervened_impact, NO_IMPACT_TAG)
        self.assertFalse(self.config.abort)
        
        # Reset with custom values
        self.config.reset_intervention("custom_answer", "custom_impact")
        self.assertEqual(self.config.expected_answer, "custom_answer")
        self.assertEqual(self.config.expected_impact, "custom_impact")
        
    def test_reset_intervention_totals(self):
        """Test resetting intervention totals"""
        # Set some values first
        self.config.num_filtered_nodes = 5
        self.config.num_tests_run = 10
        self.config.num_tags_added = 3
        
        # Reset and verify
        self.config.reset_intervention_totals()
        self.assertEqual(self.config.num_filtered_nodes, 0)
        self.assertEqual(self.config.num_tests_run, 0)
        self.assertEqual(self.config.num_tags_added, 0)
        
    def test_ablate_node_names_property(self):
        """Test ablate_node_names property"""
        # Empty case
        self.assertEqual(self.config.ablate_node_names, "")
        
        # Single node
        node1 = NodeLocation(0, 1, True, 2)
        self.config.ablate_node_locations = [node1]
        expected_name = node1.name()
        self.assertEqual(self.config.ablate_node_names, expected_name)
        
        # Multiple nodes
        node2 = NodeLocation(1, 2, False, 3)
        self.config.ablate_node_locations = [node1, node2]
        expected_names = f"{node1.name()}, {node2.name()}"
        self.assertEqual(self.config.ablate_node_names, expected_names)
        
    def test_print_prediction_success_rate(self):
        """Test prediction success rate printing"""
        # Test with no questions (should not crash)
        self.config.num_varied_questions = 0
        self.config.num_varied_successes = 0
        try:
            self.config.print_prediction_success_rate()
        except Exception as e:
            self.fail(f"print_prediction_success_rate raised {e} unexpectedly!")
            
        # Test with some questions
        self.config.num_varied_questions = 10
        self.config.num_varied_successes = 8
        try:
            self.config.print_prediction_success_rate()
        except Exception as e:
            self.fail(f"print_prediction_success_rate raised {e} unexpectedly!")
            
        # Test with perfect success rate
        self.config.num_varied_questions = 5
        self.config.num_varied_successes = 5
        try:
            self.config.print_prediction_success_rate()
        except Exception as e:
            self.fail(f"print_prediction_success_rate raised {e} unexpectedly!")


if __name__ == '__main__':
    unittest.main()
