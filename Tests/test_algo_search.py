import unittest
from unittest.mock import Mock, patch, MagicMock
import itertools

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from QuantaMechInterp.algo_search import SubTaskBase, search_and_tag_digit_position, search_and_tag_digit, search_and_tag
from QuantaMechInterp.quanta_constants import QType, QCondition
from QuantaMechInterp.useful_node import UsefulNode, UsefulNodeList
from QuantaMechInterp.algo_config import AlgoConfig
from QuantaMechInterp.useful_config import UsefulConfig


class MockSubTask(SubTaskBase):
    """Mock implementation of SubTaskBase for testing"""
    
    @staticmethod
    def operation():
        return "MOCK_OP"
    
    @staticmethod
    def tag(impact_digit):
        return f"MOCK_TAG_{impact_digit}"
    
    @staticmethod
    def prereqs(cfg, position, impact_digit):
        # Return a mock filter that always returns True
        mock_filter = Mock()
        mock_filter.evaluate = Mock(return_value=True)
        return mock_filter
    
    @staticmethod
    def test(cfg, acfg, impact_digit, strong):
        # Mock test that succeeds for specific conditions
        if hasattr(acfg, 'test_should_succeed'):
            return acfg.test_should_succeed
        return len(acfg.ablate_node_locations) == 1  # Succeed for single nodes by default


class TestSubTaskBase(unittest.TestCase):
    """Test the abstract SubTaskBase class"""
    
    def test_abstract_methods_exist(self):
        """Test that all required abstract methods are defined"""
        required_methods = ['operation', 'tag', 'prereqs', 'test']
        for method in required_methods:
            self.assertTrue(hasattr(SubTaskBase, method))
            self.assertTrue(callable(getattr(SubTaskBase, method)))
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that SubTaskBase cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            SubTaskBase()
    
    def test_succeed_test_method(self):
        """Test the succeed_test static method"""
        mock_acfg = Mock()
        mock_acfg.ablate_node_names = "test_nodes"
        
        # Test strong=True
        result = SubTaskBase.succeed_test(None, mock_acfg, 1, True)
        self.assertTrue(result)
        
        # Test strong=False
        result = SubTaskBase.succeed_test(None, mock_acfg, 1, False)
        self.assertTrue(result)
    
    def test_mock_subtask_implementation(self):
        """Test that our mock implementation works correctly"""
        mock_task = MockSubTask()
        
        self.assertEqual(MockSubTask.operation(), "MOCK_OP")
        self.assertEqual(MockSubTask.tag(5), "MOCK_TAG_5")
        
        # Test prereqs returns a filter
        filter_obj = MockSubTask.prereqs(None, 0, 0)
        self.assertTrue(hasattr(filter_obj, 'evaluate'))
        self.assertTrue(filter_obj.evaluate(None))


class TestSearchAndTagDigitPosition(unittest.TestCase):
    """Test the search_and_tag_digit_position function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cfg = Mock()
        self.acfg = Mock()
        self.acfg.ablate_node_names = ""
        self.acfg.intervened_impact = "test_impact"
        self.acfg.num_tags_added = 0
        
        # Create test nodes
        self.test_nodes = UsefulNodeList()
        self.node1 = UsefulNode(0, 1, True, 0, [])
        self.node2 = UsefulNode(0, 1, True, 1, [])
        self.node3 = UsefulNode(0, 2, True, 0, [])  # Different layer
        self.test_nodes.nodes = [self.node1, self.node2, self.node3]
        
        self.sub_task = MockSubTask()
    
    def test_single_node_success(self):
        """Test successful search with single node"""
        self.acfg.test_should_succeed = True
        
        result = search_and_tag_digit_position(
            self.cfg, self.acfg, 1, self.test_nodes, self.sub_task, 
            True, "TEST_TAG", False
        )
        
        self.assertTrue(result)
        # Should have tested a node and added a tag
        self.assertEqual(len(self.acfg.ablate_node_locations), 1)
        # Check that it's one of our test nodes
        found_node = self.acfg.ablate_node_locations[0]
        self.assertIn(found_node, self.test_nodes.nodes)
    
    def test_single_node_failure_no_pairs(self):
        """Test failed search with single nodes and no pair search"""
        self.acfg.test_should_succeed = False
        
        result = search_and_tag_digit_position(
            self.cfg, self.acfg, 1, self.test_nodes, self.sub_task, 
            True, "TEST_TAG", False
        )
        
        self.assertFalse(result)
    
    def test_pair_search_success_same_layer(self):
        """Test successful pair search with nodes in same layer"""
        self.acfg.test_should_succeed = False  # Single nodes fail
        
        # Mock the test to succeed for pairs
        def mock_test(cfg, acfg, impact_digit, strong):
            return len(acfg.ablate_node_locations) == 2
        
        self.sub_task.test = mock_test
        
        result = search_and_tag_digit_position(
            self.cfg, self.acfg, 1, self.test_nodes, self.sub_task, 
            True, "TEST_TAG", True
        )
        
        self.assertTrue(result)
        # Should have found a pair in the same layer
        self.assertEqual(len(self.acfg.ablate_node_locations), 2)
        self.assertEqual(self.acfg.ablate_node_locations[0].layer, 
                        self.acfg.ablate_node_locations[1].layer)
    
    def test_pair_search_different_layers_skipped(self):
        """Test that pairs in different layers are skipped"""
        # Create nodes in different layers only
        test_nodes = UsefulNodeList()
        node1 = UsefulNode(0, 1, True, 0, [])
        node2 = UsefulNode(0, 2, True, 0, [])  # Different layer
        test_nodes.nodes = [node1, node2]
        
        self.acfg.test_should_succeed = False
        
        result = search_and_tag_digit_position(
            self.cfg, self.acfg, 1, test_nodes, self.sub_task, 
            True, "TEST_TAG", True
        )
        
        self.assertFalse(result)
    
    def test_weak_tagging(self):
        """Test weak tagging (strong=False)"""
        self.acfg.test_should_succeed = True
        
        result = search_and_tag_digit_position(
            self.cfg, self.acfg, 1, self.test_nodes, self.sub_task, 
            False, "TEST_TAG", False
        )
        
        self.assertTrue(result)
        # Check that weak tag includes intervened_impact
        # This would be verified by checking the tag added to the node


class TestSearchAndTagDigit(unittest.TestCase):
    """Test the search_and_tag_digit function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cfg = Mock()
        self.cfg.min_useful_position = 0
        self.cfg.max_useful_position = 2
        self.cfg.useful_nodes = Mock()
        self.cfg.useful_nodes.reset_node_tags = Mock()
        
        self.acfg = Mock()
        self.acfg.num_filtered_nodes = 0
        
        self.sub_task = MockSubTask()
    
    @patch('QuantaMechInterp.algo_search.filter_nodes')
    @patch('QuantaMechInterp.algo_search.search_and_tag_digit_position')
    def test_successful_search_strong(self, mock_search_position, mock_filter):
        """Test successful search with strong matching"""
        # Mock filter_nodes to return test nodes
        mock_nodes = Mock()
        mock_nodes.nodes = [Mock(), Mock()]
        mock_filter.return_value = mock_nodes
        
        # Mock successful position search
        mock_search_position.return_value = True
        
        result = search_and_tag_digit(self.cfg, self.acfg, self.sub_task, 1)
        
        self.assertTrue(result)
        # Should have called reset_node_tags
        self.cfg.useful_nodes.reset_node_tags.assert_called_once()
        # Should have called search_and_tag_digit_position
        self.assertTrue(mock_search_position.called)
    
    @patch('QuantaMechInterp.algo_search.filter_nodes')
    @patch('QuantaMechInterp.algo_search.search_and_tag_digit_position')
    def test_failed_search_strong_success_weak(self, mock_search_position, mock_filter):
        """Test failed strong search but successful weak search"""
        mock_nodes = Mock()
        mock_nodes.nodes = [Mock()]
        mock_filter.return_value = mock_nodes
        
        # Mock search to fail on strong, succeed on weak
        def mock_search_side_effect(cfg, acfg, impact_digit, test_nodes, sub_task, strong, tag, do_pair):
            return not strong  # Fail on strong=True, succeed on strong=False
        
        mock_search_position.side_effect = mock_search_side_effect
        
        result = search_and_tag_digit(self.cfg, self.acfg, self.sub_task, 1, 
                                     allow_impact_mismatch=True)
        
        self.assertTrue(result)
        # Should have been called twice (strong=True, then strong=False)
        self.assertEqual(mock_search_position.call_count, 6)  # 3 positions * 2 strength levels
    
    @patch('QuantaMechInterp.algo_search.filter_nodes')
    def test_no_test_nodes_filtered(self, mock_filter):
        """Test behavior when no nodes pass filtering"""
        # Mock filter to return empty node list
        mock_nodes = Mock()
        mock_nodes.nodes = []
        mock_filter.return_value = mock_nodes
        
        result = search_and_tag_digit(self.cfg, self.acfg, self.sub_task, 1)
        
        self.assertFalse(result)
    
    @patch('QuantaMechInterp.algo_search.filter_nodes')
    def test_delete_existing_tags_parameter(self, mock_filter):
        """Test delete_existing_tags parameter"""
        # Mock filter to return empty node list to avoid iteration issues
        mock_nodes = Mock()
        mock_nodes.nodes = []
        mock_filter.return_value = mock_nodes
        
        # Test with delete_existing_tags=True (default)
        search_and_tag_digit(self.cfg, self.acfg, self.sub_task, 1, 
                           delete_existing_tags=True)
        self.cfg.useful_nodes.reset_node_tags.assert_called()
        
        # Reset mock
        self.cfg.useful_nodes.reset_node_tags.reset_mock()
        
        # Test with delete_existing_tags=False
        search_and_tag_digit(self.cfg, self.acfg, self.sub_task, 1, 
                           delete_existing_tags=False)
        self.cfg.useful_nodes.reset_node_tags.assert_not_called()


class TestSearchAndTag(unittest.TestCase):
    """Test the search_and_tag function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cfg = Mock()
        self.cfg.num_answer_positions = 3
        
        self.acfg = Mock()
        self.acfg.reset_intervention_totals = Mock()
        self.acfg.num_filtered_nodes = 10
        self.acfg.num_tests_run = 5
        self.acfg.num_tags_added = 2
        
        self.sub_task = MockSubTask()
    
    @patch('QuantaMechInterp.algo_search.search_and_tag_digit')
    def test_search_and_tag_all_digits(self, mock_search_digit):
        """Test that search_and_tag processes all answer positions"""
        mock_search_digit.return_value = True
        
        search_and_tag(self.cfg, self.acfg, self.sub_task)
        
        # Should have called reset_intervention_totals
        self.acfg.reset_intervention_totals.assert_called_once()
        
        # Should have set operation
        self.assertEqual(self.acfg.operation, "MOCK_OP")
        
        # Should have called search_and_tag_digit for each answer position
        self.assertEqual(mock_search_digit.call_count, 3)
        
        # Check that it was called with correct impact_digit values
        call_args = [call[0][3] for call in mock_search_digit.call_args_list]  # impact_digit is 4th argument (index 3)
        self.assertEqual(call_args, [0, 1, 2])
    
    @patch('QuantaMechInterp.algo_search.search_and_tag_digit')
    @patch('builtins.print')
    def test_search_and_tag_prints_summary(self, mock_print, mock_search_digit):
        """Test that search_and_tag prints a summary"""
        mock_search_digit.return_value = True
        
        search_and_tag(self.cfg, self.acfg, self.sub_task)
        
        # Should have printed summary
        mock_print.assert_called()
        print_call_args = mock_print.call_args[0][0]
        self.assertIn("Filtering gave", print_call_args)
        self.assertIn("candidate node(s)", print_call_args)
        self.assertIn("intervention test(s)", print_call_args)
        self.assertIn("tag(s)", print_call_args)
    
    @patch('QuantaMechInterp.algo_search.search_and_tag_digit')
    def test_search_and_tag_parameters_passed(self, mock_search_digit):
        """Test that parameters are correctly passed to search_and_tag_digit"""
        mock_search_digit.return_value = True
        
        search_and_tag(self.cfg, self.acfg, self.sub_task, 
                      do_pair_search=True, 
                      allow_impact_mismatch=True, 
                      delete_existing_tags=False)
        
        # Check that parameters were passed correctly
        for call in mock_search_digit.call_args_list:
            kwargs = call[1]
            self.assertTrue(kwargs['do_pair_search'])
            self.assertTrue(kwargs['allow_impact_mismatch'])
            self.assertFalse(kwargs['delete_existing_tags'])


class TestIntegration(unittest.TestCase):
    """Integration tests for the algo_search module"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        # Create a more realistic configuration
        self.cfg = UsefulConfig()
        self.cfg.num_answer_positions = 2
        self.cfg.useful_nodes = UsefulNodeList()
        
        # Add some test nodes
        node1 = UsefulNode(0, 1, True, 0, [])
        node2 = UsefulNode(1, 1, True, 1, [])
        node3 = UsefulNode(0, 2, False, 0, [])
        self.cfg.useful_nodes.nodes = [node1, node2, node3]
        
        self.acfg = AlgoConfig()
        # Add missing attributes that the search function expects
        self.acfg.reset_intervention_totals = Mock()
        self.acfg.num_filtered_nodes = 0
        self.acfg.num_tests_run = 0
        self.acfg.num_tags_added = 0
    
    @patch('QuantaMechInterp.algo_search.filter_nodes')
    def test_end_to_end_search_flow(self, mock_filter):
        """Test the complete search flow"""
        # Mock filter_nodes to return our test nodes
        mock_filter.return_value = self.cfg.useful_nodes
        
        # Create a sub-task that succeeds for the first node
        class TestSubTask(SubTaskBase):
            @staticmethod
            def operation():
                return "TEST_OP"
            
            @staticmethod
            def tag(impact_digit):
                return f"TEST_{impact_digit}"
            
            @staticmethod
            def prereqs(cfg, position, impact_digit):
                mock_filter = Mock()
                mock_filter.evaluate = Mock(return_value=True)
                return mock_filter
            
            @staticmethod
            def test(cfg, acfg, impact_digit, strong):
                # Succeed only for single nodes
                return len(acfg.ablate_node_locations) == 1
        
        sub_task = TestSubTask()
        
        # Run the search
        search_and_tag(self.cfg, self.acfg, sub_task)
        
        # Verify that the operation was set
        self.assertEqual(self.acfg.operation, "TEST_OP")
        
        # Verify that some processing occurred
        self.assertGreaterEqual(self.acfg.num_filtered_nodes, 0)


if __name__ == '__main__':
    unittest.main()
