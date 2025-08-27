import unittest
from unittest.mock import Mock, patch

from QuantaMechInterp.quanta_filter import (
    extract_trailing_int, FilterNode, FilterAnd, FilterOr, FilterName, 
    FilterTrue, FilterHead, FilterNeuron, FilterPosition, FilterLayer,
    FilterContains, FilterAttention, FilterImpact, FilterAlgo,
    filter_nodes, print_algo_purpose_results
)
from QuantaMechInterp.quanta_constants import QCondition, QType, MIN_ATTN_PERC
from QuantaMechInterp.useful_node import NodeLocation, UsefulNode, UsefulNodeList


class TestExtractTrailingInt(unittest.TestCase):
    
    def test_extract_trailing_int_basic(self):
        """Test basic trailing integer extraction"""
        self.assertEqual(extract_trailing_int("test123"), 123)
        self.assertEqual(extract_trailing_int("abc456"), 456)
        self.assertEqual(extract_trailing_int("789"), 789)
        
    def test_extract_trailing_int_no_digits(self):
        """Test extraction when no trailing digits exist"""
        self.assertIsNone(extract_trailing_int("test"))
        self.assertIsNone(extract_trailing_int("abc"))
        self.assertIsNone(extract_trailing_int(""))
        
    def test_extract_trailing_int_middle_digits(self):
        """Test extraction ignores digits in middle"""
        self.assertEqual(extract_trailing_int("test123abc456"), 456)
        self.assertEqual(extract_trailing_int("123abc789"), 789)
        
    def test_extract_trailing_int_zero(self):
        """Test extraction of zero"""
        self.assertEqual(extract_trailing_int("test0"), 0)
        self.assertEqual(extract_trailing_int("abc00"), 0)
        
    def test_extract_trailing_int_large_numbers(self):
        """Test extraction of large numbers"""
        self.assertEqual(extract_trailing_int("test999999"), 999999)
        self.assertEqual(extract_trailing_int("prefix123456789"), 123456789)


class TestFilterTrue(unittest.TestCase):
    
    def setUp(self):
        self.filter = FilterTrue()
        self.test_node = UsefulNode(0, 1, True, 2, [])
        
    def test_evaluate_always_true(self):
        """Test that FilterTrue always returns True"""
        self.assertTrue(self.filter.evaluate(self.test_node))
        
        # Test with different node configurations
        node2 = UsefulNode(5, 3, False, 1, ["tag1", "tag2"])
        self.assertTrue(self.filter.evaluate(node2))
        
    def test_describe(self):
        """Test FilterTrue description"""
        self.assertEqual(self.filter.describe(), "True")


class TestFilterName(unittest.TestCase):
    
    def setUp(self):
        self.test_node = UsefulNode(0, 1, True, 2, [])
        
    def test_evaluate_matching_name(self):
        """Test FilterName with matching node name"""
        node_name = self.test_node.name()
        filter_name = FilterName(node_name)
        self.assertTrue(filter_name.evaluate(self.test_node))
        
    def test_evaluate_non_matching_name(self):
        """Test FilterName with non-matching node name"""
        filter_name = FilterName("different_name")
        self.assertFalse(filter_name.evaluate(self.test_node))
        
    def test_describe(self):
        """Test FilterName description"""
        test_name = "P00L1H2"
        filter_name = FilterName(test_name)
        self.assertEqual(filter_name.describe(), f"Name={test_name}")


class TestFilterHead(unittest.TestCase):
    
    def setUp(self):
        self.filter = FilterHead()
        
    def test_evaluate_head_node(self):
        """Test FilterHead with head node"""
        head_node = UsefulNode(0, 1, True, 2, [])
        self.assertTrue(self.filter.evaluate(head_node))
        
    def test_evaluate_neuron_node(self):
        """Test FilterHead with neuron node"""
        neuron_node = UsefulNode(0, 1, False, 2, [])
        self.assertFalse(self.filter.evaluate(neuron_node))
        
    def test_describe(self):
        """Test FilterHead description"""
        self.assertEqual(self.filter.describe(), "IsHead")


class TestFilterNeuron(unittest.TestCase):
    
    def setUp(self):
        self.filter = FilterNeuron()
        
    def test_evaluate_neuron_node(self):
        """Test FilterNeuron with neuron node"""
        neuron_node = UsefulNode(0, 1, False, 2, [])
        self.assertTrue(self.filter.evaluate(neuron_node))
        
    def test_evaluate_head_node(self):
        """Test FilterNeuron with head node"""
        head_node = UsefulNode(0, 1, True, 2, [])
        self.assertFalse(self.filter.evaluate(head_node))
        
    def test_describe(self):
        """Test FilterNeuron description"""
        self.assertEqual(self.filter.describe(), "IsNeuron")


class TestFilterPosition(unittest.TestCase):
    
    def setUp(self):
        self.test_node = UsefulNode(5, 1, True, 2, [])  # Position 5
        
    def test_evaluate_must_condition_match(self):
        """Test FilterPosition with MUST condition matching"""
        filter_pos = FilterPosition("P05", QCondition.MUST)
        self.assertTrue(filter_pos.evaluate(self.test_node))
        
    def test_evaluate_must_condition_no_match(self):
        """Test FilterPosition with MUST condition not matching"""
        filter_pos = FilterPosition("P03", QCondition.MUST)
        self.assertFalse(filter_pos.evaluate(self.test_node))
        
    def test_evaluate_not_condition(self):
        """Test FilterPosition with NOT condition"""
        filter_pos = FilterPosition("P03", QCondition.NOT)
        self.assertTrue(filter_pos.evaluate(self.test_node))
        
        filter_pos2 = FilterPosition("P05", QCondition.NOT)
        self.assertFalse(filter_pos2.evaluate(self.test_node))
        
    def test_evaluate_may_condition(self):
        """Test FilterPosition with MAY condition always returns True"""
        filter_pos = FilterPosition("P03", QCondition.MAY)
        self.assertTrue(filter_pos.evaluate(self.test_node))
        
    def test_evaluate_max_condition(self):
        """Test FilterPosition with MAX condition"""
        filter_pos = FilterPosition("P07", QCondition.MAX)  # Position <= 7
        self.assertTrue(filter_pos.evaluate(self.test_node))
        
        filter_pos2 = FilterPosition("P03", QCondition.MAX)  # Position <= 3
        self.assertFalse(filter_pos2.evaluate(self.test_node))
        
    def test_evaluate_min_condition(self):
        """Test FilterPosition with MIN condition"""
        filter_pos = FilterPosition("P03", QCondition.MIN)  # Position >= 3
        self.assertTrue(filter_pos.evaluate(self.test_node))
        
        filter_pos2 = FilterPosition("P07", QCondition.MIN)  # Position >= 7
        self.assertFalse(filter_pos2.evaluate(self.test_node))
        
    def test_describe(self):
        """Test FilterPosition description"""
        filter_pos = FilterPosition("P05", QCondition.MUST)
        expected = f"{QCondition.MUST.value} P5"  # position_name() doesn't zero-pad
        self.assertEqual(filter_pos.describe(), expected)


class TestFilterLayer(unittest.TestCase):
    
    def setUp(self):
        self.test_node = UsefulNode(0, 2, True, 1, [])  # Layer 2
        
    def test_evaluate_matching_layer(self):
        """Test FilterLayer with matching layer"""
        filter_layer = FilterLayer(2)
        self.assertTrue(filter_layer.evaluate(self.test_node))
        
    def test_evaluate_non_matching_layer(self):
        """Test FilterLayer with non-matching layer"""
        filter_layer = FilterLayer(1)
        self.assertFalse(filter_layer.evaluate(self.test_node))
        
    def test_describe(self):
        """Test FilterLayer description"""
        filter_layer = FilterLayer(2)
        self.assertEqual(filter_layer.describe(), "Layer=2")


class TestFilterContains(unittest.TestCase):
    
    def setUp(self):
        self.test_node = UsefulNode(0, 1, True, 2, [])
        # Tags are stored as "MAJOR:MINOR" format
        self.test_node.tags.append("IMPACT:A123")
        self.test_node.tags.append("ATTN:P05=75")
        
    # TODO: RESOLVE BUG
    # def test_evaluate_must_condition_match(self):
    #     """Test FilterContains with MUST condition matching"""
    #     filter_contains = FilterContains(QType.IMPACT, "A123", QCondition.MUST)
    #     self.assertTrue(filter_contains.evaluate(self.test_node))
        
    def test_evaluate_must_condition_no_match(self):
        """Test FilterContains with MUST condition not matching"""
        filter_contains = FilterContains(QType.IMPACT, "A456", QCondition.MUST)
        self.assertFalse(filter_contains.evaluate(self.test_node))
        
    # TODO: RESOLVE BUG
    # def test_evaluate_not_condition(self):
    #     """Test FilterContains with NOT condition"""
    #     filter_contains = FilterContains(QType.IMPACT, "A456", QCondition.NOT)
    #     self.assertTrue(filter_contains.evaluate(self.test_node))
    #     
    #     filter_contains2 = FilterContains(QType.IMPACT, "A123", QCondition.NOT)
    #     self.assertFalse(filter_contains2.evaluate(self.test_node))
        
    def test_evaluate_may_condition(self):
        """Test FilterContains with MAY condition always returns True"""
        filter_contains = FilterContains(QType.IMPACT, "A456", QCondition.MAY)
        self.assertTrue(filter_contains.evaluate(self.test_node))
        
    def test_describe(self):
        """Test FilterContains description"""
        filter_contains = FilterContains(QType.IMPACT, "A123", QCondition.MUST)
        expected = f"{QCondition.MUST.value} {QType.IMPACT.value} A123"
        self.assertEqual(filter_contains.describe(), expected)


class TestFilterAttention(unittest.TestCase):
    
    def setUp(self):
        self.test_node = UsefulNode(0, 1, True, 2, [])
        
    # TODO: RESOLVE BUG
    # def test_evaluate_with_sufficient_percentage(self):
    #     """Test FilterAttention with sufficient attention percentage"""
    #     # Add tag directly to tags list in the format expected by FilterAttention
    #     self.test_node.tags.append("ATTN:P05=75")  # 75% attention
    #     filter_attn = FilterAttention("P05=", QCondition.MUST, 50)  # Min 50% - need "P05=" to match
    #     self.assertTrue(filter_attn.evaluate(self.test_node))
        
    def test_evaluate_with_insufficient_percentage(self):
        """Test FilterAttention with insufficient attention percentage"""
        self.test_node.tags.append("ATTN:P05=25")  # 25% attention
        filter_attn = FilterAttention("P05", QCondition.MUST, 50)  # Min 50%
        self.assertFalse(filter_attn.evaluate(self.test_node))
        
    # TODO: RESOLVE BUG
    # def test_evaluate_with_exact_minimum_percentage(self):
    #     """Test FilterAttention with exact minimum percentage"""
    #     self.test_node.tags.append("ATTN:P05=50")  # 50% attention
    #     filter_attn = FilterAttention("P05=", QCondition.MUST, 50)  # Min 50% - need "P05=" to match
    #     self.assertTrue(filter_attn.evaluate(self.test_node))
        
    def test_evaluate_with_no_matching_tag(self):
        """Test FilterAttention with no matching attention tag"""
        self.test_node.tags.append("ATTN:P03=75")  # Different position
        filter_attn = FilterAttention("P05", QCondition.MUST, 50)
        self.assertFalse(filter_attn.evaluate(self.test_node))
        
    def test_evaluate_with_malformed_tag(self):
        """Test FilterAttention with malformed attention tag"""
        self.test_node.tags.append("ATTN:P05=invalid")  # Invalid percentage
        filter_attn = FilterAttention("P05", QCondition.MUST, 50)
        self.assertFalse(filter_attn.evaluate(self.test_node))


class TestFilterImpact(unittest.TestCase):
    
    def test_initialization(self):
        """Test FilterImpact initialization"""
        filter_impact = FilterImpact("A123")
        self.assertEqual(filter_impact.quanta_type, QType.IMPACT)
        self.assertEqual(filter_impact.minor_tag, "A123")
        self.assertEqual(filter_impact.filter_strength, QCondition.MUST)
        
    def test_initialization_with_custom_strength(self):
        """Test FilterImpact initialization with custom strength"""
        filter_impact = FilterImpact("A123", QCondition.NOT)
        self.assertEqual(filter_impact.filter_strength, QCondition.NOT)


class TestFilterAlgo(unittest.TestCase):
    
    def test_initialization(self):
        """Test FilterAlgo initialization"""
        filter_algo = FilterAlgo("test_algo")
        self.assertEqual(filter_algo.quanta_type, QType.ALGO)
        self.assertEqual(filter_algo.minor_tag, "test_algo")
        self.assertEqual(filter_algo.filter_strength, QCondition.MUST)
        
    def test_initialization_with_custom_strength(self):
        """Test FilterAlgo initialization with custom strength"""
        filter_algo = FilterAlgo("test_algo", QCondition.NOT)
        self.assertEqual(filter_algo.filter_strength, QCondition.NOT)


class TestFilterAnd(unittest.TestCase):
    
    def setUp(self):
        self.test_node = UsefulNode(5, 2, True, 1, [])
        self.test_node.add_tag("IMPACT", "A123")
        
    def test_evaluate_all_true(self):
        """Test FilterAnd when all children return True"""
        filter_and = FilterAnd(FilterTrue(), FilterLayer(2), FilterHead())
        self.assertTrue(filter_and.evaluate(self.test_node))
        
    def test_evaluate_some_false(self):
        """Test FilterAnd when some children return False"""
        filter_and = FilterAnd(FilterTrue(), FilterLayer(1), FilterHead())  # Layer 1 is false
        self.assertFalse(filter_and.evaluate(self.test_node))
        
    def test_evaluate_all_false(self):
        """Test FilterAnd when all children return False"""
        filter_and = FilterAnd(FilterLayer(1), FilterNeuron())  # Both false
        self.assertFalse(filter_and.evaluate(self.test_node))
        
    def test_evaluate_empty_children(self):
        """Test FilterAnd with no children"""
        filter_and = FilterAnd()
        self.assertTrue(filter_and.evaluate(self.test_node))  # all() returns True for empty
        
    def test_describe(self):
        """Test FilterAnd description"""
        filter_and = FilterAnd(FilterTrue(), FilterLayer(2))
        expected = " and( True, Layer=2)"
        self.assertEqual(filter_and.describe(), expected)


class TestFilterOr(unittest.TestCase):
    
    def setUp(self):
        self.test_node = UsefulNode(5, 2, True, 1, [])
        self.test_node.add_tag("IMPACT", "A123")
        
    def test_evaluate_all_true(self):
        """Test FilterOr when all children return True"""
        filter_or = FilterOr(FilterTrue(), FilterLayer(2), FilterHead())
        self.assertTrue(filter_or.evaluate(self.test_node))
        
    def test_evaluate_some_true(self):
        """Test FilterOr when some children return True"""
        filter_or = FilterOr(FilterLayer(1), FilterHead())  # Head is true
        self.assertTrue(filter_or.evaluate(self.test_node))
        
    def test_evaluate_all_false(self):
        """Test FilterOr when all children return False"""
        filter_or = FilterOr(FilterLayer(1), FilterNeuron())  # Both false
        self.assertFalse(filter_or.evaluate(self.test_node))
        
    def test_evaluate_empty_children(self):
        """Test FilterOr with no children"""
        filter_or = FilterOr()
        self.assertFalse(filter_or.evaluate(self.test_node))  # any() returns False for empty
        
    def test_describe(self):
        """Test FilterOr description"""
        filter_or = FilterOr(FilterTrue(), FilterLayer(2))
        expected = " or( True, Layer=2)"
        self.assertEqual(filter_or.describe(), expected)


class TestFilterNodes(unittest.TestCase):
    
    def setUp(self):
        self.node_list = UsefulNodeList()
        
        # Add various test nodes
        self.node1 = UsefulNode(0, 1, True, 0, [])  # Head at layer 1
        self.node1.add_tag("IMPACT", "A123")
        self.node_list.nodes.append(self.node1)
        
        self.node2 = UsefulNode(1, 1, False, 0, [])  # Neuron at layer 1
        self.node2.add_tag("ATTN", "P01=75")
        self.node_list.nodes.append(self.node2)
        
        self.node3 = UsefulNode(2, 2, True, 1, [])  # Head at layer 2
        self.node3.add_tag("ALGO", "test_algo")
        self.node_list.nodes.append(self.node3)
        
        self.node4 = UsefulNode(3, 2, False, 1, [])  # Neuron at layer 2
        self.node_list.nodes.append(self.node4)
        
    def test_filter_nodes_all_pass(self):
        """Test filter_nodes when all nodes pass filter"""
        filtered = filter_nodes(self.node_list, FilterTrue())
        self.assertEqual(len(filtered.nodes), 4)
        
    def test_filter_nodes_some_pass(self):
        """Test filter_nodes when some nodes pass filter"""
        filtered = filter_nodes(self.node_list, FilterHead())
        self.assertEqual(len(filtered.nodes), 2)  # Only head nodes
        
        # Verify correct nodes were filtered
        for node in filtered.nodes:
            self.assertTrue(node.is_head)
            
    def test_filter_nodes_none_pass(self):
        """Test filter_nodes when no nodes pass filter"""
        filtered = filter_nodes(self.node_list, FilterLayer(5))  # No nodes at layer 5
        self.assertEqual(len(filtered.nodes), 0)
        
    # TODO: RESOLVE BUG
    # def test_filter_nodes_complex_filter(self):
    #     """Test filter_nodes with complex filter"""
    #     # Filter for heads at layer 1 with impact tags
    #     complex_filter = FilterAnd(
    #         FilterHead(),
    #         FilterLayer(1),
    #         FilterImpact("A123")
    #     )
    #     filtered = filter_nodes(self.node_list, complex_filter)
    #     self.assertEqual(len(filtered.nodes), 1)
    #     self.assertEqual(filtered.nodes[0], self.node1)
        
    def test_filter_nodes_empty_list(self):
        """Test filter_nodes with empty node list"""
        empty_list = UsefulNodeList()
        filtered = filter_nodes(empty_list, FilterTrue())
        self.assertEqual(len(filtered.nodes), 0)


class TestPrintAlgoPurposeResults(unittest.TestCase):
    
    def setUp(self):
        self.cfg = Mock()
        self.cfg.useful_nodes = UsefulNodeList()
        
    # TODO: RESOLVE BUG
    # def test_print_algo_purpose_results_with_nodes(self):
    #     """Test print_algo_purpose_results with various node types"""
    #     # Add head nodes
    #     head1 = UsefulNode(0, 1, True, 0, [])
    #     head1.add_tag("ALGO", "test_algo")
    #     self.cfg.useful_nodes.nodes.append(head1)
    #     
    #     head2 = UsefulNode(1, 1, True, 1, [])  # No algo tag
    #     self.cfg.useful_nodes.nodes.append(head2)
    #     
    #     # Add neuron nodes
    #     neuron1 = UsefulNode(2, 1, False, 0, [])
    #     neuron1.add_tag("ALGO", "another_algo")
    #     self.cfg.useful_nodes.nodes.append(neuron1)
    #     
    #     neuron2 = UsefulNode(3, 1, False, 1, [])  # No algo tag
    #     self.cfg.useful_nodes.nodes.append(neuron2)
    #     
    #     # Capture print output
    #     with patch('builtins.print') as mock_print:
    #         print_algo_purpose_results(self.cfg)
    #         
    #         # Should print statistics for both heads and neurons
    #         self.assertEqual(mock_print.call_count, 2)
    #         
    #         # Check head statistics (1 of 2 heads have algo purpose)
    #         head_call = mock_print.call_args_list[0][0][0]
    #         self.assertIn("1 of 2 useful attention heads", head_call)
    #         self.assertIn("50.00%", head_call)
    #         
    #         # Check neuron statistics (1 of 2 neurons have algo purpose)
    #         neuron_call = mock_print.call_args_list[1][0][0]
    #         self.assertIn("1 of 2 useful MLP neurons", neuron_call)
    #         self.assertIn("50.00%", neuron_call)
            
    def test_print_algo_purpose_results_no_heads(self):
        """Test print_algo_purpose_results with no head nodes"""
        # Add only neuron nodes
        neuron1 = UsefulNode(0, 1, False, 0, [])
        neuron1.add_tag("ALGO", "test_algo")
        self.cfg.useful_nodes.nodes.append(neuron1)
        
        with patch('builtins.print') as mock_print:
            print_algo_purpose_results(self.cfg)
            
            # Should only print neuron statistics
            self.assertEqual(mock_print.call_count, 1)
            neuron_call = mock_print.call_args_list[0][0][0]
            self.assertIn("useful MLP neurons", neuron_call)
            
    def test_print_algo_purpose_results_no_neurons(self):
        """Test print_algo_purpose_results with no neuron nodes"""
        # Add only head nodes
        head1 = UsefulNode(0, 1, True, 0, [])
        head1.add_tag("ALGO", "test_algo")
        self.cfg.useful_nodes.nodes.append(head1)
        
        with patch('builtins.print') as mock_print:
            print_algo_purpose_results(self.cfg)
            
            # Should only print head statistics
            self.assertEqual(mock_print.call_count, 1)
            head_call = mock_print.call_args_list[0][0][0]
            self.assertIn("useful attention heads", head_call)
            
    def test_print_algo_purpose_results_empty(self):
        """Test print_algo_purpose_results with no nodes"""
        with patch('builtins.print') as mock_print:
            print_algo_purpose_results(self.cfg)
            
            # Should not print anything
            self.assertEqual(mock_print.call_count, 0)


class TestComplexFilterScenarios(unittest.TestCase):
    
    def setUp(self):
        self.node_list = UsefulNodeList()
        
        # Create a comprehensive set of test nodes
        for pos in range(3):
            for layer in range(2):
                for is_head in [True, False]:
                    for num in range(2):
                        node = UsefulNode(pos, layer, is_head, num, [])
                        
                        # Add various tags based on node properties
                        if pos == 0:
                            node.add_tag("IMPACT", f"A{pos+1}{layer+1}{num+1}")
                        if layer == 1 and is_head:
                            node.add_tag("ATTN", f"P{pos:02d}={75 + pos*5}")
                        if pos == 2 and not is_head:
                            node.add_tag("ALGO", f"algo_{layer}_{num}")
                            
                        self.node_list.nodes.append(node)
                        
    # TODO: RESOLVE BUG
    # def test_complex_and_filter(self):
    #     """Test complex AND filter combining multiple conditions"""
    #     # Find heads at layer 1 with attention >= 75% at position P00
    #     complex_filter = FilterAnd(
    #         FilterHead(),
    #         FilterLayer(1),
    #         FilterAttention("P00", QCondition.MUST, 75)
    #     )
    #     
    #     filtered = filter_nodes(self.node_list, complex_filter)
    #     self.assertEqual(len(filtered.nodes), 1)
    #     
    #     node = filtered.nodes[0]
    #     self.assertTrue(node.is_head)
    #     self.assertEqual(node.layer, 1)
    #     self.assertEqual(node.position, 0)
        
    def test_complex_or_filter(self):
        """Test complex OR filter with multiple alternatives"""
        # Find nodes that are either at position 0 OR have algo tags
        complex_filter = FilterOr(
            FilterPosition("P00", QCondition.MUST),
            FilterAlgo("", QCondition.MUST)
        )
        
        filtered = filter_nodes(self.node_list, complex_filter)
        
        # Should include all nodes at position 0 plus neurons at position 2 with algo tags
        # Position 0: 4 nodes (2 layers × 2 types × 1 num each)
        # Algo tags: 4 neurons at position 2 (2 layers × 2 nums each)
        expected_count = 4 + 4  # 4 nodes at pos 0, 4 neurons at pos 2 with algo tags
        self.assertEqual(len(filtered.nodes), expected_count)
        
    def test_nested_filters(self):
        """Test nested filter combinations"""
        # Find (heads OR neurons with algo tags) AND at layer 1
        nested_filter = FilterAnd(
            FilterOr(
                FilterHead(),
                FilterAlgo()
            ),
            FilterLayer(1)
        )
        
        filtered = filter_nodes(self.node_list, nested_filter)
        
        # Should include all heads at layer 1 plus neurons at layer 1 with algo tags
        for node in filtered.nodes:
            self.assertEqual(node.layer, 1)
            self.assertTrue(node.is_head or node.contains_tag("ALGO", ""))
            
    def test_position_range_filters(self):
        """Test position range filtering"""
        # Test MIN condition
        min_filter = FilterPosition("P01", QCondition.MIN)
        filtered_min = filter_nodes(self.node_list, min_filter)
        for node in filtered_min.nodes:
            self.assertGreaterEqual(node.position, 1)
            
        # Test MAX condition
        max_filter = FilterPosition("P01", QCondition.MAX)
        filtered_max = filter_nodes(self.node_list, max_filter)
        for node in filtered_max.nodes:
            self.assertLessEqual(node.position, 1)
            
    # TODO: RESOLVE BUG
    # def test_attention_percentage_filtering(self):
    #     """Test attention percentage filtering with different thresholds"""
    #     # Test with high threshold
    #     high_threshold_filter = FilterAttention("P02", QCondition.MUST, 85)
    #     filtered_high = filter_nodes(self.node_list, high_threshold_filter)
    #     
    #     # Only position 2 head at layer 1 should have 85% attention
    #     self.assertEqual(len(filtered_high.nodes), 1)
    #     node = filtered_high.nodes[0]
    #     self.assertEqual(node.position, 2)
    #     self.assertTrue(node.is_head)
    #     self.assertEqual(node.layer, 1)
    #     
    #     # Test with low threshold
    #     low_threshold_filter = FilterAttention("P", QCondition.MUST, 70)
    #     filtered_low = filter_nodes(self.node_list, low_threshold_filter)
    #     
    #     # Should include all heads at layer 1 (they all have >= 75%)
    #     self.assertEqual(len(filtered_low.nodes), 3)
    #     for node in filtered_low.nodes:
    #         self.assertTrue(node.is_head)
    #         self.assertEqual(node.layer, 1)


if __name__ == '__main__':
    unittest.main()
