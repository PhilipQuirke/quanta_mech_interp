import unittest
import json
import tempfile
import os
from unittest.mock import patch, mock_open

from QuantaMechInterp.useful_node import (
    position_name, position_name_to_int, row_location_name, location_name, 
    answer_name, NodeLocation, str_to_node_location, UsefulNode, UsefulNodeList
)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions for node naming and conversion."""
    
    def test_position_name(self):
        """Test position_name function."""
        self.assertEqual(position_name(0), "P0")
        self.assertEqual(position_name(14), "P14")
        self.assertEqual(position_name(999), "P999")
    
    def test_position_name_to_int(self):
        """Test position_name_to_int function."""
        self.assertEqual(position_name_to_int("P0"), 0)
        self.assertEqual(position_name_to_int("P14"), 14)
        self.assertEqual(position_name_to_int("P999"), 999)
        
        # Test with extra P's
        self.assertEqual(position_name_to_int("PP14"), 14)
        self.assertEqual(position_name_to_int("PPP5"), 5)
    
    def test_row_location_name(self):
        """Test row_location_name function."""
        # Test attention heads
        self.assertEqual(row_location_name(1, True, 2), "L1H2")
        self.assertEqual(row_location_name(0, True, 0), "L0H0")
        self.assertEqual(row_location_name(5, True, 10), "L5H10")
        
        # Test MLP neurons
        self.assertEqual(row_location_name(1, False, 0), "L1MLP")
        self.assertEqual(row_location_name(2, False, 5), "L2MLP")
    
    def test_location_name(self):
        """Test location_name function."""
        # Test with short_position=False (default)
        self.assertEqual(location_name(1, 2, True, 3), "P01L2H3")
        self.assertEqual(location_name(14, 2, True, 3), "P14L2H3")
        self.assertEqual(location_name(5, 1, False, 0), "P05L1MLP")
        
        # Test with short_position=True
        self.assertEqual(location_name(1, 2, True, 3, True), "P1L2H3")
        self.assertEqual(location_name(14, 2, True, 3, True), "P14L2H3")
    
    def test_answer_name(self):
        """Test answer_name function."""
        self.assertEqual(answer_name(0), "A0")
        self.assertEqual(answer_name(3), "A3")
        self.assertEqual(answer_name(99), "A99")


class TestNodeLocation(unittest.TestCase):
    """Test NodeLocation class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.node_head = NodeLocation(14, 2, True, 3)
        self.node_mlp = NodeLocation(5, 1, False, 0)
    
    def test_initialization(self):
        """Test NodeLocation initialization."""
        self.assertEqual(self.node_head.position, 14)
        self.assertEqual(self.node_head.layer, 2)
        self.assertEqual(self.node_head.is_head, True)
        self.assertEqual(self.node_head.num, 3)
        
        self.assertEqual(self.node_mlp.position, 5)
        self.assertEqual(self.node_mlp.layer, 1)
        self.assertEqual(self.node_mlp.is_head, False)
        self.assertEqual(self.node_mlp.num, 0)
    
    def test_name_method(self):
        """Test name method."""
        self.assertEqual(self.node_head.name(), "P14L2H3")
        self.assertEqual(self.node_head.name(short_position=True), "P14L2H3")
        self.assertEqual(self.node_mlp.name(), "P05L1MLP")
        self.assertEqual(self.node_mlp.name(short_position=True), "P5L1MLP")
    
    def test_row_name_property(self):
        """Test row_name property."""
        self.assertEqual(self.node_head.row_name, "L2H3")
        self.assertEqual(self.node_mlp.row_name, "L1MLP")


class TestStrToNodeLocation(unittest.TestCase):
    """Test str_to_node_location function."""
    
    def test_valid_head_patterns(self):
        """Test parsing valid attention head patterns."""
        node = str_to_node_location("P14L2H3")
        self.assertIsNotNone(node)
        self.assertEqual(node.position, 14)
        self.assertEqual(node.layer, 2)
        self.assertEqual(node.is_head, True)
        self.assertEqual(node.num, 3)
        
        # Test with larger numbers
        node = str_to_node_location("P12345L67890H11111")
        self.assertIsNotNone(node)
        self.assertEqual(node.position, 12345)
        self.assertEqual(node.layer, 67890)
        self.assertEqual(node.num, 11111)
    
    def test_valid_mlp_patterns(self):
        """Test parsing valid MLP patterns."""
        node = str_to_node_location("P05L1M0")
        self.assertIsNotNone(node)
        self.assertEqual(node.position, 5)
        self.assertEqual(node.layer, 1)
        self.assertEqual(node.is_head, False)
        self.assertEqual(node.num, 0)
    
    def test_invalid_patterns(self):
        """Test parsing invalid patterns."""
        self.assertIsNone(str_to_node_location("invalid"))
        self.assertIsNone(str_to_node_location("P14L2"))
        self.assertIsNone(str_to_node_location("L2H3"))
        self.assertIsNone(str_to_node_location("P14H3"))
        self.assertIsNone(str_to_node_location(""))


class TestUsefulNode(unittest.TestCase):
    """Test UsefulNode class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.node = UsefulNode(14, 2, True, 3, ["Major1:Minor1", "Major2:Minor2"])
        self.empty_node = UsefulNode(5, 1, False, 0, [])
    
    def test_initialization(self):
        """Test UsefulNode initialization."""
        self.assertEqual(self.node.position, 14)
        self.assertEqual(self.node.layer, 2)
        self.assertEqual(self.node.is_head, True)
        self.assertEqual(self.node.num, 3)
        self.assertEqual(len(self.node.tags), 2)
        self.assertIn("Major1:Minor1", self.node.tags)
        self.assertIn("Major2:Minor2", self.node.tags)
    
    def test_add_tag(self):
        """Test add_tag method."""
        # Add new tag
        result = self.empty_node.add_tag("Major1", "Minor1")
        self.assertEqual(result, 1)
        self.assertIn("Major1:Minor1", self.empty_node.tags)
        
        # Add duplicate tag
        result = self.empty_node.add_tag("Major1", "Minor1")
        self.assertEqual(result, 0)
        self.assertEqual(len(self.empty_node.tags), 1)
        
        # Add tag with empty minor
        result = self.empty_node.add_tag("Major2", "")
        self.assertEqual(result, 1)
        self.assertIn("Major2:", self.empty_node.tags)
    
    def test_add_tag_assertion(self):
        """Test add_tag assertion for empty major_tag."""
        with self.assertRaises(AssertionError):
            self.node.add_tag("", "Minor1")
    
    def test_reset_tags_all(self):
        """Test reset_tags with empty major_tag (removes all)."""
        self.node.reset_tags("")
        self.assertEqual(len(self.node.tags), 0)
    
    def test_reset_tags_by_major(self):
        """Test reset_tags by major tag."""
        self.node.add_tag("Major1", "Minor2")
        self.node.add_tag("Major3", "Minor1")
        
        self.node.reset_tags("Major1")
        
        # Should remove all Major1 tags
        remaining_tags = [tag for tag in self.node.tags if tag.startswith("Major1")]
        self.assertEqual(len(remaining_tags), 0)
        
        # Should keep other tags
        self.assertIn("Major2:Minor2", self.node.tags)
        self.assertIn("Major3:Minor1", self.node.tags)
    
    def test_reset_tags_by_major_and_minor(self):
        """Test reset_tags by major and minor tag."""
        self.node.add_tag("Major1", "Minor2")
        self.node.add_tag("Major1", "Minor3")
        
        self.node.reset_tags("Major1", "Minor2")
        
        # Should remove only Major1:Minor2
        self.assertNotIn("Major1:Minor2", self.node.tags)
        self.assertIn("Major1:Minor1", self.node.tags)
        self.assertIn("Major1:Minor3", self.node.tags)
    
    def test_filter_tags(self):
        """Test filter_tags method."""
        self.node.add_tag("Major1", "Minor2")
        self.node.add_tag("Major1", "Minor3")
        self.node.add_tag("Major3", "Minor1")
        
        # Filter by major tag only
        result = self.node.filter_tags("Major1")
        self.assertIn("Minor1", result)
        self.assertIn("Minor2", result)
        self.assertIn("Minor3", result)
        self.assertEqual(len(result), 3)
        
        # Filter by major and minor tag
        result = self.node.filter_tags("Major1", "Minor2")
        self.assertIn("Minor2", result)
        self.assertEqual(len(result), 1)
        
        # Filter non-existent major tag
        result = self.node.filter_tags("NonExistent")
        self.assertEqual(len(result), 0)
    
    def test_filter_tags_assertion(self):
        """Test filter_tags assertion for empty major_tag."""
        with self.assertRaises(AssertionError):
            self.node.filter_tags("")
    
    def test_min_tag_suffix(self):
        """Test min_tag_suffix method."""
        self.node.add_tag("Major1", "Minor3")
        self.node.add_tag("Major1", "Minor1")
        
        # Should return minimum minor tag
        result = self.node.min_tag_suffix("Major1")
        self.assertEqual(result, "Minor1")
        
        # With minor tag filter
        result = self.node.min_tag_suffix("Major1", "Minor3")
        self.assertEqual(result, "Minor3")
        
        # Non-existent major tag
        result = self.node.min_tag_suffix("NonExistent")
        self.assertEqual(result, "")
    
    def test_min_tag_suffix_assertion(self):
        """Test min_tag_suffix assertion for empty major_tag."""
        with self.assertRaises(AssertionError):
            self.node.min_tag_suffix("")
    
    def test_only_tag(self):
        """Test only_tag method."""
        # Single tag case
        result = self.node.only_tag("Major2")
        self.assertEqual(result, "Minor2")
        
        # No matching tag
        result = self.node.only_tag("NonExistent")
        self.assertEqual(result, "")
    
    def test_only_tag_multiple_assertion(self):
        """Test only_tag assertion when multiple tags exist."""
        self.node.add_tag("Major1", "Minor2")
        
        with patch('builtins.print') as mock_print:
            with self.assertRaises(AssertionError):
                self.node.only_tag("Major1")
            mock_print.assert_called()
    
    def test_only_tag_assertion(self):
        """Test only_tag assertion for empty major_tag."""
        with self.assertRaises(AssertionError):
            self.node.only_tag("")
    
    def test_contains_tag(self):
        """Test contains_tag method."""
        self.node.add_tag("Major1", "P14=25")
        
        # Exact match
        self.assertTrue(self.node.contains_tag("Major1", "Minor1"))
        self.assertTrue(self.node.contains_tag("Major2", "Minor2"))
        
        # Partial match (contains)
        self.assertTrue(self.node.contains_tag("Major1", "P14"))
        self.assertTrue(self.node.contains_tag("Major1", "25"))
        
        # No match
        self.assertFalse(self.node.contains_tag("Major1", "NonExistent"))
        self.assertFalse(self.node.contains_tag("NonExistent", "Minor1"))
    
    def test_contains_tag_assertion(self):
        """Test contains_tag assertion for empty major_tag."""
        with self.assertRaises(AssertionError):
            self.node.contains_tag("", "Minor1")
    
    def test_to_dict(self):
        """Test to_dict method."""
        # All tags
        result = self.node.to_dict()
        expected = {
            "position": 14,
            "layer": 2,
            "is_head": True,
            "num": 3,
            "tags": ["Major1:Minor1", "Major2:Minor2"]
        }
        self.assertEqual(result, expected)
        
        # Filtered by major tag
        result = self.node.to_dict("Major1")
        self.assertEqual(result["tags"], ["Major1:Minor1"])
        
        # Non-existent major tag
        result = self.node.to_dict("NonExistent")
        self.assertEqual(result["tags"], [])
    
    def test_from_dict(self):
        """Test from_dict class method."""
        data = {
            "position": 10,
            "layer": 1,
            "is_head": False,
            "num": 5,
            "tags": ["Test:Tag1", "Test:Tag2"]
        }
        
        node = UsefulNode.from_dict(data)
        self.assertEqual(node.position, 10)
        self.assertEqual(node.layer, 1)
        self.assertEqual(node.is_head, False)
        self.assertEqual(node.num, 5)
        self.assertEqual(node.tags, ["Test:Tag1", "Test:Tag2"])


class TestUsefulNodeList(unittest.TestCase):
    """Test UsefulNodeList class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.node_list = UsefulNodeList()
        self.node1 = UsefulNode(14, 2, True, 3, ["Major1:Minor1"])
        self.node2 = UsefulNode(5, 1, False, 0, ["Major2:Minor2"])
        self.node3 = UsefulNode(10, 0, True, 1, ["Major1:Minor3"])
    
    def test_initialization(self):
        """Test UsefulNodeList initialization."""
        self.assertEqual(len(self.node_list.nodes), 0)
        self.assertEqual(self.node_list.node_names, "")
        self.assertEqual(self.node_list.num_heads, 0)
        self.assertEqual(self.node_list.num_neurons, 0)
    
    def test_node_names_property(self):
        """Test node_names property."""
        self.node_list.nodes = [self.node1, self.node2]
        expected = "P14L2H3, P05L1MLP"
        self.assertEqual(self.node_list.node_names, expected)
        
        # Test with single node
        self.node_list.nodes = [self.node1]
        self.assertEqual(self.node_list.node_names, "P14L2H3")
    
    def test_num_heads_property(self):
        """Test num_heads property."""
        self.node_list.nodes = [self.node1, self.node2, self.node3]
        self.assertEqual(self.node_list.num_heads, 2)  # node1 and node3 are heads
        
        # Test with no heads
        self.node_list.nodes = [self.node2]
        self.assertEqual(self.node_list.num_heads, 0)
    
    def test_num_neurons_property(self):
        """Test num_neurons property."""
        self.node_list.nodes = [self.node1, self.node2, self.node3]
        self.assertEqual(self.node_list.num_neurons, 1)  # only node2 is MLP
        
        # Test with no neurons
        self.node_list.nodes = [self.node1, self.node3]
        self.assertEqual(self.node_list.num_neurons, 0)
    
    def test_get_node(self):
        """Test get_node method."""
        self.node_list.nodes = [self.node1, self.node2]
        
        # Find existing node
        location = NodeLocation(14, 2, True, 3)
        found_node = self.node_list.get_node(location)
        self.assertIsNotNone(found_node)
        self.assertEqual(found_node.position, 14)
        
        # Try to find non-existent node
        location = NodeLocation(99, 99, True, 99)
        found_node = self.node_list.get_node(location)
        self.assertIsNone(found_node)
    
    def test_get_node_by_tag(self):
        """Test get_node_by_tag method."""
        self.node_list.nodes = [self.node1, self.node2, self.node3]
        
        # Find existing tag
        found_node = self.node_list.get_node_by_tag("Major1", "Minor1")
        self.assertIsNotNone(found_node)
        self.assertEqual(found_node.position, 14)
        
        # Find non-existent tag
        found_node = self.node_list.get_node_by_tag("NonExistent", "Tag")
        self.assertIsNone(found_node)
    
    def test_add_node_tag_existing_node(self):
        """Test add_node_tag with existing node."""
        self.node_list.nodes = [self.node1]
        
        location = NodeLocation(14, 2, True, 3)
        result = self.node_list.add_node_tag(location, "Major3", "Minor3")
        
        self.assertEqual(result, 1)
        self.assertTrue(self.node1.contains_tag("Major3", "Minor3"))
    
    def test_add_node_tag_new_node(self):
        """Test add_node_tag with new node."""
        location = NodeLocation(99, 99, True, 99)
        result = self.node_list.add_node_tag(location, "Major1", "Minor1")
        
        self.assertEqual(result, 1)
        self.assertEqual(len(self.node_list.nodes), 1)
        
        new_node = self.node_list.get_node(location)
        self.assertIsNotNone(new_node)
        self.assertTrue(new_node.contains_tag("Major1", "Minor1"))
    
    def test_add_node_tag_duplicate(self):
        """Test add_node_tag with duplicate tag."""
        self.node_list.nodes = [self.node1]
        
        location = NodeLocation(14, 2, True, 3)
        result = self.node_list.add_node_tag(location, "Major1", "Minor1")
        
        self.assertEqual(result, 0)  # Tag already exists
    
    def test_reset_node_tags_all(self):
        """Test reset_node_tags with no parameters (removes all)."""
        self.node_list.nodes = [self.node1, self.node2]
        
        self.node_list.reset_node_tags()
        
        self.assertEqual(len(self.node1.tags), 0)
        self.assertEqual(len(self.node2.tags), 0)
    
    def test_reset_node_tags_by_major(self):
        """Test reset_node_tags by major tag."""
        self.node1.add_tag("Major1", "Minor2")
        self.node_list.nodes = [self.node1, self.node2]
        
        self.node_list.reset_node_tags("Major1")
        
        # Should remove all Major1 tags from node1
        major1_tags = [tag for tag in self.node1.tags if tag.startswith("Major1")]
        self.assertEqual(len(major1_tags), 0)
        
        # Should keep Major2 tags in node2
        self.assertTrue(self.node2.contains_tag("Major2", "Minor2"))
    
    def test_print_node_tags(self):
        """Test print_node_tags method."""
        self.node_list.nodes = [self.node1, self.node2]
        
        with patch('builtins.print') as mock_print:
            self.node_list.print_node_tags()
            self.assertEqual(mock_print.call_count, 2)
        
        # Test with major tag filter
        with patch('builtins.print') as mock_print:
            self.node_list.print_node_tags("Major1")
            mock_print.assert_called()
        
        # Test with show_empty_tags=False
        empty_node = UsefulNode(1, 1, True, 1, [])
        self.node_list.nodes = [empty_node]
        
        with patch('builtins.print') as mock_print:
            self.node_list.print_node_tags(show_empty_tags=False)
            mock_print.assert_not_called()
    
    def test_sort_nodes(self):
        """Test sort_nodes method."""
        # Create nodes in unsorted order
        node_a = UsefulNode(20, 1, True, 1, [])
        node_b = UsefulNode(5, 0, False, 0, [])
        node_c = UsefulNode(10, 2, True, 0, [])
        
        self.node_list.nodes = [node_a, node_b, node_c]
        self.node_list.sort_nodes()
        
        # Should be sorted by name (which includes position with padding)
        expected_order = ["P05L0MLP", "P10L2H0", "P20L1H1"]
        actual_order = [node.name() for node in self.node_list.nodes]
        self.assertEqual(actual_order, expected_order)
    
    def test_save_and_load_nodes(self):
        """Test save_nodes and load_nodes methods."""
        self.node_list.nodes = [self.node1, self.node2]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            temp_filename = temp_file.name
        
        try:
            # Test save
            self.node_list.save_nodes(temp_filename)
            
            # Verify file was created and contains expected data
            with open(temp_filename, 'r') as file:
                data = json.load(file)
                self.assertEqual(len(data), 2)
                self.assertEqual(data[0]['position'], 14)
                self.assertEqual(data[1]['position'], 5)
            
            # Test load into new list
            new_node_list = UsefulNodeList()
            new_node_list.load_nodes(temp_filename)
            
            self.assertEqual(len(new_node_list.nodes), 2)
            
            # Verify loaded nodes
            loaded_node1 = new_node_list.get_node(NodeLocation(14, 2, True, 3))
            self.assertIsNotNone(loaded_node1)
            self.assertTrue(loaded_node1.contains_tag("Major1", "Minor1"))
            
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def test_save_nodes_with_major_tag_filter(self):
        """Test save_nodes with major tag filter."""
        self.node1.add_tag("Major2", "Minor4")
        self.node_list.nodes = [self.node1]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            temp_filename = temp_file.name
        
        try:
            # Save only Major1 tags
            self.node_list.save_nodes(temp_filename, "Major1")
            
            with open(temp_filename, 'r') as file:
                data = json.load(file)
                self.assertEqual(len(data), 1)
                self.assertEqual(data[0]['tags'], ["Major1:Minor1"])
                
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def test_load_nodes_merge_existing(self):
        """Test load_nodes merges with existing nodes."""
        # Start with one node
        self.node_list.nodes = [self.node1]
        
        # Create test data with overlapping and new nodes
        test_data = [
            {
                "position": 14,
                "layer": 2,
                "is_head": True,
                "num": 3,
                "tags": ["Major3:Minor3"]
            },
            {
                "position": 99,
                "layer": 99,
                "is_head": False,
                "num": 99,
                "tags": ["Major4:Minor4"]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            json.dump(test_data, temp_file)
            temp_filename = temp_file.name
        
        try:
            self.node_list.load_nodes(temp_filename)
            
            # Should have 2 nodes total (existing + new)
            self.assertEqual(len(self.node_list.nodes), 2)
            
            # Existing node should have both old and new tags
            existing_node = self.node_list.get_node(NodeLocation(14, 2, True, 3))
            self.assertTrue(existing_node.contains_tag("Major1", "Minor1"))  # Original
            self.assertTrue(existing_node.contains_tag("Major3", "Minor3"))  # Loaded
            
            # New node should exist
            new_node = self.node_list.get_node(NodeLocation(99, 99, False, 99))
            self.assertIsNotNone(new_node)
            self.assertTrue(new_node.contains_tag("Major4", "Minor4"))
            
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


if __name__ == '__main__':
    unittest.main()
