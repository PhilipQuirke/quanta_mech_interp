import unittest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from QuantaMechInterp.quanta_map import (
    QuantaResult, calc_quanta_results, create_colormap, pale_color,
    find_quanta_result_by_row_col, show_quanta_patch, show_quanta_text,
    str_to_perc, show_quanta_perc, show_quanta_cells, calc_quanta_map_size,
    calc_quanta_rows_cols, calc_quanta_map, calc_quanta_map_numeric
)
from QuantaMechInterp.useful_node import NodeLocation, UsefulNode, UsefulNodeList
from QuantaMechInterp.model_config import ModelConfig


class TestQuantaResult(unittest.TestCase):
    """Test QuantaResult class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.node = UsefulNode(14, 2, True, 3, ["Major1:Minor1"])
        self.result = QuantaResult(self.node, "Test Text", 5)
    
    def test_initialization(self):
        """Test QuantaResult initialization."""
        # Test that it inherits from NodeLocation correctly
        self.assertIsInstance(self.result, NodeLocation)
        self.assertEqual(self.result.position, 14)
        self.assertEqual(self.result.layer, 2)
        self.assertTrue(self.result.is_head)
        self.assertEqual(self.result.num, 3)
        
        # Test QuantaResult-specific attributes
        self.assertEqual(self.result.cell_text, "Test Text")
        self.assertEqual(self.result.color_index, 5)
    
    def test_initialization_with_defaults(self):
        """Test QuantaResult initialization with default parameters."""
        result = QuantaResult(self.node)
        self.assertEqual(result.cell_text, "")
        self.assertEqual(result.color_index, 0)
    
    def test_initialization_with_mlp_node(self):
        """Test QuantaResult initialization with MLP node."""
        mlp_node = UsefulNode(5, 1, False, 0, [])
        result = QuantaResult(mlp_node, "MLP Text", 3)
        
        self.assertEqual(result.position, 5)
        self.assertEqual(result.layer, 1)
        self.assertFalse(result.is_head)
        self.assertEqual(result.num, 0)
        self.assertEqual(result.cell_text, "MLP Text")
        self.assertEqual(result.color_index, 3)
    
    def test_inherited_methods(self):
        """Test that inherited methods from NodeLocation work correctly."""
        self.assertEqual(self.result.name(), "P14L2H3")
        self.assertEqual(self.result.row_name, "L2H3")
    
    def test_cell_text_modification(self):
        """Test that cell_text can be modified after initialization."""
        self.result.cell_text = "Modified Text"
        self.assertEqual(self.result.cell_text, "Modified Text")
    
    def test_color_index_modification(self):
        """Test that color_index can be modified after initialization."""
        self.result.color_index = 10
        self.assertEqual(self.result.color_index, 10)


class TestCalcQuantaResults(unittest.TestCase):
    """Test calc_quanta_results function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cfg = ModelConfig()
        self.node_list = UsefulNodeList()
        
        # Add test nodes
        node1 = UsefulNode(0, 1, True, 2, ["test:tag1"])
        node2 = UsefulNode(1, 2, False, 0, ["test:tag2"])
        node3 = UsefulNode(2, 0, True, 1, ["test:tag3"])
        
        self.node_list.nodes = [node1, node2, node3]
        
        # Mock get_node_details function
        def mock_get_node_details(cfg, node, major_tag, minor_tag, num_shades):
            if node.position == 0:
                return "Text 1", 1
            elif node.position == 1:
                return "Long Text With Spaces", 2
            elif node.position == 2:
                return "", 0  # Empty text should be filtered out
            else:
                return "Default", 0
        
        self.mock_get_node_details = mock_get_node_details
    
    def test_calc_quanta_results_basic(self):
        """Test basic calc_quanta_results functionality."""
        results, num_text_lines, max_word_len = calc_quanta_results(
            self.cfg, self.node_list, "test", "tag", self.mock_get_node_details, 5
        )
        
        # Should only include nodes with non-empty cell_text
        self.assertEqual(len(results), 2)
        
        # Check first result
        self.assertEqual(results[0].position, 0)
        self.assertEqual(results[0].cell_text, "Text 1")
        self.assertEqual(results[0].color_index, 1)
        
        # Check second result
        self.assertEqual(results[1].position, 1)
        self.assertEqual(results[1].cell_text, "Long Text With Spaces")
        self.assertEqual(results[1].color_index, 2)
        
        # Check calculated metrics
        self.assertEqual(num_text_lines, 4)  # "Long Text With Spaces" has 3 spaces + 1
        self.assertEqual(max_word_len, 6)  # "Spaces" is 6 characters
    
    def test_calc_quanta_results_empty_nodes(self):
        """Test calc_quanta_results with empty node list."""
        empty_list = UsefulNodeList()
        results, num_text_lines, max_word_len = calc_quanta_results(
            self.cfg, empty_list, "test", "tag", self.mock_get_node_details, 5
        )
        
        self.assertEqual(len(results), 0)
        self.assertEqual(num_text_lines, 1)  # Default minimum
        self.assertEqual(max_word_len, 0)
    
    def test_calc_quanta_results_all_empty_text(self):
        """Test calc_quanta_results when all nodes return empty text."""
        def empty_get_node_details(cfg, node, major_tag, minor_tag, num_shades):
            return "", 0
        
        results, num_text_lines, max_word_len = calc_quanta_results(
            self.cfg, self.node_list, "test", "tag", empty_get_node_details, 5
        )
        
        self.assertEqual(len(results), 0)
        self.assertEqual(num_text_lines, 1)
        self.assertEqual(max_word_len, 0)


class TestCreateColormap(unittest.TestCase):
    """Test create_colormap function."""
    
    def test_create_colormap_standard(self):
        """Test create_colormap with standard_quanta=True."""
        colormap = create_colormap(True)
        self.assertEqual(colormap, plt.cm.winter)
    
    def test_create_colormap_custom(self):
        """Test create_colormap with standard_quanta=False."""
        colormap = create_colormap(False)
        # Should be a LinearSegmentedColormap
        self.assertIsInstance(colormap, type(plt.cm.winter))
        # Test that it can generate colors
        color = colormap(0.5)
        self.assertEqual(len(color), 4)  # RGBA


class TestPaleColor(unittest.TestCase):
    """Test pale_color function."""
    
    def test_pale_color_default_factor(self):
        """Test pale_color with default factor."""
        original_color = [1.0, 0.0, 0.0, 1.0]  # Red
        pale = pale_color(original_color)
        
        # Should be blend of original and white
        expected = np.array([1.0, 0.5, 0.5, 1.0])  # Pale red
        np.testing.assert_array_almost_equal(pale, expected)
    
    def test_pale_color_custom_factor(self):
        """Test pale_color with custom factor."""
        original_color = [0.0, 1.0, 0.0, 1.0]  # Green
        pale = pale_color(original_color, factor=0.8)
        
        # Should be mostly white with some green
        expected = np.array([0.8, 1.0, 0.8, 1.0])
        np.testing.assert_array_almost_equal(pale, expected)
    
    def test_pale_color_zero_factor(self):
        """Test pale_color with factor=0 (no change)."""
        original_color = [0.0, 0.0, 1.0, 1.0]  # Blue
        pale = pale_color(original_color, factor=0.0)
        
        np.testing.assert_array_almost_equal(pale, original_color)
    
    def test_pale_color_full_factor(self):
        """Test pale_color with factor=1 (pure white)."""
        original_color = [0.0, 0.0, 1.0, 1.0]  # Blue
        pale = pale_color(original_color, factor=1.0)
        
        expected = np.array([1.0, 1.0, 1.0, 1.0])  # White
        np.testing.assert_array_almost_equal(pale, expected)


class TestFindQuantaResultByRowCol(unittest.TestCase):
    """Test find_quanta_result_by_row_col function."""
    
    def setUp(self):
        """Set up test fixtures."""
        node1 = UsefulNode(0, 1, True, 2, [])
        node2 = UsefulNode(1, 2, False, 0, [])
        
        self.results = [
            QuantaResult(node1, "Text1", 1),
            QuantaResult(node2, "Text2", 2)
        ]
    
    def test_find_existing_result(self):
        """Test finding an existing result."""
        result = find_quanta_result_by_row_col("L1H2", 0, self.results)
        self.assertIsNotNone(result)
        self.assertEqual(result.cell_text, "Text1")
        self.assertEqual(result.position, 0)
    
    def test_find_existing_mlp_result(self):
        """Test finding an existing MLP result."""
        result = find_quanta_result_by_row_col("L2MLP", 1, self.results)
        self.assertIsNotNone(result)
        self.assertEqual(result.cell_text, "Text2")
        self.assertEqual(result.position, 1)
    
    def test_find_nonexistent_result(self):
        """Test finding a non-existent result."""
        result = find_quanta_result_by_row_col("L3H1", 5, self.results)
        self.assertIsNone(result)
    
    def test_find_with_empty_results(self):
        """Test finding in empty results list."""
        result = find_quanta_result_by_row_col("L1H2", 0, [])
        self.assertIsNone(result)


class TestStrToPerc(unittest.TestCase):
    """Test str_to_perc function."""
    
    def test_str_to_perc_basic(self):
        """Test basic percentage conversion."""
        self.assertEqual(str_to_perc("32%"), 32)
        self.assertEqual(str_to_perc("100%"), 100)
        self.assertEqual(str_to_perc("0%"), 0)
    
    def test_str_to_perc_less_than(self):
        """Test less-than percentage conversion."""
        self.assertEqual(str_to_perc("<1%"), 1)
        self.assertEqual(str_to_perc("<5%"), 5)
    
    def test_str_to_perc_edge_cases(self):
        """Test edge cases."""
        self.assertEqual(str_to_perc("99%"), 99)
        self.assertEqual(str_to_perc("<10%"), 10)


class TestCalcQuantaMapSize(unittest.TestCase):
    """Test calc_quanta_map_size function."""
    
    def test_calc_quanta_map_size_auto_width(self):
        """Test size calculation with automatic width."""
        width, height, square = calc_quanta_map_size(5, 10, 2, 4, -1, 8)
        
        # Width should be calculated based on columns and word length
        expected_width = 2 * (11 + 0) / 3  # (num_cols + max_word_len//5) * 2 / 3
        self.assertAlmostEqual(width, expected_width)
        self.assertEqual(height, 8)
        self.assertFalse(square)
    
    def test_calc_quanta_map_size_auto_height(self):
        """Test size calculation with automatic height."""
        width, height, square = calc_quanta_map_size(8, 6, 3, 6, 12, -1)
        
        # Height should be calculated based on rows and text lines
        # Formula: (7 + (num_text_lines-1)*4) * (num_data_rows + 2) / 12
        expected_height = (7 + (3-1)*4) * (8 + 2) / 12  # +2 for header and footer
        self.assertEqual(width, 12)
        self.assertAlmostEqual(height, expected_height)
        self.assertFalse(square)
    
    def test_calc_quanta_map_size_auto_both(self):
        """Test size calculation with both dimensions automatic."""
        width, height, square = calc_quanta_map_size(4, 6, 1, 8, -1, -1)
        
        expected_width = 2 * (7 + 1) / 3  # (num_cols + 1) + max_word_len//5 = (6+1) + 8//5 = 7 + 1
        expected_height = (7 + 0) * (4 + 2) / 12  # (7 + (num_text_lines-1)*4) * (num_data_rows + 2) / 12
        
        self.assertAlmostEqual(width, expected_width)
        self.assertAlmostEqual(height, expected_height)
        self.assertTrue(square)
    
    def test_calc_quanta_map_size_fixed_dimensions(self):
        """Test size calculation with fixed dimensions."""
        width, height, square = calc_quanta_map_size(5, 8, 2, 10, 15, 10)
        
        self.assertEqual(width, 15)
        self.assertEqual(height, 10)
        self.assertFalse(square)


class TestCalcQuantaRowsCols(unittest.TestCase):
    """Test calc_quanta_rows_cols function."""
    
    def setUp(self):
        """Set up test fixtures."""
        node1 = UsefulNode(0, 1, True, 2, [])  # L1H2, position 0
        node2 = UsefulNode(5, 2, False, 0, [])  # L2MLP, position 5
        node3 = UsefulNode(0, 0, True, 1, [])  # L0H1, position 0
        node4 = UsefulNode(10, 1, True, 0, [])  # L1H0, position 10
        
        self.results = [
            QuantaResult(node1, "Text1", 1),
            QuantaResult(node2, "Text2", 2),
            QuantaResult(node3, "Text3", 3),
            QuantaResult(node4, "Text4", 4)
        ]
    
    def test_calc_quanta_rows_cols_basic(self):
        """Test basic rows and columns calculation."""
        num_rows, num_cols, row_names, positions = calc_quanta_rows_cols(self.results)
        
        self.assertEqual(num_rows, 4)  # L0H1, L1H0, L1H2, L2MLP
        self.assertEqual(num_cols, 3)  # positions 0, 5, 10
        
        # Check sorted order
        self.assertEqual(row_names, ["L0H1", "L1H0", "L1H2", "L2MLP"])
        self.assertEqual(positions, [0, 5, 10])
    
    def test_calc_quanta_rows_cols_empty(self):
        """Test with empty results."""
        num_rows, num_cols, row_names, positions = calc_quanta_rows_cols([])
        
        self.assertEqual(num_rows, 0)
        self.assertEqual(num_cols, 0)
        self.assertEqual(row_names, [])
        self.assertEqual(positions, [])
    
    def test_calc_quanta_rows_cols_single_result(self):
        """Test with single result."""
        single_result = [self.results[0]]
        num_rows, num_cols, row_names, positions = calc_quanta_rows_cols(single_result)
        
        self.assertEqual(num_rows, 1)
        self.assertEqual(num_cols, 1)
        self.assertEqual(row_names, ["L1H2"])
        self.assertEqual(positions, [0])


class TestQuantaMapIntegration(unittest.TestCase):
    """Integration tests for quanta map functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cfg = ModelConfig()
        self.cfg.initialize_token_positions(5, 3, True)
        
        self.node_list = UsefulNodeList()
        node1 = UsefulNode(0, 1, True, 0, ["test:tag1"])
        node2 = UsefulNode(1, 0, False, 0, ["test:tag2"])
        self.node_list.nodes = [node1, node2]
        
        def mock_get_node_details(cfg, node, major_tag, minor_tag, num_shades):
            return f"Node{node.position}", node.position
        
        self.mock_get_node_details = mock_get_node_details
    
    @patch('matplotlib.pyplot.subplots')
    def test_calc_quanta_map_basic(self, mock_subplots):
        """Test basic calc_quanta_map functionality."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        ax, results, num_results = calc_quanta_map(
            self.cfg, True, 5, self.node_list, "test", "tag", 
            self.mock_get_node_details
        )
        
        self.assertIsNotNone(ax)
        self.assertEqual(len(results), 2)
        self.assertEqual(num_results, 2)
        
        # Verify matplotlib was called
        mock_subplots.assert_called_once()
    
    @patch('matplotlib.pyplot.subplots')
    def test_calc_quanta_map_empty_nodes(self, mock_subplots):
        """Test calc_quanta_map with empty node list."""
        empty_list = UsefulNodeList()
        
        result = calc_quanta_map(
            self.cfg, True, 5, empty_list, "test", "tag", 
            self.mock_get_node_details
        )
        
        # Should return None for empty results
        self.assertIsNone(result[0])
        self.assertEqual(len(result[1]), 0)
        self.assertEqual(result[2], 0)
        
        # Matplotlib should not be called
        mock_subplots.assert_not_called()
    
    @patch('matplotlib.pyplot.subplots')
    def test_calc_quanta_map_numeric_basic(self, mock_subplots):
        """Test basic calc_quanta_map_numeric functionality."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        ax, results, num_results = calc_quanta_map_numeric(
            self.cfg, False, 3, self.node_list, "test", "tag", 
            self.mock_get_node_details
        )
        
        self.assertIsNotNone(ax)
        self.assertEqual(len(results), 2)
        self.assertEqual(num_results, 2)
        
        # Verify matplotlib was called
        mock_subplots.assert_called_once()


class TestQuantaMapVisualizationFunctions(unittest.TestCase):
    """Test visualization helper functions."""
    
    @patch('matplotlib.pyplot.Rectangle')
    def test_show_quanta_patch(self, mock_rectangle):
        """Test show_quanta_patch function."""
        mock_ax = MagicMock()
        
        show_quanta_patch(mock_ax, 1.0, 2.0, "red", 3, 4)
        
        # Verify Rectangle was created with correct parameters
        mock_rectangle.assert_called_once_with((1.0, 2.0), 3, 4, fill=True, color="red")
        mock_ax.add_patch.assert_called_once()
    
    def test_show_quanta_text_with_text(self):
        """Test show_quanta_text with non-empty text."""
        mock_ax = MagicMock()
        
        show_quanta_text(mock_ax, 1.0, 2.0, "Test Text", 12)
        
        mock_ax.text.assert_called_once_with(
            1.5, 2.5, "Test Text", ha='center', va='center', color='black', fontsize=12
        )
    
    def test_show_quanta_text_empty(self):
        """Test show_quanta_text with empty text."""
        mock_ax = MagicMock()
        
        show_quanta_text(mock_ax, 1.0, 2.0, "", 12)
        show_quanta_text(mock_ax, 1.0, 2.0, None, 12)
        
        # Should not call ax.text for empty/None text
        mock_ax.text.assert_not_called()
    
    @patch('matplotlib.pyplot.Circle')
    def test_show_quanta_perc_large_percentage(self, mock_circle):
        """Test show_quanta_perc with percentage >= 20."""
        mock_ax = MagicMock()
        
        show_quanta_perc(mock_ax, 1, 2, "blue", 50, 100)
        
        # Should create circle with radius based on percentage
        expected_radius = (50 / 100) * 0.5
        mock_circle.assert_called_once_with((1.5, 2.5), expected_radius, color="blue")
        mock_ax.add_artist.assert_called_once()
    
    @patch('matplotlib.pyplot.Circle')
    def test_show_quanta_perc_small_percentage(self, mock_circle):
        """Test show_quanta_perc with percentage < 20."""
        mock_ax = MagicMock()
        
        show_quanta_perc(mock_ax, 1, 2, "blue", 10, 100)
        
        # Should not create circle for small percentages
        mock_circle.assert_not_called()
        mock_ax.add_artist.assert_not_called()


if __name__ == '__main__':
    unittest.main()
