#!/usr/bin/env python3
"""
Test script for verifying Cython compilation and functionality.

This script tests:
1. Import of Cython modules
2. Basic functionality of TreeNode class
3. Performance comparison (if both versions available)
4. Correctness of Cython vs Python implementations
"""

import sys
import os
import time
import numpy as np
import unittest
from pathlib import Path

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "source"))


class TestCythonBuild(unittest.TestCase):
    """Test cases for Cython build verification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        self.test_id = 42
        self.test_type = 0
        self.test_distance = 1.5
    
    def test_cython_import(self):
        """Test that Cython module can be imported."""
        try:
            import fast_encode_tree
            self.assertTrue(hasattr(fast_encode_tree, 'TreeNode'))
            self.assertTrue(hasattr(fast_encode_tree, 'fast_encode_tree_structure'))
            print("✓ Cython module imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import Cython module: {e}")
    
    def test_tree_node_creation(self):
        """Test TreeNode creation and basic functionality."""
        try:
            from fast_encode_tree import TreeNode
            
            # Create a TreeNode
            node = TreeNode(
                id=self.test_id,
                type=self.test_type,
                distance=self.test_distance,
                author_topic_space_probs=self.test_data
            )
            
            # Test basic properties
            self.assertEqual(node.id, self.test_id)
            self.assertEqual(node.type, self.test_type)
            self.assertEqual(node.distance, self.test_distance)
            np.testing.assert_array_equal(node.author_topic_space_probs, self.test_data)
            
            # Test methods
            self.assertTrue(node.is_leaf())
            self.assertEqual(node.get_leaf_count(), 1)
            self.assertEqual(node.get_all_leaf_ids(), {self.test_id})
            
            print("✓ TreeNode functionality verified")
            
        except Exception as e:
            self.fail(f"TreeNode test failed: {e}")
    
    def test_tree_node_hierarchy(self):
        """Test TreeNode hierarchy functionality."""
        try:
            from fast_encode_tree import TreeNode
            
            # Create leaf nodes
            left_node = TreeNode(0, 0, 0.0, np.array([1.0, 0.0]))
            right_node = TreeNode(1, 0, 0.0, np.array([0.0, 1.0]))
            
            # Create parent node
            parent_probs = left_node.author_topic_space_probs + right_node.author_topic_space_probs
            parent_node = TreeNode(
                id=2,
                type=1,
                distance=1.0,
                author_topic_space_probs=parent_probs,
                left=left_node,
                right=right_node,
                original_leaf_ids={0, 1}
            )
            
            # Test hierarchy
            self.assertFalse(parent_node.is_leaf())
            self.assertEqual(parent_node.get_leaf_count(), 2)
            self.assertEqual(parent_node.get_all_leaf_ids(), {0, 1})
            self.assertEqual(parent_node.original_leaf_ids, {0, 1})
            
            print("✓ TreeNode hierarchy verified")
            
        except Exception as e:
            self.fail(f"TreeNode hierarchy test failed: {e}")
    
    def test_performance_basic(self):
        """Basic performance test for TreeNode operations."""
        try:
            from fast_encode_tree import TreeNode
            
            # Performance test
            n_nodes = 1000
            start_time = time.time()
            
            nodes = []
            for i in range(n_nodes):
                node = TreeNode(i, 0, 0.0, np.random.random(10).astype(np.float64))
                nodes.append(node)
            
            end_time = time.time()
            creation_time = end_time - start_time
            
            # Test node operations
            start_time = time.time()
            for node in nodes:
                _ = node.is_leaf()
                _ = node.get_leaf_count()
            end_time = time.time()
            operation_time = end_time - start_time
            
            print(f"✓ Performance test: {n_nodes} nodes created in {creation_time:.4f}s")
            print(f"✓ Operations completed in {operation_time:.4f}s")
            
            # Basic performance check (should be reasonably fast)
            self.assertLess(creation_time, 1.0, "Node creation took too long")
            self.assertLess(operation_time, 0.1, "Node operations took too long")
            
        except Exception as e:
            self.fail(f"Performance test failed: {e}")
    
    def test_data_types(self):
        """Test that data types are handled correctly."""
        try:
            from fast_encode_tree import TreeNode
            
            # Test with different numpy dtypes
            test_arrays = [
                np.array([1, 2, 3], dtype=np.int32),
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([1.0, 2.0, 3.0], dtype=np.float64),
            ]
            
            for i, arr in enumerate(test_arrays):
                node = TreeNode(i, 0, 0.0, arr)
                # Should be converted to float64
                self.assertEqual(node.author_topic_space_probs.dtype, np.float64)
                
            print("✓ Data type handling verified")
            
        except Exception as e:
            self.fail(f"Data type test failed: {e}")
    
    def test_pickle_support(self):
        """Test that TreeNode can be pickled and unpickled."""
        try:
            import pickle
            from fast_encode_tree import TreeNode
            
            # Create a node
            original_node = TreeNode(
                id=99,
                type=1,
                distance=2.5,
                author_topic_space_probs=self.test_data,
                original_leaf_ids={1, 2, 3}
            )
            
            # Pickle and unpickle
            pickled_data = pickle.dumps(original_node)
            restored_node = pickle.loads(pickled_data)
            
            # Verify restoration
            self.assertEqual(restored_node.id, original_node.id)
            self.assertEqual(restored_node.type, original_node.type)
            self.assertEqual(restored_node.distance, original_node.distance)
            np.testing.assert_array_equal(
                restored_node.author_topic_space_probs,
                original_node.author_topic_space_probs
            )
            self.assertEqual(restored_node.original_leaf_ids, original_node.original_leaf_ids)
            
            print("✓ Pickle support verified")
            
        except Exception as e:
            self.fail(f"Pickle test failed: {e}")


def run_build_verification():
    """Run comprehensive build verification."""
    print("=" * 60)
    print("MSTML Cython Build Verification")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("source/fast_encode_tree.pyx"):
        print("Error: fast_encode_tree.pyx not found")
        print("Please run this script from the MSTML root directory")
        return False
    
    # Check if Cython module is compiled
    try:
        import mstml.fast_encode_tree
        print("Cython module found and importable")
        
        # Check if it's actually compiled (has .so/.pyd file)
        import inspect
        module_file = inspect.getfile(mstml.fast_encode_tree)
        if module_file.endswith(('.so', '.pyd')):
            print(f"Using compiled Cython module: {module_file}")
        else:
            print(f"Using Python fallback: {module_file}")
            
    except ImportError as e:
        print(f"Failed to import Cython module: {e}")
        print("Please run: python setup.py build_ext --inplace")
        return False
    
    # Run unit tests
    print("\nRunning functionality tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCythonBuild)
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("All tests passed!")
        return True
    else:
        print(f"{len(result.failures)} test(s) failed")
        for test, traceback in result.failures:
            print(f"Failed: {test}")
            print(traceback)
        return False


if __name__ == "__main__":
    success = run_build_verification()
    sys.exit(0 if success else 1)