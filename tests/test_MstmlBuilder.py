#!/usr/bin/env python3
"""
Unit tests for MstmlBuilder class
"""

import sys
import os
import time
import numpy as np
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from build import MstmlBuilder

# Add root/build directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMstmlBuilder(unittest.TestCase):
    """Test cases for MstmlBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        self.env_name = "mstml_tester"
        self.setUp_complete = True

    """==================================================="""
    """ Add unit test cases below """
    """==================================================="""

    def test_constructor(self):
        """Test if MstmlBuilder constructor works as expected"""
        try:
            mstmlbuilder = MstmlBuilder()

            # Check that basic values were initialized
            self.assertEqual("mstml", mstmlbuilder.env_name)
            if mstmlbuilder.conda_packages:
                self.assertTrue(type(mstmlbuilder.conda_packages) == list)
                if len(mstmlbuilder.conda_packages) > 0:
                    for cpkg in mstmlbuilder.conda_packages:
                        self.assertTrue(type(cpkg) == str)
            if mstmlbuilder.pip_packages:
                self.assertTrue(type(mstmlbuilder.pip_packages) == list)
                if len(mstmlbuilder.pip_packages) > 0:
                    for ppkg in mstmlbuilder.pip_packages:
                        self.assertTrue(type(ppkg) == str)

        except Exception as e:
            self.fail(f"MstmlBuilder constructor test failed: {e}")



def run_MstmlBuilder_tests():
    """Run MstmlBuilder verification."""
    print("=" * 60)
    print("MstmlBuilder class verification")
    print("=" * 60)

    # Run unit tests
    print("\nRunning unit tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMstmlBuilder)
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
    success = run_MstmlBuilder_tests()
    sys.exit(0 if success else 1)