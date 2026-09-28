"""
ClaimTrace: Test Suite Runner
Runs unit and integration tests using Python's standard test runner.
Works seamlessly in any Python environment (Windows, Mac, Linux) with or without pytest installed.
"""

import sys
import os
import unittest

# Add workspace root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_preprocessing import TestPreprocessing
from tests.test_retrieval import TestRetrieval
from tests.test_verification import TestVerification
from tests.test_pipeline import TestPipeline
from tests.test_data_validation import TestDataValidation
from tests.test_evaluation import TestEvaluation


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestPreprocessing,
        TestRetrieval,
        TestVerification,
        TestPipeline,
        TestDataValidation,
        TestEvaluation,
    ]

    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    print("=" * 65)
    print("ClaimTrace: Automated Verification & Test Suite")
    print(f"Total test cases discovered: {suite.countTestCases()}")
    print("=" * 65)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 65)
    print(f"Tests Run: {result.testsRun}")
    print(f"Errors: {len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Status: {'PASSED' if result.wasSuccessful() else 'FAILED'}")
    print("=" * 65)

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
