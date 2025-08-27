import unittest

from Tests.tests_useful import TestUseful
from Tests.tests_huggingface import TestHuggingFace
from Tests.tests_config import TestModelConfig, TestUsefulConfig, TestAlgoConfig, TestAblateConfig
from Tests.tests_filter import (
    TestExtractTrailingInt, TestFilterTrue, TestFilterName, TestFilterHead, 
    TestFilterNeuron, TestFilterPosition, TestFilterLayer, TestFilterContains,
    TestFilterAttention, TestFilterImpact, TestFilterAlgo, TestFilterAnd,
    TestFilterOr, TestFilterNodes, TestPrintAlgoPurposeResults, TestComplexFilterScenarios
)
from Tests.tests_model_sae import (
    TestSafeMathFunctions, TestAdaptiveSparseAutoencoder, TestSparseAutoencoderConfig,
    TestSparseAutoencoderForHF, TestSaveToHuggingFace, TestIntegration
)


if __name__ == '__main__':
    test_classes_to_run = [
        TestUseful, TestHuggingFace, TestModelConfig, TestUsefulConfig, 
        TestAlgoConfig, TestAblateConfig, TestExtractTrailingInt, TestFilterTrue,
        TestFilterName, TestFilterHead, TestFilterNeuron, TestFilterPosition,
        TestFilterLayer, TestFilterContains, TestFilterAttention, TestFilterImpact,
        TestFilterAlgo, TestFilterAnd, TestFilterOr, TestFilterNodes,
        TestPrintAlgoPurposeResults, TestComplexFilterScenarios,
        TestSafeMathFunctions, TestAdaptiveSparseAutoencoder, TestSparseAutoencoderConfig,
        TestSparseAutoencoderForHF, TestSaveToHuggingFace, TestIntegration
    ]

    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
        
    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
