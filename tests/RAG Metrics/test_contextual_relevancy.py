import pytest

from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

from utils.LLMUtils import get_response_from_llm
from utils.TestDataUtils import load_data_sets


@pytest.mark.parametrize("build_test_case",
                         load_data_sets("ContextualRelevancyDataFeed.json"),
                         indirect=True)
def test_contextual_relevancy(model_name, build_test_case, expected_threshold):
    contextual_relevancy_metric = ContextualRelevancyMetric(threshold=expected_threshold,
                                                            model=model_name,
                                                            include_reason=True)
    contextual_relevancy_metric.measure(build_test_case)
    assert contextual_relevancy_metric.score >= expected_threshold, (
        f"Score {contextual_relevancy_metric.score:.2f} below threshold {expected_threshold}. Reason: {contextual_relevancy_metric.reason}"
    )


@pytest.fixture
def build_test_case(request):
    test_data = request.param

    response_from_rag_llm = get_response_from_llm(test_data)
    # Extract the response and retrieved context
    retrieved_contexts = [doc["page_content"] for doc in response_from_rag_llm["retrieved_docs"]]

    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=response_from_rag_llm["answer"],
        retrieval_context=retrieved_contexts
    )

    return test_case


