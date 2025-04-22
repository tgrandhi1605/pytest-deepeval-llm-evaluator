import os
from enum import Enum

import pytest

# DeepEval use the OpenAI API key to access the LLMs under the hood for the 'model' param that you pass
os.environ["OPENAI_API_KEY"] = "Enter your OpenAI API key here"  # Replace with your OpenAI API key

# DeepEval use the DEEPEVAL_APP_TOKEN to access the DeepEval cloud dashboard to publish the metric results and generated synthetic test sets
os.environ["DEEPEVAL_APP_TOKEN"] = "Enter your DEEPEVAL app token here"  # Replace with your DeepEval app token

class ModelName(Enum):
    GPT_4 = "gpt-4"
    GPT_3_5 = "gpt-3.5"
    CUSTOM_MODEL = "custom-model"

@pytest.fixture
def model_name():
    # Default to GPT-4
    return ModelName.GPT_4.value

@pytest.fixture
def expected_threshold(request):
    test_data = request.param
    return test_data["expected_threshold"]