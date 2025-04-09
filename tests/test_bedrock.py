import llm
from llm.plugins import pm
import os
import pytest

API_KEY = os.environ.get("PYTEST_BEDROCK_API_KEY") or "fake-access:fake-secret"


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization"]}


@pytest.mark.vcr()
def test_prompt():
    model = llm.get_model("us.amazon.nova-micro-v1:0")
    model.key = API_KEY
    response = model.prompt("What is the capital of France?", system="One word answer")
    assert response.text() == "Paris"


@pytest.mark.vcr()
def test_conversation():
    model = llm.get_model("us.amazon.nova-lite-v1:0")
    model.key = API_KEY
    conversation = model.conversation()
    response = conversation.prompt(
        "What is the capital of France?", system="One word answer"
    )
    assert response.text() == "Paris"
    response2 = conversation.prompt("Germany?", system="One word answer")
    assert response2.text() == "Berlin"
