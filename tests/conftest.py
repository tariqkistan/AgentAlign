import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://fake-endpoint.openai.azure.com")
os.environ.setdefault("AZURE_OPENAI_KEY", "fake-key-for-testing")
os.environ.setdefault("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
os.environ.setdefault("JUDGE_DEPLOYMENT", "gpt-4o")
os.environ.setdefault("LOG_LEVEL", "DEBUG")


@pytest.fixture
def fixture_loader():
    def load_fixture(name: str) -> dict:
        fixture_path = Path(__file__).parent / "fixtures" / f"{name}.json"
        with open(fixture_path) as f:
            return json.load(f)

    return load_fixture
