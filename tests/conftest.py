"""
Pytest configuration and shared fixtures.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src to path for tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may require external services)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from text_to_sql_rag.config.new_settings import Settings
    
    return Settings(
        mongodb_url="mongodb://test:test@localhost:27017",
        mongodb_database="test_db",
        opensearch_host="localhost",
        opensearch_port=9200,
        opensearch_index_name="test_index",
        bedrock_endpoint_url="https://api.test.com",
        bedrock_llm_model="test-llm",
        bedrock_embedding_model="test-embedding",
        hitl_timeout_minutes=5
    )