"""
Unit tests for HITLService.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
import uuid

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from text_to_sql_rag.models.view_models import HITLRequest
from text_to_sql_rag.services.hitl_service import HITLService


@pytest.fixture
def mock_db():
    """Create a mock database."""
    db = MagicMock()
    hitl_collection = AsyncMock()
    session_collection = AsyncMock()
    db.hitl_requests = hitl_collection
    db.session_states = session_collection
    return db


@pytest.fixture
def hitl_service(mock_db):
    """Create a HITLService with mock database."""
    return HITLService(mock_db, timeout_minutes=5)  # Short timeout for tests


@pytest.fixture
def sample_hitl_request():
    """Create a sample HITL request for testing."""
    return HITLRequest(
        request_id="test-request-id",
        session_id="test-session-id",
        user_query="Show me all users",
        generated_sql="SELECT * FROM users",
        sql_explanation="This query selects all columns from the users table",
        selected_views=["V_USERS"],
        expires_at=datetime.utcnow() + timedelta(minutes=30)
    )


@pytest.mark.asyncio
async def test_create_approval_request(hitl_service):
    """Test creating a HITL approval request."""
    # Mock database insert
    hitl_service.hitl_collection.insert_one.return_value = AsyncMock()
    hitl_service._update_session_hitl_status = AsyncMock()
    
    request_id = await hitl_service.create_approval_request(
        session_id="test-session",
        user_query="Test query",
        generated_sql="SELECT 1",
        sql_explanation="Test explanation",
        selected_views=["TEST_VIEW"]
    )
    
    assert request_id is not None
    assert len(request_id) > 0
    hitl_service.hitl_collection.insert_one.assert_called_once()
    hitl_service._update_session_hitl_status.assert_called_once()


@pytest.mark.asyncio
async def test_get_request_found(hitl_service, sample_hitl_request):
    """Test getting an existing HITL request."""
    # Mock database find
    hitl_service.hitl_collection.find_one.return_value = sample_hitl_request.dict()
    
    result = await hitl_service.get_request("test-request-id")
    
    assert result is not None
    assert result.request_id == "test-request-id"
    assert result.user_query == "Show me all users"


@pytest.mark.asyncio
async def test_get_request_not_found(hitl_service):
    """Test getting a non-existent HITL request."""
    # Mock database find returning None
    hitl_service.hitl_collection.find_one.return_value = None
    
    result = await hitl_service.get_request("nonexistent")
    
    assert result is None


@pytest.mark.asyncio
async def test_approve_request_success(hitl_service, sample_hitl_request):
    """Test successful request approval."""
    # Mock successful update
    mock_result = MagicMock()
    mock_result.modified_count = 1
    hitl_service.hitl_collection.update_one.return_value = mock_result
    hitl_service.get_request = AsyncMock(return_value=sample_hitl_request)
    hitl_service._update_session_hitl_status = AsyncMock()
    
    result = await hitl_service.approve_request(
        "test-request-id",
        reviewer_notes="Looks good",
        resolution_reason="Approved by reviewer"
    )
    
    assert result is True
    hitl_service.hitl_collection.update_one.assert_called_once()


@pytest.mark.asyncio
async def test_reject_request_success(hitl_service, sample_hitl_request):
    """Test successful request rejection."""
    # Mock successful update
    mock_result = MagicMock()
    mock_result.modified_count = 1
    hitl_service.hitl_collection.update_one.return_value = mock_result
    hitl_service.get_request = AsyncMock(return_value=sample_hitl_request)
    hitl_service._update_session_hitl_status = AsyncMock()
    
    result = await hitl_service.reject_request(
        "test-request-id",
        reviewer_notes="Needs improvement",
        resolution_reason="SQL syntax error"
    )
    
    assert result is True
    hitl_service.hitl_collection.update_one.assert_called_once()


@pytest.mark.asyncio
async def test_approve_request_not_found(hitl_service):
    """Test approving a non-existent or already resolved request."""
    # Mock no update
    mock_result = MagicMock()
    mock_result.modified_count = 0
    hitl_service.hitl_collection.update_one.return_value = mock_result
    
    result = await hitl_service.approve_request("nonexistent")
    
    assert result is False


@pytest.mark.asyncio
async def test_get_pending_requests(hitl_service, sample_hitl_request):
    """Test getting pending requests."""
    # Mock database cursor
    mock_cursor = AsyncMock()
    mock_cursor.__aiter__ = AsyncMock(return_value=iter([sample_hitl_request.dict()]))
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.limit.return_value = mock_cursor
    hitl_service.hitl_collection.find.return_value = mock_cursor
    
    result = await hitl_service.get_pending_requests()
    
    assert len(result) == 1
    assert result[0].request_id == "test-request-id"
    assert result[0].status == "pending"


@pytest.mark.asyncio
async def test_get_requests_for_session(hitl_service, sample_hitl_request):
    """Test getting requests for a specific session."""
    # Mock database cursor
    mock_cursor = AsyncMock()
    mock_cursor.__aiter__ = AsyncMock(return_value=iter([sample_hitl_request.dict()]))
    mock_cursor.sort.return_value = mock_cursor
    hitl_service.hitl_collection.find.return_value = mock_cursor
    
    result = await hitl_service.get_requests_for_session("test-session-id")
    
    assert len(result) == 1
    assert result[0].session_id == "test-session-id"


@pytest.mark.asyncio
async def test_cleanup_expired_requests(hitl_service):
    """Test cleaning up expired requests."""
    # Mock successful update of expired requests
    mock_result = MagicMock()
    mock_result.modified_count = 2
    hitl_service.hitl_collection.update_many.return_value = mock_result
    
    await hitl_service.cleanup_expired_requests()
    
    hitl_service.hitl_collection.update_many.assert_called_once()
    
    # Verify the query filters for expired pending requests
    call_args = hitl_service.hitl_collection.update_many.call_args[0]
    query = call_args[0]
    assert query["status"] == "pending"
    assert "$lt" in query["expires_at"]


@pytest.mark.asyncio
async def test_get_stats(hitl_service):
    """Test statistics retrieval."""
    # Mock count queries
    hitl_service.hitl_collection.count_documents = AsyncMock()
    hitl_service.hitl_collection.count_documents.side_effect = [
        10,  # total
        2,   # pending
        6,   # approved
        1,   # rejected
        1    # expired
    ]
    
    result = await hitl_service.get_stats()
    
    assert result["total_requests"] == 10
    assert result["pending_requests"] == 2
    assert result["approved_requests"] == 6
    assert result["rejected_requests"] == 1
    assert result["expired_requests"] == 1
    assert "active_callbacks" in result


@pytest.mark.asyncio
async def test_wait_for_approval_immediate_resolution(hitl_service, sample_hitl_request):
    """Test waiting for approval when request is already resolved."""
    # Mock request that's already approved
    approved_request = sample_hitl_request.copy()
    approved_request.status = "approved"
    approved_request.reviewer_notes = "Approved"
    
    hitl_service.get_request = AsyncMock(return_value=approved_request)
    
    result = await hitl_service.wait_for_approval("test-request-id")
    
    assert result["status"] == "approved"
    assert result["reviewer_notes"] == "Approved"


@pytest.mark.asyncio
async def test_wait_for_approval_timeout(hitl_service):
    """Test wait for approval timeout behavior."""
    # Create a service with very short timeout
    short_timeout_service = HITLService(hitl_service.db, timeout_minutes=0.001)  # Very short
    short_timeout_service.get_request = AsyncMock(return_value=None)
    short_timeout_service._expire_request = AsyncMock()
    
    result = await short_timeout_service.wait_for_approval("test-request-id")
    
    assert result["status"] == "expired"
    assert "timed out" in result["message"]


def test_hitl_request_model():
    """Test HITLRequest model validation."""
    request = HITLRequest(
        request_id="test-id",
        session_id="test-session",
        user_query="Test query",
        generated_sql="SELECT 1",
        sql_explanation="Test",
        selected_views=["VIEW1"],
        expires_at=datetime.utcnow() + timedelta(hours=1)
    )
    
    assert request.request_id == "test-id"
    assert request.status == "pending"  # Default value
    assert request.resolved_at is None  # Default value
    
    # Test dict serialization
    request_dict = request.dict()
    assert "request_id" in request_dict
    assert "created_at" in request_dict


def test_hitl_request_expires_at_validation():
    """Test that HITLRequest requires expires_at."""
    with pytest.raises(ValueError):
        HITLRequest(
            request_id="test-id",
            session_id="test-session",
            user_query="Test query",
            generated_sql="SELECT 1",
            sql_explanation="Test",
            selected_views=["VIEW1"]
            # Missing expires_at - should raise error
        )