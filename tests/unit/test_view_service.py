"""
Unit tests for ViewService.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from motor.motor_asyncio import AsyncIOMotorDatabase

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from text_to_sql_rag.models.view_models import ViewMetadata, ViewColumn, ViewJoin
from text_to_sql_rag.services.view_service import ViewService


@pytest.fixture
def mock_db():
    """Create a mock database."""
    db = MagicMock(spec=AsyncIOMotorDatabase)
    collection = AsyncMock()
    db.view_metadata = collection
    return db


@pytest.fixture
def view_service(mock_db):
    """Create a ViewService with mock database."""
    return ViewService(mock_db)


@pytest.fixture
def sample_view():
    """Create a sample ViewMetadata for testing."""
    return ViewMetadata(
        view_name="TEST_VIEW",
        view_type="CORE",
        description="Test view description",
        use_cases="Testing purposes",
        columns=[
            ViewColumn(name="id", type="NUMBER", notNull=True),
            ViewColumn(name="name", type="VARCHAR2", notNull=False)
        ],
        joins=[
            ViewJoin(table_name="USERS", join_type="INNER", join_condition="t.user_id = u.id")
        ]
    )


@pytest.mark.asyncio
async def test_create_view_success(view_service, sample_view):
    """Test successful view creation."""
    # Mock the database insert
    mock_result = MagicMock()
    mock_result.inserted_id = "mock_id"
    view_service.collection.insert_one.return_value = mock_result
    
    result = await view_service.create_view(sample_view)
    
    assert result == "mock_id"
    view_service.collection.insert_one.assert_called_once()
    
    # Verify full_text was generated
    call_args = view_service.collection.insert_one.call_args[0][0]
    assert "full_text" in call_args
    assert call_args["full_text"] is not None


@pytest.mark.asyncio
async def test_get_view_by_name_found(view_service, sample_view):
    """Test getting an existing view by name."""
    # Mock the database find
    view_service.collection.find_one.return_value = sample_view.dict()
    
    result = await view_service.get_view_by_name("TEST_VIEW")
    
    assert result is not None
    assert result.view_name == "TEST_VIEW"
    view_service.collection.find_one.assert_called_once_with({"view_name": "TEST_VIEW"})


@pytest.mark.asyncio
async def test_get_view_by_name_not_found(view_service):
    """Test getting a non-existent view by name."""
    # Mock the database find returning None
    view_service.collection.find_one.return_value = None
    
    result = await view_service.get_view_by_name("NONEXISTENT")
    
    assert result is None


@pytest.mark.asyncio
async def test_get_all_views_no_filter(view_service, sample_view):
    """Test getting all views without type filter."""
    # Mock the database cursor
    mock_cursor = AsyncMock()
    mock_cursor.__aiter__ = AsyncMock(return_value=iter([sample_view.dict()]))
    view_service.collection.find.return_value = mock_cursor
    
    result = await view_service.get_all_views()
    
    assert len(result) == 1
    assert result[0].view_name == "TEST_VIEW"
    view_service.collection.find.assert_called_once_with({})


@pytest.mark.asyncio
async def test_get_all_views_with_filter(view_service, sample_view):
    """Test getting views filtered by type."""
    # Mock the database cursor
    mock_cursor = AsyncMock()
    mock_cursor.__aiter__ = AsyncMock(return_value=iter([sample_view.dict()]))
    view_service.collection.find.return_value = mock_cursor
    
    result = await view_service.get_all_views(view_type="CORE")
    
    assert len(result) == 1
    view_service.collection.find.assert_called_once_with({"view_type": "CORE"})


@pytest.mark.asyncio
async def test_update_view_success(view_service):
    """Test successful view update."""
    # Mock successful update
    mock_result = MagicMock()
    mock_result.modified_count = 1
    view_service.collection.update_one.return_value = mock_result
    view_service.get_view_by_name = AsyncMock(return_value=None)  # Skip regeneration
    
    updates = {"description": "Updated description"}
    result = await view_service.update_view("TEST_VIEW", updates)
    
    assert result is True
    view_service.collection.update_one.assert_called_once()


@pytest.mark.asyncio
async def test_update_view_with_regeneration(view_service, sample_view):
    """Test view update that requires full_text regeneration."""
    # Mock successful update
    mock_result = MagicMock()
    mock_result.modified_count = 1
    view_service.collection.update_one.return_value = mock_result
    view_service.get_view_by_name = AsyncMock(return_value=sample_view)
    
    updates = {"description": "Updated description"}
    result = await view_service.update_view("TEST_VIEW", updates)
    
    assert result is True
    
    # Verify full_text was included in updates
    call_args = view_service.collection.update_one.call_args[0][1]
    assert "full_text" in call_args["$set"]


@pytest.mark.asyncio
async def test_delete_view_success(view_service):
    """Test successful view deletion."""
    # Mock successful deletion
    mock_result = MagicMock()
    mock_result.deleted_count = 1
    view_service.collection.delete_one.return_value = mock_result
    
    result = await view_service.delete_view("TEST_VIEW")
    
    assert result is True
    view_service.collection.delete_one.assert_called_once_with({"view_name": "TEST_VIEW"})


@pytest.mark.asyncio
async def test_delete_view_not_found(view_service):
    """Test deleting a non-existent view."""
    # Mock no deletion
    mock_result = MagicMock()
    mock_result.deleted_count = 0
    view_service.collection.delete_one.return_value = mock_result
    
    result = await view_service.delete_view("NONEXISTENT")
    
    assert result is False


@pytest.mark.asyncio
async def test_search_views_by_text(view_service, sample_view):
    """Test text search functionality."""
    # Mock the database cursor for text search
    mock_cursor = AsyncMock()
    mock_cursor.__aiter__ = AsyncMock(return_value=iter([sample_view.dict()]))
    mock_cursor.limit.return_value = mock_cursor
    view_service.collection.find.return_value = mock_cursor
    
    result = await view_service.search_views_by_text("test search", 5)
    
    assert len(result) == 1
    assert result[0].view_name == "TEST_VIEW"


@pytest.mark.asyncio
async def test_get_stats(view_service):
    """Test statistics retrieval."""
    # Mock count queries
    view_service.collection.count_documents = AsyncMock()
    view_service.collection.count_documents.side_effect = [10, 7, 3]  # total, core, supporting
    
    # Mock latest update time
    view_service._get_latest_update_time = AsyncMock(return_value=datetime.utcnow())
    
    result = await view_service.get_stats()
    
    assert result["total_views"] == 10
    assert result["core_views"] == 7
    assert result["supporting_views"] == 3
    assert "last_updated" in result


def test_view_metadata_generate_full_text(sample_view):
    """Test ViewMetadata full_text generation."""
    full_text = sample_view.generate_full_text()
    
    # Verify all components are included
    assert "View: TEST_VIEW" in full_text
    assert "Type: CORE" in full_text
    assert "Description: Test view description" in full_text
    assert "Use Cases: Testing purposes" in full_text
    assert "id (NUMBER) NOT NULL" in full_text
    assert "name (VARCHAR2)" in full_text
    assert "INNER JOIN USERS" in full_text


def test_view_metadata_generate_full_text_minimal():
    """Test full_text generation with minimal data."""
    view = ViewMetadata(
        view_name="MINIMAL_VIEW",
        view_type="SUPPORTING",
        columns=[]
    )
    
    full_text = view.generate_full_text()
    
    assert "View: MINIMAL_VIEW" in full_text
    assert "Type: SUPPORTING" in full_text
    assert len(full_text) > 0