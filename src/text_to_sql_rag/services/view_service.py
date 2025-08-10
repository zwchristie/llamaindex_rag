"""
Simplified view metadata service without domain concepts.
"""

from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import logging
from datetime import datetime

from ..models.view_models import ViewMetadata, ViewColumn, ViewJoin

logger = logging.getLogger(__name__)


class ViewService:
    """Service for managing view metadata without domain concepts."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db.view_metadata
    
    async def create_view(self, view_metadata: ViewMetadata) -> str:
        """Create a new view metadata document."""
        try:
            # Generate full text for embedding
            view_metadata.full_text = view_metadata.generate_full_text()
            view_metadata.updated_at = datetime.utcnow()
            
            result = await self.collection.insert_one(view_metadata.dict())
            logger.info(f"Created view metadata for {view_metadata.view_name}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error creating view {view_metadata.view_name}: {e}")
            raise
    
    async def get_view_by_name(self, view_name: str) -> Optional[ViewMetadata]:
        """Get view metadata by view name."""
        try:
            doc = await self.collection.find_one({"view_name": view_name})
            if doc:
                return ViewMetadata(**doc)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving view {view_name}: {e}")
            raise
    
    async def get_all_views(self, view_type: Optional[str] = None) -> List[ViewMetadata]:
        """Get all view metadata, optionally filtered by type."""
        try:
            query = {}
            if view_type:
                query["view_type"] = view_type
                
            cursor = self.collection.find(query)
            views = []
            async for doc in cursor:
                views.append(ViewMetadata(**doc))
                
            logger.info(f"Retrieved {len(views)} views" + (f" of type {view_type}" if view_type else ""))
            return views
            
        except Exception as e:
            logger.error(f"Error retrieving views: {e}")
            raise
    
    async def update_view(self, view_name: str, updates: Dict[str, Any]) -> bool:
        """Update view metadata."""
        try:
            updates["updated_at"] = datetime.utcnow()
            
            # If structural changes, regenerate full_text
            if any(key in updates for key in ["description", "columns", "joins", "use_cases"]):
                # Get current view to regenerate full text
                current_view = await self.get_view_by_name(view_name)
                if current_view:
                    # Apply updates to current view
                    for key, value in updates.items():
                        setattr(current_view, key, value)
                    updates["full_text"] = current_view.generate_full_text()
            
            result = await self.collection.update_one(
                {"view_name": view_name},
                {"$set": updates}
            )
            
            if result.modified_count > 0:
                logger.info(f"Updated view metadata for {view_name}")
                return True
            else:
                logger.warning(f"No updates made for view {view_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating view {view_name}: {e}")
            raise
    
    async def delete_view(self, view_name: str) -> bool:
        """Delete view metadata."""
        try:
            result = await self.collection.delete_one({"view_name": view_name})
            if result.deleted_count > 0:
                logger.info(f"Deleted view metadata for {view_name}")
                return True
            else:
                logger.warning(f"View {view_name} not found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting view {view_name}: {e}")
            raise
    
    async def search_views_by_text(self, search_text: str, limit: int = 10) -> List[ViewMetadata]:
        """Search views by text content (basic text search)."""
        try:
            # MongoDB text search on description, view_name, and full_text
            query = {
                "$text": {
                    "$search": search_text
                }
            }
            
            cursor = self.collection.find(query).limit(limit)
            views = []
            async for doc in cursor:
                views.append(ViewMetadata(**doc))
                
            logger.info(f"Text search returned {len(views)} views for query: {search_text}")
            return views
            
        except Exception as e:
            logger.error(f"Error in text search: {e}")
            # Fallback to basic regex search
            return await self._fallback_text_search(search_text, limit)
    
    async def _fallback_text_search(self, search_text: str, limit: int) -> List[ViewMetadata]:
        """Fallback text search using regex."""
        try:
            query = {
                "$or": [
                    {"view_name": {"$regex": search_text, "$options": "i"}},
                    {"description": {"$regex": search_text, "$options": "i"}},
                    {"use_cases": {"$regex": search_text, "$options": "i"}},
                    {"full_text": {"$regex": search_text, "$options": "i"}}
                ]
            }
            
            cursor = self.collection.find(query).limit(limit)
            views = []
            async for doc in cursor:
                views.append(ViewMetadata(**doc))
                
            return views
            
        except Exception as e:
            logger.error(f"Error in fallback text search: {e}")
            return []
    
    async def get_views_by_type(self, view_type: str) -> List[ViewMetadata]:
        """Get all views of a specific type."""
        return await self.get_all_views(view_type=view_type)
    
    async def ensure_indexes(self):
        """Ensure proper indexes exist for efficient queries."""
        try:
            # Create indexes
            await self.collection.create_index("view_name", unique=True)
            await self.collection.create_index("view_type")
            await self.collection.create_index("updated_at")
            
            # Text index for search
            await self.collection.create_index([
                ("view_name", "text"),
                ("description", "text"),
                ("use_cases", "text"),
                ("full_text", "text")
            ])
            
            logger.info("Created MongoDB indexes for view_metadata collection")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the view collection."""
        try:
            total_views = await self.collection.count_documents({})
            core_views = await self.collection.count_documents({"view_type": "CORE"})
            supporting_views = await self.collection.count_documents({"view_type": "SUPPORTING"})
            
            return {
                "total_views": total_views,
                "core_views": core_views,
                "supporting_views": supporting_views,
                "last_updated": await self._get_latest_update_time()
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    async def _get_latest_update_time(self) -> Optional[datetime]:
        """Get the most recent update time."""
        try:
            cursor = self.collection.find().sort("updated_at", -1).limit(1)
            doc = await cursor.to_list(length=1)
            if doc:
                return doc[0].get("updated_at")
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest update time: {e}")
            return None