"""
Session management service for agent state persistence.
"""

from typing import Optional, List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging
from datetime import datetime

from ..models.view_models import SessionState

logger = logging.getLogger(__name__)


class SessionService:
    """Service for managing agent session states."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db.session_states
    
    async def save_session(self, session_state: SessionState) -> bool:
        """Save or update a session state."""
        try:
            session_state.updated_at = datetime.utcnow()
            
            result = await self.collection.replace_one(
                {"session_id": session_state.session_id},
                session_state.dict(),
                upsert=True
            )
            
            if result.upserted_id or result.modified_count > 0:
                logger.info(f"Saved session state for {session_state.session_id}")
                return True
            else:
                logger.warning(f"No changes made to session {session_state.session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving session {session_state.session_id}: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session state by ID."""
        try:
            doc = await self.collection.find_one({"session_id": session_id})
            if doc:
                return SessionState(**doc)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            return None
    
    async def get_active_sessions(self, limit: int = 50) -> List[SessionState]:
        """Get recently active sessions."""
        try:
            cursor = self.collection.find().sort("updated_at", -1).limit(limit)
            sessions = []
            
            async for doc in cursor:
                sessions.append(SessionState(**doc))
                
            return sessions
            
        except Exception as e:
            logger.error(f"Error retrieving active sessions: {e}")
            return []
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session state."""
        try:
            result = await self.collection.delete_one({"session_id": session_id})
            
            if result.deleted_count > 0:
                logger.info(f"Deleted session {session_id}")
                return True
            else:
                logger.warning(f"Session {session_id} not found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            raise
    
    async def cleanup_old_sessions(self, days_old: int = 7) -> int:
        """Clean up old sessions (run periodically)."""
        try:
            cutoff_date = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=days_old)
            
            result = await self.collection.delete_many(
                {"updated_at": {"$lt": cutoff_date}}
            )
            
            if result.deleted_count > 0:
                logger.info(f"Cleaned up {result.deleted_count} old sessions")
                
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
            return 0
    
    async def update_session_step(self, session_id: str, step: str, data: Dict[str, Any] = None) -> bool:
        """Update current step and optional data for a session."""
        try:
            update_data = {
                "current_step": step,
                "updated_at": datetime.utcnow()
            }
            
            if data:
                update_data.update(data)
            
            result = await self.collection.update_one(
                {"session_id": session_id},
                {"$set": update_data},
                upsert=True
            )
            
            return result.upserted_id is not None or result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating session step {session_id}: {e}")
            return False
    
    async def get_sessions_by_status(self, hitl_status: str) -> List[SessionState]:
        """Get sessions by HITL status."""
        try:
            cursor = self.collection.find({"hitl_status": hitl_status}).sort("updated_at", -1)
            sessions = []
            
            async for doc in cursor:
                sessions.append(SessionState(**doc))
                
            return sessions
            
        except Exception as e:
            logger.error(f"Error retrieving sessions by status {hitl_status}: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        try:
            total_sessions = await self.collection.count_documents({})
            
            # Count by current step
            pipeline = [
                {"$group": {"_id": "$current_step", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            
            step_counts = {}
            async for doc in self.collection.aggregate(pipeline):
                step_counts[doc["_id"]] = doc["count"]
            
            # Count by HITL status
            hitl_pipeline = [
                {"$group": {"_id": "$hitl_status", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            
            hitl_counts = {}
            async for doc in self.collection.aggregate(hitl_pipeline):
                if doc["_id"]:  # Skip null values
                    hitl_counts[doc["_id"]] = doc["count"]
            
            return {
                "total_sessions": total_sessions,
                "sessions_by_step": step_counts,
                "sessions_by_hitl_status": hitl_counts
            }
            
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {}
    
    async def ensure_indexes(self):
        """Ensure proper indexes exist for efficient queries."""
        try:
            await self.collection.create_index("session_id", unique=True)
            await self.collection.create_index("updated_at")
            await self.collection.create_index("current_step")
            await self.collection.create_index("hitl_status")
            
            logger.info("Created MongoDB indexes for session_states collection")
            
        except Exception as e:
            logger.error(f"Error creating session indexes: {e}")
            raise