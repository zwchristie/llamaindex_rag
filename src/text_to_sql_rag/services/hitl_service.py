"""
Human-in-the-Loop (HITL) approval service.
"""

from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging
from datetime import datetime, timedelta
import uuid
import asyncio

from ..models.view_models import HITLRequest, SessionState

logger = logging.getLogger(__name__)


class HITLService:
    """Service for managing human-in-the-loop approval workflows."""
    
    def __init__(self, db: AsyncIOMotorDatabase, timeout_minutes: int = 30):
        self.db = db
        self.hitl_collection = db.hitl_requests
        self.session_collection = db.session_states
        self.timeout_minutes = timeout_minutes
        self._pending_callbacks: Dict[str, asyncio.Event] = {}
    
    async def create_approval_request(
        self,
        session_id: str,
        user_query: str,
        generated_sql: str,
        sql_explanation: str,
        selected_views: List[str]
    ) -> str:
        """Create a new HITL approval request."""
        try:
            request_id = str(uuid.uuid4())
            expires_at = datetime.utcnow() + timedelta(minutes=self.timeout_minutes)
            
            hitl_request = HITLRequest(
                request_id=request_id,
                session_id=session_id,
                user_query=user_query,
                generated_sql=generated_sql,
                sql_explanation=sql_explanation,
                selected_views=selected_views,
                expires_at=expires_at
            )
            
            # Store request in database
            await self.hitl_collection.insert_one(hitl_request.dict())
            
            # Update session state
            await self._update_session_hitl_status(session_id, request_id, "pending")
            
            # Set up callback event
            self._pending_callbacks[request_id] = asyncio.Event()
            
            logger.info(f"Created HITL approval request {request_id} for session {session_id}")
            return request_id
            
        except Exception as e:
            logger.error(f"Error creating HITL request: {e}")
            raise
    
    async def wait_for_approval(self, request_id: str) -> Dict[str, Any]:
        """Wait for approval/rejection of a HITL request."""
        try:
            # Get the event for this request
            if request_id not in self._pending_callbacks:
                # Request might already be resolved, check database
                request = await self.get_request(request_id)
                if request and request.status != "pending":
                    return {
                        "status": request.status,
                        "reviewer_notes": request.reviewer_notes,
                        "resolution_reason": request.resolution_reason
                    }
                else:
                    # Create new event for existing request
                    self._pending_callbacks[request_id] = asyncio.Event()
            
            event = self._pending_callbacks[request_id]
            
            # Wait for approval with timeout
            try:
                await asyncio.wait_for(event.wait(), timeout=self.timeout_minutes * 60)
                
                # Get the resolved request
                request = await self.get_request(request_id)
                if request:
                    return {
                        "status": request.status,
                        "reviewer_notes": request.reviewer_notes,
                        "resolution_reason": request.resolution_reason
                    }
                else:
                    logger.error(f"Request {request_id} not found after resolution")
                    return {"status": "error", "message": "Request not found"}
                    
            except asyncio.TimeoutError:
                # Request timed out, mark as expired
                await self._expire_request(request_id)
                return {"status": "expired", "message": "Request timed out"}
                
            finally:
                # Clean up callback
                self._pending_callbacks.pop(request_id, None)
                
        except Exception as e:
            logger.error(f"Error waiting for approval {request_id}: {e}")
            self._pending_callbacks.pop(request_id, None)
            return {"status": "error", "message": str(e)}
    
    async def approve_request(
        self,
        request_id: str,
        reviewer_notes: Optional[str] = None,
        resolution_reason: Optional[str] = None
    ) -> bool:
        """Approve a HITL request."""
        return await self._resolve_request(request_id, "approved", reviewer_notes, resolution_reason)
    
    async def reject_request(
        self,
        request_id: str,
        reviewer_notes: Optional[str] = None,
        resolution_reason: Optional[str] = None
    ) -> bool:
        """Reject a HITL request."""
        return await self._resolve_request(request_id, "rejected", reviewer_notes, resolution_reason)
    
    async def _resolve_request(
        self,
        request_id: str,
        status: str,
        reviewer_notes: Optional[str],
        resolution_reason: Optional[str]
    ) -> bool:
        """Resolve a HITL request with given status."""
        try:
            # Update request in database
            update_data = {
                "status": status,
                "resolved_at": datetime.utcnow()
            }
            
            if reviewer_notes:
                update_data["reviewer_notes"] = reviewer_notes
            if resolution_reason:
                update_data["resolution_reason"] = resolution_reason
            
            result = await self.hitl_collection.update_one(
                {"request_id": request_id, "status": "pending"},
                {"$set": update_data}
            )
            
            if result.modified_count == 0:
                logger.warning(f"No pending request found with ID {request_id}")
                return False
            
            # Update session state
            request = await self.get_request(request_id)
            if request:
                await self._update_session_hitl_status(request.session_id, request_id, status)
            
            # Notify waiting process
            if request_id in self._pending_callbacks:
                self._pending_callbacks[request_id].set()
            
            logger.info(f"Resolved HITL request {request_id} with status {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving request {request_id}: {e}")
            raise
    
    async def _expire_request(self, request_id: str):
        """Mark a request as expired due to timeout."""
        try:
            await self.hitl_collection.update_one(
                {"request_id": request_id, "status": "pending"},
                {
                    "$set": {
                        "status": "expired",
                        "resolved_at": datetime.utcnow(),
                        "resolution_reason": "Request timed out"
                    }
                }
            )
            
            # Update session state
            request = await self.get_request(request_id)
            if request:
                await self._update_session_hitl_status(request.session_id, request_id, "expired")
            
            logger.info(f"Expired HITL request {request_id}")
            
        except Exception as e:
            logger.error(f"Error expiring request {request_id}: {e}")
    
    async def get_request(self, request_id: str) -> Optional[HITLRequest]:
        """Get a HITL request by ID."""
        try:
            doc = await self.hitl_collection.find_one({"request_id": request_id})
            if doc:
                return HITLRequest(**doc)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving request {request_id}: {e}")
            return None
    
    async def get_pending_requests(self, limit: int = 50) -> List[HITLRequest]:
        """Get all pending HITL requests."""
        try:
            cursor = self.hitl_collection.find(
                {"status": "pending"}
            ).sort("created_at", 1).limit(limit)
            
            requests = []
            async for doc in cursor:
                requests.append(HITLRequest(**doc))
            
            return requests
            
        except Exception as e:
            logger.error(f"Error retrieving pending requests: {e}")
            return []
    
    async def get_requests_for_session(self, session_id: str) -> List[HITLRequest]:
        """Get all HITL requests for a session."""
        try:
            cursor = self.hitl_collection.find(
                {"session_id": session_id}
            ).sort("created_at", -1)
            
            requests = []
            async for doc in cursor:
                requests.append(HITLRequest(**doc))
            
            return requests
            
        except Exception as e:
            logger.error(f"Error retrieving requests for session {session_id}: {e}")
            return []
    
    async def _update_session_hitl_status(self, session_id: str, request_id: str, status: str):
        """Update session state with HITL information."""
        try:
            await self.session_collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "hitl_request_id": request_id,
                        "hitl_status": status,
                        "updated_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Error updating session {session_id} HITL status: {e}")
    
    async def cleanup_expired_requests(self):
        """Clean up expired requests (run periodically)."""
        try:
            cutoff_time = datetime.utcnow()
            
            # Find and expire old pending requests
            result = await self.hitl_collection.update_many(
                {
                    "status": "pending",
                    "expires_at": {"$lt": cutoff_time}
                },
                {
                    "$set": {
                        "status": "expired",
                        "resolved_at": cutoff_time,
                        "resolution_reason": "Automatic expiration"
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Expired {result.modified_count} old HITL requests")
            
            # Clean up callback events for expired requests
            expired_callbacks = []
            for request_id, event in self._pending_callbacks.items():
                request = await self.get_request(request_id)
                if not request or request.status != "pending":
                    expired_callbacks.append(request_id)
            
            for request_id in expired_callbacks:
                event = self._pending_callbacks.pop(request_id, None)
                if event:
                    event.set()  # Unblock any waiting processes
            
            if expired_callbacks:
                logger.info(f"Cleaned up {len(expired_callbacks)} callback events")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get HITL service statistics."""
        try:
            total_requests = await self.hitl_collection.count_documents({})
            pending_requests = await self.hitl_collection.count_documents({"status": "pending"})
            approved_requests = await self.hitl_collection.count_documents({"status": "approved"})
            rejected_requests = await self.hitl_collection.count_documents({"status": "rejected"})
            expired_requests = await self.hitl_collection.count_documents({"status": "expired"})
            
            return {
                "total_requests": total_requests,
                "pending_requests": pending_requests,
                "approved_requests": approved_requests,
                "rejected_requests": rejected_requests,
                "expired_requests": expired_requests,
                "active_callbacks": len(self._pending_callbacks)
            }
            
        except Exception as e:
            logger.error(f"Error getting HITL stats: {e}")
            return {}