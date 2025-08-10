"""
New simplified FastAPI application for text-to-SQL system without domain concepts.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager

from ..models.view_models import ViewMetadata, HITLRequest, SessionState
from ..services.view_service import ViewService
from ..services.embedding_service import EmbeddingService, VectorService
from ..services.hitl_service import HITLService
from ..services.session_service import SessionService
from ..services.bedrock_endpoint_service import BedrockEndpointService
from ..core.text_to_sql_agent import TextToSQLAgent
from ..config.new_settings import get_settings

import motor.motor_asyncio
from opensearchpy import AsyncOpenSearch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
app_services = {}


async def initialize_services():
    """Initialize all required services."""
    try:
        settings = get_settings()
        
        # MongoDB connection
        client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongodb_url)
        db = client[settings.mongodb_database]
        
        # OpenSearch connection
        opensearch_client = AsyncOpenSearch(
            hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
            http_auth=None,
            use_ssl=settings.opensearch_use_ssl,
            verify_certs=settings.opensearch_verify_certs,
        )
        
        # Initialize services
        view_service = ViewService(db)
        embedding_service = EmbeddingService(settings.bedrock_endpoint_url, settings.bedrock_embedding_model)
        vector_service = VectorService(opensearch_client, settings.opensearch_index_name, settings.opensearch_vector_field)
        hitl_service = HITLService(db, settings.hitl_timeout_minutes)
        session_service = SessionService(db)
        llm_service = BedrockEndpointService(settings.bedrock_endpoint_url, settings.bedrock_llm_model)
        
        # Create indexes
        await view_service.ensure_indexes()
        await session_service.ensure_indexes()
        
        # Initialize text-to-SQL agent
        agent = TextToSQLAgent(
            view_service=view_service,
            embedding_service=embedding_service,
            vector_service=vector_service,
            hitl_service=hitl_service,
            llm_service=llm_service,
            session_service=session_service
        )
        
        # Store services globally
        app_services.update({
            "db": db,
            "view_service": view_service,
            "embedding_service": embedding_service,
            "vector_service": vector_service,
            "hitl_service": hitl_service,
            "session_service": session_service,
            "llm_service": llm_service,
            "agent": agent,
            "mongodb_client": client,
            "opensearch_client": opensearch_client
        })
        
        logger.info("All services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return False


async def cleanup_services():
    """Cleanup services on shutdown."""
    try:
        if "mongodb_client" in app_services:
            app_services["mongodb_client"].close()
        
        if "opensearch_client" in app_services:
            await app_services["opensearch_client"].close()
        
        logger.info("Services cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    success = await initialize_services()
    if not success:
        logger.error("Failed to initialize application")
        
    yield
    
    # Shutdown
    await cleanup_services()


# Initialize FastAPI app
app = FastAPI(
    title="Text-to-SQL RAG API",
    description="Simplified text-to-SQL system with HITL approval",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Request/Response models
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class HITLApprovalRequest(BaseModel):
    request_id: str
    action: str  # "approve" or "reject"
    notes: Optional[str] = None
    reason: Optional[str] = None


# Dependency injection
def get_view_service() -> ViewService:
    return app_services["view_service"]


def get_agent() -> TextToSQLAgent:
    return app_services["agent"]


def get_hitl_service() -> HITLService:
    return app_services["hitl_service"]


def get_session_service() -> SessionService:
    return app_services["session_service"]


# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not app_services:
        return {"status": "starting", "message": "Services are initializing"}
    
    try:
        # Check MongoDB
        await app_services["db"].command("ping")
        mongodb_status = "healthy"
    except:
        mongodb_status = "unhealthy"
    
    try:
        # Check OpenSearch
        await app_services["opensearch_client"].ping()
        opensearch_status = "healthy"
    except:
        opensearch_status = "unhealthy"
    
    return {
        "status": "healthy" if mongodb_status == "healthy" and opensearch_status == "healthy" else "degraded",
        "services": {
            "mongodb": mongodb_status,
            "opensearch": opensearch_status,
            "agent": "initialized" if "agent" in app_services else "not_initialized"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/stats")
async def get_stats(view_service: ViewService = Depends(get_view_service)):
    """Get system statistics."""
    try:
        view_stats = await view_service.get_stats()
        hitl_stats = await app_services["hitl_service"].get_stats()
        session_stats = await app_services["session_service"].get_stats()
        
        try:
            index_stats = await app_services["vector_service"].get_index_stats()
        except:
            index_stats = {"error": "Could not retrieve index stats"}
        
        return {
            "views": view_stats,
            "hitl": hitl_stats,
            "sessions": session_stats,
            "opensearch": index_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# Main query endpoint
@app.post("/query")
async def process_query(
    request: QueryRequest,
    agent: TextToSQLAgent = Depends(get_agent)
):
    """Process a text-to-SQL query with HITL approval."""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        result = await agent.process_query(request.query, request.session_id)
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


# HITL endpoints
@app.get("/hitl/requests")
async def get_pending_hitl_requests(
    hitl_service: HITLService = Depends(get_hitl_service)
):
    """Get all pending HITL approval requests."""
    try:
        requests = await hitl_service.get_pending_requests()
        return {
            "requests": [req.dict() for req in requests],
            "count": len(requests)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get HITL requests: {str(e)}")


@app.get("/hitl/requests/{request_id}")
async def get_hitl_request(
    request_id: str,
    hitl_service: HITLService = Depends(get_hitl_service)
):
    """Get a specific HITL request."""
    try:
        request = await hitl_service.get_request(request_id)
        if not request:
            raise HTTPException(status_code=404, detail="HITL request not found")
        
        return request.dict()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get HITL request: {str(e)}")


@app.post("/hitl/resolve")
async def resolve_hitl_request(
    request: HITLApprovalRequest,
    hitl_service: HITLService = Depends(get_hitl_service)
):
    """Approve or reject a HITL request."""
    try:
        if request.action not in ["approve", "reject"]:
            raise HTTPException(status_code=400, detail="Action must be 'approve' or 'reject'")
        
        if request.action == "approve":
            success = await hitl_service.approve_request(
                request.request_id, 
                request.notes, 
                request.reason
            )
        else:
            success = await hitl_service.reject_request(
                request.request_id, 
                request.notes, 
                request.reason
            )
        
        if success:
            return {
                "success": True,
                "message": f"Request {request.action}d successfully",
                "request_id": request.request_id
            }
        else:
            raise HTTPException(status_code=404, detail="HITL request not found or already resolved")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve HITL request: {str(e)}")


# View management endpoints
@app.get("/views")
async def get_views(
    view_type: Optional[str] = None,
    view_service: ViewService = Depends(get_view_service)
):
    """Get all views, optionally filtered by type."""
    try:
        views = await view_service.get_all_views(view_type)
        return {
            "views": [view.dict() for view in views],
            "count": len(views),
            "filtered_by_type": view_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get views: {str(e)}")


@app.get("/views/{view_name}")
async def get_view(
    view_name: str,
    view_service: ViewService = Depends(get_view_service)
):
    """Get a specific view by name."""
    try:
        view = await view_service.get_view_by_name(view_name)
        if not view:
            raise HTTPException(status_code=404, detail="View not found")
        
        return view.dict()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get view: {str(e)}")


@app.post("/views/search")
async def search_views(
    request: Dict[str, Any],
    view_service: ViewService = Depends(get_view_service)
):
    """Search views by text."""
    try:
        search_text = request.get("query", "")
        limit = request.get("limit", 10)
        
        if not search_text:
            raise HTTPException(status_code=400, detail="Search query cannot be empty")
        
        views = await view_service.search_views_by_text(search_text, limit)
        return {
            "query": search_text,
            "views": [view.dict() for view in views],
            "count": len(views)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Session management endpoints
@app.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service)
):
    """Get session state."""
    try:
        session = await session_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session.dict()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@app.get("/sessions")
async def get_active_sessions(
    limit: int = 20,
    session_service: SessionService = Depends(get_session_service)
):
    """Get active sessions."""
    try:
        sessions = await session_service.get_active_sessions(limit)
        return {
            "sessions": [session.dict() for session in sessions],
            "count": len(sessions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")


# Admin/utility endpoints
@app.post("/admin/reindex")
async def reindex_views():
    """Reindex all views in OpenSearch (admin endpoint)."""
    try:
        view_service = app_services["view_service"]
        vector_service = app_services["vector_service"]
        embedding_service = app_services["embedding_service"]
        
        # Get all views
        views = await view_service.get_all_views()
        if not views:
            return {"message": "No views found to reindex"}
        
        # Reindex
        await vector_service.reindex_all_views(views, embedding_service)
        
        return {
            "message": f"Successfully reindexed {len(views)} views",
            "reindexed_count": len(views)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")


@app.post("/admin/cleanup")
async def cleanup_old_data():
    """Cleanup old sessions and expired HITL requests."""
    try:
        session_service = app_services["session_service"]
        hitl_service = app_services["hitl_service"]
        
        # Cleanup old sessions (7 days)
        deleted_sessions = await session_service.cleanup_old_sessions(7)
        
        # Cleanup expired HITL requests
        await hitl_service.cleanup_expired_requests()
        
        return {
            "message": "Cleanup completed",
            "deleted_sessions": deleted_sessions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


# Development/testing endpoints
@app.get("/dev/test-embedding")
async def test_embedding(text: str = "test query"):
    """Test embedding generation (dev endpoint)."""
    try:
        embedding_service = app_services["embedding_service"]
        embedding = await embedding_service.get_embedding(text)
        
        return {
            "text": text,
            "embedding_dimension": len(embedding),
            "embedding_sample": embedding[:5]  # First 5 dimensions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding test failed: {str(e)}")


@app.get("/dev/test-llm")
async def test_llm(prompt: str = "What is SQL?"):
    """Test LLM service (dev endpoint)."""
    try:
        llm_service = app_services["llm_service"]
        response = await llm_service.generate_text(prompt)
        
        return {
            "prompt": prompt,
            "response": response,
            "response_length": len(response)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM test failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "src.text_to_sql_rag.api.new_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )