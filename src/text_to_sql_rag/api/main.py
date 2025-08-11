"""
Consolidated FastAPI application for text-to-SQL RAG system.
Focused on conversation-based workflows, metadata management, and one-shot queries.
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Optional, Dict, Any, Union
import uuid
import json
from datetime import datetime
from pathlib import Path
import logging
from contextlib import asynccontextmanager
from pydantic import BaseModel

# Core imports
from ..config.settings import settings
from ..core.startup import run_startup_tasks, get_initialized_services
from ..core.langgraph_agent import TextToSQLAgent
from ..services.query_execution_service import QueryExecutionService
from ..services.mongodb_service import MongoDBService
from ..services.vector_service import LlamaIndexVectorService
from ..services.embedding_service import EmbeddingService
from ..services.llm_provider_factory import llm_factory
from ..models.simple_models import (
    DocumentType, SQLGenerationRequest, HealthCheckResponse
)
from ..models.conversation import ConversationState, ConversationStatus
import structlog

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)

# Global services storage
app_services = {}

# In-memory storage for conversations (use Redis in production)
conversations = {}
conversation_threads = {}


class ConversationRequest(BaseModel):
    """Request model for starting conversations."""
    query: str
    context: Optional[Dict[str, Any]] = {}
    session_id: Optional[str] = None


class ContinueConversationRequest(BaseModel):
    """Request model for continuing conversations."""
    message: str
    context: Optional[Dict[str, Any]] = {}


class OneShotRequest(BaseModel):
    """Request model for one-shot SQL generation."""
    query: str
    mode: str = "generate"  # "generate" or "edit"
    existing_sql: Optional[str] = None


class MetadataRequest(BaseModel):
    """Request model for saving/updating metadata."""
    document_type: str  # "view", "report", "lookup", "business_domain"
    data: Dict[str, Any]
    metadata_id: Optional[str] = None  # For updates


class MetadataSearchRequest(BaseModel):
    """Request model for searching metadata."""
    query: Optional[str] = None
    document_type: Optional[str] = None
    limit: int = 20
    offset: int = 0


async def initialize_services():
    """Initialize all required services."""
    try:
        success = await run_startup_tasks()
        if success:
            # Get initialized services
            mongodb_service, vector_service, _ = get_initialized_services()
            
            # Initialize additional services
            query_execution_service = QueryExecutionService()
            sql_agent = TextToSQLAgent(vector_service, query_execution_service)
            
            # Store services globally
            app_services.update({
                "mongodb_service": mongodb_service,
                "vector_service": vector_service,
                "query_execution_service": query_execution_service,
                "sql_agent": sql_agent
            })
            
            logger.info("All services initialized successfully")
            return True
        else:
            logger.error("Failed to initialize services")
            return False
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return False


async def cleanup_services():
    """Cleanup services on shutdown."""
    try:
        if "mongodb_service" in app_services:
            await app_services["mongodb_service"].close()
        logger.info("Services cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    success = await initialize_services()
    if not success:
        logger.error("Failed to initialize application")
    yield
    await cleanup_services()


# Initialize FastAPI app
app = FastAPI(
    title="Text-to-SQL RAG API",
    description="Consolidated API for conversation-based text-to-SQL with metadata management",
    version="3.0.0",
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


# Dependency injection
def get_sql_agent() -> TextToSQLAgent:
    if "sql_agent" not in app_services:
        raise HTTPException(status_code=503, detail="SQL agent not initialized")
    return app_services["sql_agent"]


def get_mongodb_service() -> MongoDBService:
    if "mongodb_service" not in app_services:
        raise HTTPException(status_code=503, detail="MongoDB service not initialized")
    return app_services["mongodb_service"]


def get_vector_service() -> LlamaIndexVectorService:
    if "vector_service" not in app_services:
        raise HTTPException(status_code=503, detail="Vector service not initialized")
    return app_services["vector_service"]


# Note: Hierarchical service removed - using TextToSQLAgent directly


# =============================================================================
# HEALTH & STATUS ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not app_services:
        return {"status": "starting", "message": "Services are initializing"}
    
    try:
        # Check core services
        mongodb_healthy = False
        if "mongodb_service" in app_services:
            mongo_health = app_services["mongodb_service"].health_check()
            mongodb_healthy = mongo_health.get("status") == "connected"
        
        vector_healthy = False
        if "vector_service" in app_services:
            vector_healthy = app_services["vector_service"].health_check()
        
        execution_healthy = False
        if "query_execution_service" in app_services:
            execution_healthy = app_services["query_execution_service"].health_check()
        
        status = "healthy" if all([mongodb_healthy, vector_healthy, execution_healthy]) else "degraded"
        
        return {
            "status": status,
            "services": {
                "mongodb": "connected" if mongodb_healthy else "disconnected",
                "vector_store": "connected" if vector_healthy else "disconnected", 
                "execution_service": "connected" if execution_healthy else "disconnected",
                "sql_agent": "initialized" if "sql_agent" in app_services else "not_initialized"
            },
            "version": "3.0.0",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": f"Health check failed: {str(e)}"}


@app.get("/stats")
async def get_stats():
    """Get application statistics."""
    try:
        stats = {
            "active_conversations": len(conversations),
            "conversation_threads": len(conversation_threads),
            "pending_clarifications": len([c for c in conversations.values() if c.get("status") == "waiting_for_clarification"]),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add service-specific stats
        if "vector_service" in app_services:
            stats["vector_store"] = app_services["vector_service"].get_index_stats()
        
        if "mongodb_service" in app_services:
            mongo_stats = app_services["mongodb_service"].get_stats()
            stats["mongodb"] = mongo_stats
            
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# =============================================================================
# CONVERSATION-BASED TEXT-TO-SQL ENDPOINTS
# =============================================================================

@app.post("/conversations/start")
async def start_conversation(
    request: ConversationRequest,
    sql_agent: TextToSQLAgent = Depends(get_sql_agent)
):
    """Start a new conversation with text-to-SQL generation."""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        conversation_id = request.session_id or str(uuid.uuid4())
        logger.info(f"Starting conversation {conversation_id}")
        
        # Generate SQL using hierarchical context
        result = await sql_agent.generate_sql(request.query, conversation_id)
        
        # Store conversation thread metadata
        conversation_data = {
            "conversation_id": conversation_id,
            "initial_query": request.query,
            "status": result.get("status", "completed"),
            "created_at": datetime.utcnow(),
            "last_interaction": datetime.utcnow(),
            "context": request.context,
            "message_history": [
                {
                    "role": "user",
                    "content": request.query,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message_type": "initial_request"
                }
            ],
            "total_requests": 1,
            "completed_requests": 1 if result.get("response_type") != "clarification_request" else 0
        }
        
        # Handle clarification requests
        if result.get("response_type") == "clarification_request" and result.get("request_id"):
            conversation_data["status"] = "waiting_for_clarification"
            conversation_data["pending_request_id"] = result.get("request_id")
        
        # Store conversation
        conversation_threads[conversation_id] = conversation_data
        conversations[conversation_id] = conversation_data.copy()
        
        return {
            "conversation_id": conversation_id,
            "result": result,
            "conversation_status": conversation_data["status"]
        }
        
    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start conversation: {str(e)}")


@app.post("/conversations/{conversation_id}/continue")
async def continue_conversation(
    conversation_id: str,
    request: ContinueConversationRequest,
    sql_agent: TextToSQLAgent = Depends(get_sql_agent)
):
    """Continue an existing conversation with follow-up or clarification."""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if conversation_id not in conversation_threads:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        conversation_thread = conversation_threads[conversation_id]
        
        # Check if there's a pending clarification request
        pending_request_id = conversation_thread.get("pending_request_id")
        
        if pending_request_id and pending_request_id in sql_agent._checkpoint_storage:
            # Resume from checkpoint for clarification
            logger.info(f"Resuming from checkpoint for request {pending_request_id}")
            result = await sql_agent.continue_from_checkpoint(pending_request_id, request.message)
            
            # Clear pending request
            if "pending_request_id" in conversation_thread:
                del conversation_thread["pending_request_id"]
            conversation_thread["completed_requests"] += 1
            
        else:
            # New request in the conversation thread
            logger.info(f"Starting new request in conversation {conversation_id}")
            result = await sql_agent.generate_sql(request.message, conversation_id)
            
            conversation_thread["total_requests"] += 1
            
            # Check if this new request needs clarification
            if result.get("response_type") == "clarification_request" and result.get("request_id"):
                conversation_thread["pending_request_id"] = result.get("request_id")
                conversation_thread["status"] = "waiting_for_clarification"
            else:
                conversation_thread["completed_requests"] += 1
        
        # Add message to conversation history
        conversation_thread["message_history"].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.utcnow().isoformat(),
            "message_type": "clarification_response" if pending_request_id else "follow_up"
        })
        
        # Update metadata
        conversation_thread["last_interaction"] = datetime.utcnow()
        conversation_thread["status"] = result.get("status", "completed")
        
        # Maintain backward compatibility
        conversations[conversation_id] = conversation_thread.copy()
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error continuing conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to continue conversation: {str(e)}")


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation details and history."""
    if conversation_id not in conversation_threads:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation_thread = conversation_threads[conversation_id]
    result = conversation_thread.copy()
    
    # Add checkpoint information if there's a pending request
    if "pending_request_id" in conversation_thread:
        pending_request_id = conversation_thread["pending_request_id"]
        sql_agent = app_services.get("sql_agent")
        if sql_agent and pending_request_id in sql_agent._checkpoint_storage:
            result["has_pending_clarification"] = True
            result["pending_request_id"] = pending_request_id
        else:
            result["has_pending_clarification"] = False
    else:
        result["has_pending_clarification"] = False
    
    return result


@app.get("/conversations")
async def list_conversations(
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    """List conversations with optional filtering."""
    try:
        filtered_conversations = []
        
        for conv_id, conv_data in conversations.items():
            if status is None or conv_data.get("status") == status:
                filtered_conversations.append({
                    "conversation_id": conv_id,
                    "initial_query": conv_data.get("initial_query", ""),
                    "status": conv_data.get("status", "unknown"),
                    "created_at": conv_data.get("created_at"),
                    "last_interaction": conv_data.get("last_interaction"),
                    "message_count": len(conv_data.get("message_history", []))
                })
        
        # Sort by last interaction, most recent first
        filtered_conversations.sort(
            key=lambda x: x.get("last_interaction", datetime.min),
            reverse=True
        )
        
        # Apply pagination
        paginated = filtered_conversations[offset:offset + limit]
        
        return {
            "conversations": paginated,
            "total": len(filtered_conversations),
            "offset": offset,
            "limit": limit,
            "filtered_by_status": status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and its history."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Remove from both storage locations
    del conversations[conversation_id]
    if conversation_id in conversation_threads:
        del conversation_threads[conversation_id]
    
    return {"message": f"Conversation {conversation_id} deleted successfully"}


# =============================================================================
# ONE-SHOT SQL GENERATION/EDITING ENDPOINTS  
# =============================================================================

@app.post("/sql/generate")
async def generate_sql_oneshot(
    request: OneShotRequest,
    sql_agent: TextToSQLAgent = Depends(get_sql_agent)
):
    """Generate SQL query in one-shot mode without conversation context."""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Generate unique session for this one-shot request
        session_id = str(uuid.uuid4())
        
        if request.mode == "edit" and request.existing_sql:
            # Modify the query to indicate it's an edit request
            edit_query = f"Edit this SQL query: {request.existing_sql}\n\nChange requested: {request.query}"
            result = await sql_agent.generate_sql(edit_query, session_id)
        else:
            # Standard generation
            result = await sql_agent.generate_sql(request.query, session_id)
        
        # Don't store in conversations for one-shot requests
        return {
            "session_id": session_id,
            "mode": request.mode,
            "original_query": request.query,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error in one-shot SQL generation: {e}")
        raise HTTPException(status_code=500, detail=f"One-shot SQL generation failed: {str(e)}")


@app.post("/sql/validate")
async def validate_sql(request: Dict[str, str]):
    """Validate SQL query syntax and suggest improvements."""
    try:
        sql_query = request.get("sql_query", "").strip()
        if not sql_query:
            raise HTTPException(status_code=400, detail="SQL query cannot be empty")
        
        query_service = app_services.get("query_execution_service")
        if not query_service:
            raise HTTPException(status_code=503, detail="Query execution service not available")
        
        result = query_service.validate_query(sql_query)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query validation failed: {str(e)}")


@app.post("/sql/explain")
async def explain_sql(request: Dict[str, str]):
    """Explain what a SQL query does in natural language."""
    try:
        sql_query = request.get("sql_query", "").strip()
        if not sql_query:
            raise HTTPException(status_code=400, detail="SQL query cannot be empty")
        
        # Use vector service to generate explanation with LLM
        vector_service = get_vector_service()
        explanation_query = f"Explain what this SQL query does in natural language: {sql_query}"
        response = vector_service.query_engine.query(explanation_query)
        explanation = str(response)
        
        return {
            "sql_query": sql_query,
            "explanation": explanation,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL explanation failed: {str(e)}")


# =============================================================================
# METADATA MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/metadata/{document_type}")
async def get_metadata_by_type(
    document_type: str,
    limit: int = 50,
    offset: int = 0,
    mongodb_service: MongoDBService = Depends(get_mongodb_service)
):
    """Get all metadata documents of a specific type for UI rendering."""
    try:
        valid_types = ["view", "report", "lookup", "business_domain"]
        if document_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid document type. Must be one of: {valid_types}")
        
        # Adjust document type to match storage format
        storage_type = f"{document_type}_metadata" if document_type != "business_domain" else "business_domain"
        
        documents = await mongodb_service.get_documents_by_type(
            storage_type, limit=limit, offset=offset
        )
        
        return {
            "document_type": document_type,
            "documents": documents,
            "count": len(documents),
            "limit": limit,
            "offset": offset
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")


@app.post("/metadata/search")
async def search_metadata(
    request: MetadataSearchRequest,
    mongodb_service: MongoDBService = Depends(get_mongodb_service)
):
    """Search metadata documents with text query and optional type filter."""
    try:
        if request.query:
            # Use vector search if query provided
            vector_service = get_vector_service()
            results = vector_service.search_similar(
                query=request.query,
                retriever_type="hybrid",
                similarity_top_k=request.limit,
                document_type=request.document_type
            )
            
            return {
                "search_query": request.query,
                "document_type": request.document_type,
                "results": results,
                "total_found": len(results)
            }
        else:
            # Return all documents of specified type
            if not request.document_type:
                raise HTTPException(status_code=400, detail="Either query or document_type must be provided")
            
            storage_type = f"{request.document_type}_metadata" if request.document_type != "business_domain" else "business_domain"
            documents = await mongodb_service.get_documents_by_type(
                storage_type, limit=request.limit, offset=request.offset
            )
            
            return {
                "document_type": request.document_type,
                "results": documents,
                "total_found": len(documents)
            }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metadata search failed: {str(e)}")


@app.post("/metadata/save")
async def save_metadata(
    request: MetadataRequest,
    mongodb_service: MongoDBService = Depends(get_mongodb_service),
    vector_service: LlamaIndexVectorService = Depends(get_vector_service)
):
    """Save or update metadata document and refresh embeddings."""
    try:
        valid_types = ["view", "report", "lookup", "business_domain"]
        if request.document_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid document type. Must be one of: {valid_types}")
        
        # Prepare document data
        document_data = request.data.copy()
        document_data["_document_type"] = f"{request.document_type}_metadata" if request.document_type != "business_domain" else "business_domain"
        document_data["_updated_at"] = datetime.utcnow()
        
        if request.metadata_id:
            # Update existing document
            document_id = request.metadata_id
            
            # Remove old embedding from vector store
            vector_service.delete_document(document_id)
            
            # Update in MongoDB
            updated = await mongodb_service.update_document(document_id, document_data)
            if not updated:
                raise HTTPException(status_code=404, detail="Document not found for update")
            
        else:
            # Create new document
            document_data["_created_at"] = datetime.utcnow()
            document_id = await mongodb_service.insert_document(document_data)
        
        # Generate new embedding and add to vector store
        try:
            # Create full text for embedding based on document type
            if request.document_type == "view":
                full_text = f"View: {document_data.get('view_name', '')}\nDescription: {document_data.get('description', '')}\nColumns: {', '.join([col.get('name', '') for col in document_data.get('columns', [])])}"
            elif request.document_type == "report":
                full_text = f"Report: {document_data.get('report_name', '')}\nDescription: {document_data.get('description', '')}\nUse Cases: {document_data.get('use_cases', '')}"
            elif request.document_type == "lookup":
                full_text = f"Lookup: {document_data.get('lookup_name', '')}\nDescription: {document_data.get('description', '')}\nValues: {len(document_data.get('values', []))} entries"
            else:  # business_domain
                full_text = f"Domain: {document_data.get('domain_name', '')}\nDescription: {document_data.get('description', '')}\nKeywords: {', '.join(document_data.get('keywords', []))}"
            
            # Add to vector store
            success = vector_service.add_document(
                document_id=document_id,
                content=full_text,
                metadata={
                    "document_type": document_data["_document_type"],
                    "updated_at": document_data["_updated_at"].isoformat()
                }
            )
            
            if not success:
                logger.warning(f"Failed to update vector embedding for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error updating vector embedding: {e}")
            # Don't fail the whole operation if embedding update fails
        
        return {
            "document_id": document_id,
            "document_type": request.document_type,
            "action": "updated" if request.metadata_id else "created",
            "message": f"Metadata {'updated' if request.metadata_id else 'saved'} successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save metadata: {str(e)}")


@app.delete("/metadata/{document_type}/{document_id}")
async def delete_metadata(
    document_type: str,
    document_id: str,
    mongodb_service: MongoDBService = Depends(get_mongodb_service),
    vector_service: LlamaIndexVectorService = Depends(get_vector_service)
):
    """Delete metadata document and remove from vector store."""
    try:
        valid_types = ["view", "report", "lookup", "business_domain"]
        if document_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid document type. Must be one of: {valid_types}")
        
        # Delete from vector store
        try:
            vector_service.delete_document(document_id)
        except Exception as e:
            logger.warning(f"Failed to delete vector embedding for document {document_id}: {e}")
        
        # Delete from MongoDB
        deleted = await mongodb_service.delete_document(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "document_id": document_id,
            "document_type": document_type,
            "message": "Metadata deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete metadata: {str(e)}")


# =============================================================================
# DEBUG & UTILITY ENDPOINTS
# =============================================================================

@app.get("/debug/conversations")
async def debug_conversations():
    """Debug endpoint to view all conversation data."""
    return {
        "conversations": conversations,
        "conversation_threads": conversation_threads,
        "total_conversations": len(conversations),
        "total_threads": len(conversation_threads)
    }


@app.post("/admin/reindex")
async def reindex_embeddings():
    """Reindex all metadata embeddings (admin endpoint)."""
    try:
        mongodb_service = get_mongodb_service()
        vector_service = get_vector_service()
        
        # Get all documents from MongoDB
        all_docs = await mongodb_service.get_all_documents()
        if not all_docs:
            return {"message": "No documents found to reindex"}
        
        # Clear and rebuild vector index
        reindexed_count = 0
        for doc in all_docs:
            try:
                doc_id = str(doc["_id"])
                doc_type = doc.get("_document_type", "unknown")
                
                # Generate full text based on type
                if "view" in doc_type:
                    full_text = f"View: {doc.get('view_name', '')}\nDescription: {doc.get('description', '')}"
                elif "report" in doc_type:
                    full_text = f"Report: {doc.get('report_name', '')}\nDescription: {doc.get('description', '')}"
                elif "lookup" in doc_type:
                    full_text = f"Lookup: {doc.get('lookup_name', '')}\nDescription: {doc.get('description', '')}"
                else:
                    full_text = f"Domain: {doc.get('domain_name', '')}\nDescription: {doc.get('description', '')}"
                
                # Remove old and add new
                vector_service.delete_document(doc_id)
                success = vector_service.add_document(
                    document_id=doc_id,
                    content=full_text,
                    metadata={"document_type": doc_type}
                )
                
                if success:
                    reindexed_count += 1
                    
            except Exception as e:
                logger.error(f"Error reindexing document {doc_id}: {e}")
        
        return {
            "message": f"Reindexing completed",
            "total_documents": len(all_docs),
            "reindexed_count": reindexed_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "src.text_to_sql_rag.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app.debug
    )