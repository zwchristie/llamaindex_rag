"""FastAPI application for text-to-SQL RAG system."""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Optional
import tempfile
import os
import uuid
from datetime import datetime
from pathlib import Path

from ..config.settings import settings
from ..services.vector_service import LlamaIndexVectorService
from ..services.query_execution_service import QueryExecutionService
from ..core.langgraph_agent import TextToSQLAgent
from ..core.startup import run_startup_tasks, get_initialized_services
from ..models.simple_models import (
    DocumentType, DocumentUploadRequest, DocumentSearchRequest, DocumentSearchResponse,
    SQLGenerationRequest, SQLGenerationResponse, QueryValidationRequest, QueryValidationResponse,
    QueryExplanationRequest, QueryExplanationResponse, SessionRequest, SessionResponse,
    HealthCheckResponse
)
from ..models.conversation import (
    ConversationState, RequestType, ConversationStatus, AgentResponse
)
from ..services.llm_provider_factory import llm_factory

# In-memory session storage (replace with Redis in production)
sessions = {}
# In-memory conversation storage for HITL workflows
conversations = {}
# Enhanced storage for conversation threads vs individual requests
conversation_threads = {}  # conversation_id -> conversation metadata and message history
active_requests = {}       # request_id -> checkpoint data for HITL workflows

# Initialize FastAPI app
app = FastAPI(
    title=settings.app.title,
    description=settings.app.description,
    version=settings.app.version,
    debug=settings.app.debug
)

# Add CORS middleware
# Configure CORS - update for production
allowed_origins = ["http://localhost:3000", "http://localhost:8080"]  # Add your frontend URLs
if settings.app.debug:
    allowed_origins.append("*")  # Allow all origins in debug mode only

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global variables for services (will be initialized on startup)
vector_service = None
query_execution_service = None
sql_agent = None
mongodb_service = None
sync_service = None


@app.on_event("startup")
async def startup_event():
    """Run startup tasks on application startup."""
    global vector_service, query_execution_service, sql_agent, mongodb_service, sync_service
    
    success = await run_startup_tasks()
    if success:
        # Get initialized services
        mongodb_service, vector_service, sync_service = get_initialized_services()
        query_execution_service = QueryExecutionService()
        sql_agent = TextToSQLAgent(vector_service, query_execution_service)
        print("Application startup completed successfully")
    else:
        print("Application startup failed - some features may not work")


# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not vector_service or not query_execution_service:
        return {
            "status": "starting",
            "message": "Services are still initializing"
        }
    
    vector_healthy = vector_service.health_check()
    execution_healthy = query_execution_service.health_check()
    
    status = "healthy"
    if not vector_healthy or not execution_healthy:
        status = "degraded"
    
    # Add MongoDB health check
    mongodb_status = "disconnected"
    if mongodb_service:
        mongo_health = mongodb_service.health_check()
        mongodb_status = mongo_health.get("status", "disconnected")
    
    # Add document sync status
    sync_status = {}
    if sync_service:
        sync_status = sync_service.get_sync_status()
    
    return {
        "status": status,
        "vector_store": "connected" if vector_healthy else "disconnected",
        "execution_service": "connected" if execution_healthy else "disconnected",
        "mongodb": mongodb_status,
        "document_sync": sync_status,
        "version": settings.app.version,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with service statistics."""
    return {
        "status": "healthy",
        "services": {
            "vector_store": vector_service.get_index_stats() if vector_service else {"status": "not_initialized"},
            "execution_service": {
                "available": query_execution_service.health_check() if query_execution_service else False,
                "endpoint": settings.app.execution_api_url
            },
            "opensearch": {
                "host": settings.opensearch.host,
                "port": settings.opensearch.port,
                "index_name": settings.opensearch.index_name
            },
            "settings": {
                "chunk_size": settings.app.chunk_size,
                "chunk_overlap": settings.app.chunk_overlap,
                "similarity_top_k": settings.app.similarity_top_k
            }
        }
    }


# Document management endpoints
@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    document_type: DocumentType = Form(...),
    description: Optional[str] = Form(None)
):
    """Upload a new document."""
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = Path(file.filename).suffix.lower().lstrip('.')
    if file_extension not in settings.app.allowed_file_types:
        raise HTTPException(
            status_code=400, 
            detail=f"File type .{file_extension} not allowed. Allowed types: {settings.app.allowed_file_types}"
        )
    
    # Check file size
    content = await file.read()
    if len(content) > settings.app.max_upload_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.app.max_upload_size} bytes"
        )
    
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Decode content
        content_text = content.decode('utf-8')
        
        # Generate unique document ID
        document_id = int(str(uuid.uuid4().int)[:10])  # Use first 10 digits of UUID
        
        # Prepare metadata
        metadata = {
            "title": title,
            "description": description,
            "file_name": file.filename,
            "uploaded_at": datetime.utcnow().isoformat(),
            "document_type": document_type.value
        }
        
        # Add to vector store
        success = vector_service.add_document(
            document_id=document_id,
            content=content_text,
            metadata=metadata,
            document_type=document_type.value
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to index document")
        
        return {
            "document_id": document_id,
            "title": title,
            "document_type": document_type.value,
            "file_name": file.filename,
            "status": "indexed",
            "message": "Document uploaded and indexed successfully"
        }
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be text-based and UTF-8 encoded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file_path)
        except:
            pass


@app.get("/documents/{document_id}")
async def get_document_info(document_id: int):
    """Get document information from vector store."""
    try:
        info = vector_service.get_document_info(document_id)
        if info.get("num_chunks", 0) == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document info: {str(e)}")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: int):
    """Delete document from vector store."""
    try:
        success = vector_service.delete_document(document_id)
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


# Search and RAG endpoints
@app.post("/search/documents")
async def search_documents(
    search_request: DocumentSearchRequest
):
    """Search documents using vector similarity."""
    try:
        results = vector_service.search_similar(
            query=search_request.query,
            retriever_type="hybrid",
            similarity_top_k=search_request.limit,
            document_type=search_request.document_types[0].value if search_request.document_types else None
        )
        
        return {
            "query": search_request.query,
            "results": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/query/generate")
async def generate_sql_query(request: SQLGenerationRequest):
    """Generate SQL query from natural language with enhanced conversation support."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Use session_id if provided, otherwise generate new conversation
        conversation_id = request.session_id or str(uuid.uuid4())
        
        result = await sql_agent.generate_sql(request.query, conversation_id)
        
        # Store conversation state if it requires human input
        if result.get("response_type") == "clarification_request":
            # This is a placeholder - in production, store in Redis or database
            conversations[conversation_id] = {
                "conversation_id": conversation_id,
                "status": "waiting_for_clarification",
                "created_at": datetime.utcnow(),
                "last_interaction": datetime.utcnow()
            }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL generation failed: {str(e)}")


@app.post("/query/generate-and-execute")
async def generate_and_execute_sql_query(request: SQLGenerationRequest):
    """Generate SQL query and automatically execute it."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Use session_id if provided, otherwise generate new conversation
        conversation_id = request.session_id or str(uuid.uuid4())
        
        # The LangGraph agent automatically handles execution if query_execution_service is provided
        result = await sql_agent.generate_sql(request.query, conversation_id)
        
        # Enable auto-execution for this workflow
        if result.get("response_type") == "sql_result" and result.get("sql"):
            # Mark that execution should happen
            # This would be handled in the agent workflow
            pass
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL generation and execution failed: {str(e)}")


@app.post("/query/execute")
async def execute_sql_query(request: dict):
    """Execute SQL query via external service."""
    sql_query = request.get("sql_query", "").strip()
    if not sql_query:
        raise HTTPException(status_code=400, detail="SQL query cannot be empty")
    
    try:
        result = await query_execution_service.execute_query(
            sql_query=sql_query,
            session_id=request.get("session_id"),
            metadata=request.get("metadata", {})
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


@app.post("/query/validate")
async def validate_sql_query(request: QueryValidationRequest):
    """Validate SQL query and suggest fixes if invalid."""
    try:
        result = query_execution_service.validate_query(request.sql_query)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query validation failed: {str(e)}")


@app.post("/query/explain")
async def explain_sql_query(request: QueryExplanationRequest):
    """Explain what a SQL query does."""
    try:
        # Create a simple explanation using the vector service
        explanation_query = f"Explain what this SQL query does: {request.sql_query}"
        response = sql_agent.vector_service.query_engine.query(explanation_query)
        
        return {
            "sql_query": request.sql_query,
            "explanation": str(response),
            "confidence": 0.8
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query explanation failed: {str(e)}")


# Session management endpoints (simplified in-memory implementation)
@app.post("/sessions")
async def create_session(session_data: SessionRequest):
    """Create a new user session."""
    session_id = str(uuid.uuid4())
    
    session = {
        "session_id": session_id,
        "initial_query": session_data.initial_query,
        "user_id": session_data.user_id,
        "context": session_data.context,
        "created_at": datetime.utcnow(),
        "status": "active",
        "interactions": []
    }
    
    sessions[session_id] = session
    
    return {
        "session_id": session_id,
        "initial_query": session_data.initial_query,
        "user_id": session_data.user_id,
        "context": session_data.context,
        "created_at": session["created_at"],
        "status": "active"
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session by ID."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "initial_query": session["initial_query"],
        "user_id": session.get("user_id"),
        "context": session["context"],
        "created_at": session["created_at"],
        "status": session["status"],
        "interactions": session.get("interactions", [])
    }


@app.get("/stats")
async def get_stats():
    """Get application statistics."""
    return {
        "vector_store": vector_service.get_index_stats(),
        "active_sessions": len(sessions),
        "active_conversations": len(conversations),
        "pending_clarifications": len([c for c in conversations.values() if c.get("status") == "waiting_for_clarification"]),
        "llm_provider": llm_factory.get_provider_info(),
        "services": {
            "vector_store_healthy": vector_service.health_check(),
            "execution_service_healthy": query_execution_service.health_check(),
            "llm_provider_healthy": llm_factory.health_check()
        }
    }


# Enhanced conversation management endpoints for HITL workflows
@app.post("/conversations/start")
async def start_conversation(request: dict):
    """Start a new conversation with enhanced HITL support."""
    query = request.get("query", "").strip()
    context = request.get("context", {})
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        conversation_id = str(uuid.uuid4())
        result = await sql_agent.generate_sql(query, conversation_id)
        
        # Store conversation thread metadata
        conversation_threads[conversation_id] = {
            "conversation_id": conversation_id,
            "initial_query": query,
            "status": result.get("status", "completed"),
            "created_at": datetime.utcnow(),
            "last_interaction": datetime.utcnow(),
            "context": context,
            "message_history": [
                {
                    "role": "user", 
                    "content": query, 
                    "timestamp": datetime.utcnow().isoformat(),
                    "message_type": "initial_request"
                }
            ],
            "total_requests": 1,
            "completed_requests": 1 if result.get("response_type") != "clarification_request" else 0
        }
        
        # If this is a clarification request, store the checkpoint
        if result.get("response_type") == "clarification_request" and result.get("request_id"):
            # The checkpoint should already be stored by the agent
            conversation_threads[conversation_id]["status"] = "waiting_for_clarification"
            conversation_threads[conversation_id]["pending_request_id"] = result.get("request_id")
        
        # Maintain backward compatibility with old conversations storage
        conversations[conversation_id] = conversation_threads[conversation_id].copy()
        
        return {
            "conversation_id": conversation_id,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start conversation: {str(e)}")


@app.post("/conversations/{conversation_id}/continue")
async def continue_conversation(conversation_id: str, request: dict):
    """Continue an existing conversation with clarification or follow-up."""
    message = request.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if conversation_id not in conversation_threads:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    try:
        conversation_thread = conversation_threads[conversation_id]
        
        # Check if there's a pending clarification request
        pending_request_id = conversation_thread.get("pending_request_id")
        
        if pending_request_id and pending_request_id in sql_agent._checkpoint_storage:
            # This is a clarification response - resume from checkpoint
            logger.info(f"Resuming from checkpoint for request {pending_request_id}")
            result = await sql_agent.continue_from_checkpoint(pending_request_id, message)
            
            # Clear the pending request
            if "pending_request_id" in conversation_thread:
                del conversation_thread["pending_request_id"]
            
            conversation_thread["completed_requests"] += 1
            
        else:
            # This is a new request in the conversation thread
            logger.info(f"Starting new request in conversation {conversation_id}")
            result = await sql_agent.generate_sql(message, conversation_id)
            
            conversation_thread["total_requests"] += 1
            
            # Check if this new request also needs clarification
            if result.get("response_type") == "clarification_request" and result.get("request_id"):
                conversation_thread["pending_request_id"] = result.get("request_id")
                conversation_thread["status"] = "waiting_for_clarification"
            else:
                conversation_thread["completed_requests"] += 1
        
        # Add message to conversation history
        conversation_thread["message_history"].append({
            "role": "user", 
            "content": message, 
            "timestamp": datetime.utcnow().isoformat(),
            "message_type": "clarification_response" if pending_request_id else "follow_up"
        })
        
        # Update conversation metadata
        conversation_thread["last_interaction"] = datetime.utcnow()
        conversation_thread["status"] = result.get("status", "completed")
        
        # Maintain backward compatibility
        conversations[conversation_id] = conversation_thread.copy()
        
        return result
        
    except Exception as e:
        logger.error(f"Error continuing conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to continue conversation: {str(e)}")


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation details and history."""
    if conversation_id not in conversation_threads:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation_thread = conversation_threads[conversation_id]
    
    # Add checkpoint information if there's a pending request
    result = conversation_thread.copy()
    
    if "pending_request_id" in conversation_thread:
        pending_request_id = conversation_thread["pending_request_id"]
        result["has_pending_clarification"] = pending_request_id in sql_agent._checkpoint_storage
        result["pending_request_id"] = pending_request_id
    else:
        result["has_pending_clarification"] = False
    
    return result


@app.get("/conversations/{conversation_id}/status")
async def get_conversation_status(conversation_id: str):
    """Get conversation status and pending clarifications."""
    if conversation_id not in conversation_threads:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation_thread = conversation_threads[conversation_id]
    
    status = {
        "conversation_id": conversation_id,
        "status": conversation_thread.get("status", "unknown"),
        "total_requests": conversation_thread.get("total_requests", 0),
        "completed_requests": conversation_thread.get("completed_requests", 0),
        "has_pending_clarification": False,
        "pending_request_id": None,
        "last_interaction": conversation_thread.get("last_interaction")
    }
    
    # Check for pending clarifications
    if "pending_request_id" in conversation_thread:
        pending_request_id = conversation_thread["pending_request_id"]
        if pending_request_id in sql_agent._checkpoint_storage:
            status["has_pending_clarification"] = True
            status["pending_request_id"] = pending_request_id
    
    return status


@app.get("/conversations")
async def list_conversations(status: Optional[str] = None, limit: int = 20):
    """List conversations with optional status filter."""
    filtered_conversations = []
    
    for conv_id, conv_data in conversations.items():
        if status is None or conv_data.get("status") == status:
            filtered_conversations.append({
                "conversation_id": conv_id,
                "initial_query": conv_data.get("initial_query", ""),
                "status": conv_data.get("status", "unknown"),
                "created_at": conv_data.get("created_at"),
                "last_interaction": conv_data.get("last_interaction"),
                "message_count": len(conv_data.get("messages", []))
            })
    
    # Sort by last interaction, most recent first
    filtered_conversations.sort(
        key=lambda x: x.get("last_interaction", datetime.min), 
        reverse=True
    )
    
    return {
        "conversations": filtered_conversations[:limit],
        "total": len(filtered_conversations),
        "filtered_by_status": status
    }


@app.post("/conversations/{conversation_id}/describe-sql")
async def describe_sql_in_conversation(conversation_id: str, request: dict):
    """Describe an SQL query within a conversation context."""
    sql_query = request.get("sql_query", "").strip()
    if not sql_query:
        raise HTTPException(status_code=400, detail="SQL query cannot be empty")
    
    try:
        # Create a conversation state for SQL description
        from ..models.conversation import ConversationState, RequestType
        conversation_state = ConversationState(
            conversation_id=conversation_id,
            current_request=f"Describe this SQL query: {sql_query}",
            request_type=RequestType.DESCRIBE_SQL
        )
        
        # Use the agent's describe_sql workflow
        final_state = sql_agent.graph.invoke(conversation_state)
        
        return final_state.final_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to describe SQL: {str(e)}")


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and its history."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    del conversations[conversation_id]
    
    return {"message": f"Conversation {conversation_id} deleted successfully"}


# LLM Provider management endpoints
@app.get("/llm-provider/info")
async def get_llm_provider_info():
    """Get information about the current LLM provider."""
    try:
        provider_info = llm_factory.get_provider_info()
        health_status = llm_factory.health_check()
        
        return {
            "provider_info": provider_info,
            "health_status": health_status,
            "available_providers": ["bedrock", "custom"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get provider info: {str(e)}")


@app.post("/llm-provider/switch")
async def switch_llm_provider(request: dict):
    """Switch to a different LLM provider."""
    provider_name = request.get("provider", "").lower()
    
    if provider_name not in ["bedrock", "custom"]:
        raise HTTPException(status_code=400, detail="Provider must be either 'bedrock' or 'custom'")
    
    try:
        success = llm_factory.switch_provider(provider_name)
        if success:
            new_info = llm_factory.get_provider_info()
            return {
                "success": True,
                "message": f"Successfully switched to {provider_name} provider",
                "provider_info": new_info
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to switch to {provider_name} provider")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error switching provider: {str(e)}")


@app.get("/llm-provider/test")
async def test_llm_provider():
    """Test the current LLM provider with a simple query."""
    try:
        test_prompt = "Generate a simple SQL query to select all columns from a table named 'users'. Respond with just the SQL query."
        
        response = llm_factory.generate_text(test_prompt)
        
        return {
            "success": True,
            "provider": llm_factory.get_provider_info()["provider"],
            "test_prompt": test_prompt,
            "response": response,
            "response_length": len(response)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "provider": llm_factory.get_provider_info()["provider"]
        }


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "src.text_to_sql_rag.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app.debug
    )