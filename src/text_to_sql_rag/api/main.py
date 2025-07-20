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
from ..core.rag_pipeline import TextToSQLRAGPipeline
from ..models.simple_models import (
    DocumentType, DocumentUploadRequest, DocumentSearchRequest, DocumentSearchResponse,
    SQLGenerationRequest, SQLGenerationResponse, QueryValidationRequest, QueryValidationResponse,
    QueryExplanationRequest, QueryExplanationResponse, SessionRequest, SessionResponse,
    HealthCheckResponse
)

# In-memory session storage (replace with Redis in production)
sessions = {}

# Initialize FastAPI app
app = FastAPI(
    title=settings.app.title,
    description=settings.app.description,
    version=settings.app.version,
    debug=settings.app.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
vector_service = LlamaIndexVectorService()
query_execution_service = QueryExecutionService()
rag_pipeline = TextToSQLRAGPipeline(vector_service, None)


# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    vector_healthy = vector_service.health_check()
    execution_healthy = query_execution_service.health_check()
    
    status = "healthy"
    if not vector_healthy or not execution_healthy:
        status = "degraded"
    
    return {
        "status": status,
        "vector_store": "connected" if vector_healthy else "disconnected",
        "execution_service": "connected" if execution_healthy else "disconnected",
        "version": settings.app.version
    }


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with service statistics."""
    return {
        "status": "healthy",
        "services": {
            "vector_store": vector_service.get_index_stats(),
            "execution_service": {
                "available": query_execution_service.health_check(),
                "endpoint": settings.app.execution_api_url
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
    """Generate SQL query from natural language."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = rag_pipeline.generate_sql_query(
            natural_language_query=request.query,
            session_id=request.session_id,
            use_hybrid_retrieval=request.use_hybrid_retrieval
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL generation failed: {str(e)}")


@app.post("/query/generate-and-execute")
async def generate_and_execute_sql_query(request: SQLGenerationRequest):
    """Generate SQL query and optionally execute it."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = await rag_pipeline.generate_and_execute_query(
            natural_language_query=request.query,
            session_id=request.session_id,
            use_hybrid_retrieval=request.use_hybrid_retrieval,
            auto_execute=request.auto_execute
        )
        
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
        result = await rag_pipeline.validate_and_suggest_fixes(request.sql_query)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query validation failed: {str(e)}")


@app.post("/query/explain")
async def explain_sql_query(request: QueryExplanationRequest):
    """Explain what a SQL query does."""
    try:
        result = rag_pipeline.explain_query(request.sql_query)
        return result
        
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
        "services": {
            "vector_store_healthy": vector_service.health_check(),
            "execution_service_healthy": query_execution_service.health_check()
        }
    }


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "src.text_to_sql_rag.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app.debug
    )