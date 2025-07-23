# Functional Taxonomy and Architecture Documentation

## Purpose
This document provides a comprehensive functional taxonomy of the LlamaIndex RAG Text-to-SQL system, designed to help coding assistants understand the codebase structure, responsibilities, and relationships between components.

## 🗂️ Directory Structure and Responsibilities

```
src/text_to_sql_rag/
├── api/                          # Web API Layer
│   └── main.py                   # FastAPI application with all endpoints
├── core/                         # Core Business Logic
│   ├── langgraph_agent.py        # LangGraph-based SQL generation agent
│   └── startup.py                # Application initialization and service setup
├── services/                     # External Service Integrations
│   ├── vector_service.py         # LlamaIndex + Qdrant vector operations
│   ├── mongodb_service.py        # MongoDB document storage operations
│   ├── document_sync_service.py  # Document synchronization orchestration
│   ├── bedrock_service.py        # AWS Bedrock LLM/embedding integration
│   ├── custom_llm_service.py     # Custom internal LLM API integration
│   ├── llm_provider_factory.py   # LLM provider switching and management
│   └── query_execution_service.py # External database query execution
├── models/                       # Data Models and Schemas
│   ├── database.py               # Shared SQLAlchemy base
│   ├── simple_models.py          # API request/response models
│   ├── conversation.py           # Conversation and HITL models
│   ├── document.py               # Document database models
│   ├── session.py                # Session management models
│   └── meta_document.py          # Metadata document models
├── config/                       # Configuration Management
│   └── settings.py               # Pydantic settings with environment variables
└── utils/                        # Utility Functions
    └── content_processor.py      # Document content processing utilities
```

## 🧩 Component Functional Taxonomy

### 1. API Layer (`api/`)

#### `main.py` - FastAPI Application
**Purpose**: Primary web interface providing RESTful API endpoints
**Responsibilities**:
- HTTP request/response handling
- Input validation and sanitization
- Authentication and authorization
- Session and conversation management
- Error handling and response formatting

**Key Endpoints**:
- `/query/generate` - Basic SQL generation
- `/conversations/start` - Start enhanced HITL conversation threads
- `/conversations/{id}/continue` - Continue conversations with automatic checkpoint detection
- `/conversations/{id}/status` - Monitor conversation status and pending clarifications
- `/conversations/*` - Advanced HITL conversation management
- `/documents/*` - Document upload and management
- `/health` - Service health monitoring

**Dependencies**: All services, models, and configuration

---

### 2. Core Business Logic (`core/`)

#### `langgraph_agent.py` - LangGraph Agent
**Purpose**: Orchestrates the intelligent text-to-SQL workflow using LangGraph with enhanced HITL state management
**Responsibilities**:
- LLM-powered request classification (generate, execute, describe, edit)
- Confidence assessment for metadata completeness using LLM analysis
- Advanced human-in-the-loop clarification workflows with state persistence
- SQL generation with context and retry logic
- Checkpoint-based conversation state management
- Adaptive workflow routing with resumption capabilities

**Key Components**:
- `TextToSQLAgent` - Main agent class with checkpoint storage
- `ConversationState` - Enhanced workflow state with dual ID system
- `WorkflowState` - Serializable workflow execution state for HITL persistence
- Workflow nodes: classify, assess_confidence, generate_sql, request_clarification, etc.
- Checkpoint management: save/restore state for HITL interruptions
- Helper methods for parsing, context formatting, and state serialization

**Enhanced HITL Features**:
- **Dual ID System**: Separate conversation_id (thread) and request_id (fulfillment)
- **State Checkpointing**: Complete workflow state preservation before clarification
- **Intelligent Resumption**: Exact context restoration after human input
- **LLM Classification**: Advanced request type detection with reasoning

**Input**: Natural language queries, conversation context, clarification responses
**Output**: SQL queries, clarification requests, execution results, or checkpoint data

#### `startup.py` - Application Initialization
**Purpose**: Orchestrates application startup and service initialization
**Responsibilities**:
- Service health checks and initialization
- Document synchronization on startup
- Error handling for service failures
- Graceful degradation configuration

**Dependencies**: All services and configuration

---

### 3. Service Integrations (`services/`)

#### `vector_service.py` - Vector Operations
**Purpose**: Manages document indexing and semantic search using LlamaIndex + Qdrant
**Responsibilities**:
- Document ingestion and chunking
- Vector embedding generation
- Hybrid semantic search (vector + keyword)
- Document preprocessing (JSON to Dolphin format)
- Query engine integration for LLM responses

**Key Features**:
- Multi-model embedding support (Titan, Cohere)
- Configurable chunking strategies
- Metadata filtering and search
- Health monitoring and statistics

#### `mongodb_service.py` - Document Storage
**Purpose**: Manages document metadata storage and persistence
**Responsibilities**:
- Document CRUD operations
- Content hash-based change detection
- Connection management and health monitoring
- Query interfaces for document retrieval

**Data Stored**:
- Document metadata and content
- File hashes for change detection
- Timestamps and version tracking

#### `document_sync_service.py` - Synchronization Orchestration
**Purpose**: Coordinates document synchronization between filesystem, MongoDB, and vector store
**Responsibilities**:
- File system monitoring and parsing
- Multi-destination synchronization
- Error handling and retry logic
- Sync reporting and statistics

**Workflow**:
1. Scan `meta_documents/` directory
2. Parse and validate document content
3. Update MongoDB with metadata
4. Sync to vector store for search

#### `bedrock_service.py` - AWS Bedrock Integration
**Purpose**: Provides LLM and embedding services via AWS Bedrock
**Responsibilities**:
- Multi-model support (Claude, Titan, Llama)
- Profile-based authentication for local development
- Credential-based authentication for production
- Embedding generation for multiple text inputs
- Text generation with configurable parameters
- Error handling and fallback mechanisms

**Supported Models**:
- Embeddings: Amazon Titan, Cohere Embed
- LLMs: Anthropic Claude, Amazon Titan, Meta Llama

#### `custom_llm_service.py` - Custom LLM Integration
**Purpose**: Integrates with internal LLM API services
**Responsibilities**:
- HTTP client for custom LLM endpoints (invoke, invoke/followup, conversation)
- Conversation ID management for multi-turn interactions
- Deployment ID and model name configuration
- Error handling with retry logic
- Response parsing and formatting

**Key Features**:
- Support for invoke, invoke/followup, and conversation endpoints
- Automatic conversation ID generation
- Configurable timeout and retry settings
- JSON response parsing with "message" key extraction

#### `llm_provider_factory.py` - LLM Provider Management
**Purpose**: Manages switching between different LLM providers
**Responsibilities**:
- Provider initialization and switching (Bedrock ↔ Custom)
- Unified interface for all LLM provider types
- Health monitoring across providers
- Provider information and status reporting
- Runtime provider switching without restart

**Key Methods**:
- `switch_provider()` - Runtime provider switching
- `get_provider_info()` - Current provider details
- `health_check()` - Provider availability status
- `generate_text()` - Unified text generation interface

#### `query_execution_service.py` - Database Execution
**Purpose**: Interfaces with external database execution services
**Responsibilities**:
- SQL query execution via HTTP API
- Query validation and syntax checking
- Result formatting and error handling
- Health monitoring of execution service

---

### 4. Data Models (`models/`)

#### `database.py` - SQLAlchemy Base
**Purpose**: Provides shared database configuration
**Content**: Single `declarative_base()` for all database models

#### `simple_models.py` - API Models
**Purpose**: Pydantic models for API request/response validation
**Key Models**:
- `SQLGenerationRequest/Response` - SQL generation endpoints
- `DocumentUploadRequest` - Document upload validation
- `DocumentSearchRequest/Response` - Search interfaces
- `HealthCheckResponse` - Service monitoring
- `HumanInterventionRequest` - HITL workflows

#### `conversation.py` - Enhanced Conversation Models
**Purpose**: Models for advanced conversation and enhanced HITL workflows with state persistence
**Key Models**:
- `ConversationState` - Enhanced conversation state with dual ID system (conversation_id + request_id)
- `WorkflowState` - Serializable workflow execution state for checkpoint persistence
- `RequestType` - Enum for request classification (GENERATE_NEW, EDIT_PREVIOUS, FOLLOW_UP, etc.)
- `ClarificationRequest` - Human clarification requests with structured context
- `SQLArtifact` - SQL generation artifacts with metadata and execution results
- `AgentResponse` - Structured agent responses
- `ConversationMessage` - Individual messages in conversation history

**Enhanced HITL Features**:
- **State Serialization**: `save_workflow_checkpoint()` and `restore_from_checkpoint()` methods
- **Dual ID System**: Separation of conversation threads and individual request fulfillment
- **Property Delegation**: Backward compatibility while supporting new state structure
- **Context Preservation**: Complete intermediate workflow data maintenance

#### `document.py` - Document Database Models
**Purpose**: SQLAlchemy models for document persistence
**Key Models**:
- `Document` - Database table for document storage
- `DocumentStatus` - Enum for processing states
- Relationship models for document associations

#### `session.py` - Session Management
**Purpose**: Models for user session tracking and management
**Key Models**:
- `Session` - User session database model
- `Interaction` - Individual user interactions
- `SessionStatus` - Session state enumeration
- `HumanInterventionResponse` - Human feedback models

#### `meta_document.py` - Metadata Models
**Purpose**: Models for parsing and validating metadata documents
**Key Models**:
- `SchemaMetadata` - Database schema information
- `ReportMetadata` - SQL report information
- `ColumnInfo`, `TableModel` - Schema components
- `MetaDocument` - Base metadata document

---

### 5. Configuration (`config/`)

#### `settings.py` - Configuration Management
**Purpose**: Centralized configuration using Pydantic Settings
**Configuration Groups**:
- `AppSettings` - General application config
- `AWSSettings` - Bedrock service configuration (with profile support)
- `QdrantSettings` - Vector database config
- `MongoDBSettings` - Document storage config
- `SecuritySettings` - Security and authentication
- `LLMProviderSettings` - LLM provider selection (bedrock/custom)
- `CustomLLMSettings` - Custom LLM API configuration

**Features**:
- Environment variable integration
- Type validation and defaults
- Nested configuration objects
- Dynamic custom LLM loading
- AWS profile vs credential configuration
- Provider-specific settings isolation

---

### 6. Utilities (`utils/`)

#### `content_processor.py` - Content Processing
**Purpose**: Document content parsing and transformation utilities
**Responsibilities**:
- JSON to Dolphin format conversion
- SQL query extraction from text
- Document validation and parsing
- Metadata extraction and enrichment

**Key Methods**:
- `convert_json_to_dolphin_format()` - Enhanced text format conversion
- `extract_sql_queries()` - SQL parsing from documents
- `validate_document_content()` - Content validation
- `parse_schema_document()` - Schema parsing

---

## 🔄 Data Flow Patterns

### 1. Document Ingestion Flow
```
Filesystem (meta_documents/) 
→ DocumentSyncService.sync_all_documents()
→ MongoDBService.upsert_document()
→ VectorService.add_document()
→ ContentProcessor.convert_json_to_dolphin_format()
```

### 2. SQL Generation Flow
```
User Query 
→ LangGraphAgent.generate_sql()
→ ConversationState management
→ VectorService.search_similar()
→ BedrockService.generate_text()
→ SQL parsing and validation
```

### 3. Enhanced HITL Clarification Flow with State Persistence
```
Low Confidence Assessment
→ LangGraphAgent._assess_confidence_node() (LLM-powered analysis)
→ LangGraphAgent._request_clarification_node()
→ ConversationState.save_workflow_checkpoint() (Complete state serialization)
→ Checkpoint stored in agent._checkpoint_storage[request_id]
→ API response with clarification + request_id
→ User provides clarification via /conversations/{id}/continue
→ API detects pending_request_id and calls continue_from_checkpoint()
→ ConversationState.restore_from_checkpoint() (Full context restoration)
→ LangGraph workflow resumes with preserved schema context, confidence, metadata
```

### 4. Conversation vs Request ID Management Flow
```
User starts conversation
→ Generate conversation_id (thread tracking)
→ Generate request_id (individual fulfillment)
→ Store in conversation_threads[conversation_id]
→ Execute workflow for request_id
→ If clarification needed: save checkpoint with request_id
→ User continues: detect pending_request_id vs new request
→ Resume checkpoint OR start new request_id in same conversation_id
```

### 5. LLM Provider Switching Flow
```
Provider Switch Request
→ LLMProviderFactory.switch_provider()
→ Current provider health check
→ New provider initialization
→ Provider configuration validation
→ Runtime provider swap
→ Health status update
```

## 🔗 Key Relationships and Dependencies

### Service Dependencies
- **LangGraphAgent** depends on LLMProviderFactory, VectorService, QueryExecutionService
- **VectorService** depends on BedrockService for embeddings (always AWS for embeddings)
- **LLMProviderFactory** manages BedrockService and CustomLLMService
- **DocumentSyncService** orchestrates MongoDBService and VectorService
- **API endpoints** depend on all services for different operations
- **Provider switching** affects LangGraphAgent workflow routing

### Model Relationships
- **ConversationState** contains SQLArtifact history and ClarificationRequest
- **Document models** share common DocumentType enum
- **Session models** reference Document and User entities
- **MetaDocument models** provide structured parsing for raw documents

### Configuration Cascade
- **Settings** propagate to all services via dependency injection
- **Environment variables** override default configuration values
- **Debug mode** affects logging levels and CORS policies

## 🎯 Design Patterns and Principles

### 1. Service Layer Pattern
Each service has a single responsibility and clear interface:
- VectorService: Document indexing and search
- MongoDBService: Persistent storage
- BedrockService: AI/ML operations

### 2. Repository Pattern
Services act as repositories for their respective data sources:
- Abstract database operations behind service interfaces
- Consistent error handling and logging
- Health monitoring and connection management

### 3. State Machine Pattern (LangGraph)
The agent workflow implements a state machine:
- Nodes represent processing steps
- Edges define transition conditions
- State carries context between nodes

### 4. Factory Pattern
Service creation and configuration:
- Settings classes create configured service instances
- Environment-based configuration selection
- Graceful degradation for missing services

### 5. Observer Pattern
Health monitoring and status reporting:
- Services report health status
- Centralized health aggregation
- Startup synchronization coordination

## 🔍 Key Algorithms and Logic

### 1. Confidence Assessment Algorithm
```python
def assess_confidence(query, schema_context, example_context):
    # Analyze metadata completeness
    # Check for ambiguous terms
    # Evaluate context sufficiency
    # Return confidence score [0.0-1.0]
    return confidence_score, clarification_needed
```

### 2. Hybrid Search Strategy
```python
def hybrid_search(query):
    # Vector similarity search
    # Keyword-based retrieval
    # Query fusion (multiple variants)
    # Re-ranking and filtering
    return ranked_results
```

### 3. Document Synchronization Logic
```python
def sync_document(file_path):
    # Calculate content hash
    # Check if update needed
    # Parse and validate content
    # Update MongoDB and vector store
    return sync_result
```

### 4. Enhanced LLM-Powered Request Classification
```python
def _classify_request_node(state: ConversationState):
    # Build context for LLM classification
    context = build_classification_context(state)
    
    # LLM-powered classification with reasoning
    classification_prompt = create_classification_prompt(context)
    response = llm_factory.generate_text(classification_prompt)
    
    # Parse classification and reasoning
    result = parse_classification_response(response)
    
    # Map to RequestType with fallback
    state.request_type = map_classification(result)
    
    return state
```

### 5. Workflow State Checkpoint Algorithm
```python
def save_workflow_checkpoint(state: ConversationState):
    # Serialize complete workflow state
    checkpoint_data = {
        "conversation_id": state.conversation_id,
        "request_id": state.request_id,
        "workflow_state": state.workflow_state.serialize(),
        "current_request": state.current_request,
        "request_type": state.request_type,
        "message_history": state.message_history,
        "timestamps": {...}
    }
    
    # Store checkpoint for resumption
    agent._checkpoint_storage[state.request_id] = checkpoint_data
    
    return checkpoint_data

def restore_from_checkpoint(checkpoint_data, new_request):
    # Deserialize workflow state
    workflow_state = WorkflowState.deserialize(checkpoint_data["workflow_state"])
    
    # Restore conversation state with preserved context
    restored_state = ConversationState(
        conversation_id=checkpoint_data["conversation_id"],
        request_id=checkpoint_data["request_id"],
        current_request=new_request,  # New user input
        workflow_state=workflow_state  # Preserved intermediate state
    )
    
    # Reset HITL flags for continuation
    restored_state.requires_human_input = False
    restored_state.needs_clarification = False
    
    return restored_state
```

## 🚀 Extensibility Points

### 1. Adding New LLM Providers
- Create new service class implementing LLM interface (e.g., `OpenAIService`)
- Add provider configuration class to `settings.py`
- Update `LLMProviderFactory.switch_provider()` to handle new provider
- Add provider health check and information methods
- Update environment configuration examples
- Add provider-specific endpoints if needed

### 2. Supporting New Document Types
- Add to `DocumentType` enum
- Extend `ContentProcessor` parsing methods
- Update `DocumentSyncService` routing

### 3. Custom Workflow Nodes
- Add new nodes to `LangGraphAgent`
- Implement node method following pattern
- Update workflow graph construction

### 4. Additional Vector Stores
- Create new service implementing `VectorService` interface
- Add configuration settings
- Update service factory in startup

## 📊 Performance Considerations

### 1. Caching Strategies
- Vector search results caching
- Document content caching
- Configuration value caching

### 2. Connection Pooling
- MongoDB connection pools
- HTTP client session reuse
- Vector store connection management

### 3. Batch Operations
- Document synchronization batching
- Vector embedding batch generation
- Query result aggregation

### 4. Async Operations
- Non-blocking I/O for external APIs
- Concurrent document processing
- Parallel service health checks

---

This taxonomy provides a comprehensive understanding of the system architecture, enabling coding assistants to quickly understand component relationships, data flows, and extension points for effective code assistance and development.