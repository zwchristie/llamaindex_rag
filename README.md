# Text-to-SQL RAG System 2.0

A simplified, demo-ready text-to-SQL prototype with Human-in-the-Loop (HITL) approval, featuring a clean architecture without domain concepts.

## 🎯 Overview

This system transforms natural language questions into SQL queries using:
- **One-document-per-view** metadata model (no domains)
- **Vector similarity search** for view retrieval
- **Human-in-the-Loop approval** for generated SQL
- **State persistence** for resumable workflows
- **WrenAI-inspired** large chunk approach for metadata

## 🏗️ Architecture

```mermaid
graph TD
    A[User Query] --> B[Text-to-SQL Agent]
    B --> C[Embedding Service]
    C --> D[Vector Search]
    D --> E[View Retrieval]
    E --> F[SQL Generation]
    F --> G[HITL Approval]
    G --> H[SQL Execution]
    H --> I[Formatted Response]
    
    J[MongoDB] --> K[View Metadata]
    J --> L[Session States]
    J --> M[HITL Requests]
    
    N[OpenSearch] --> O[View Embeddings]
    
    P[Bedrock API Gateway] --> Q[LLM Service]
    P --> R[Embedding Service]
```

## 📊 Key Features

### 🔍 Simplified Metadata Model
- **One document per view** in MongoDB
- **No domain concepts** - views are self-contained
- **Rich metadata**: columns, types, joins, sample SQL
- **Full-text concatenation** for embedding

### 🧠 Smart Retrieval
- **Vector similarity search** using Bedrock embeddings
- **Hybrid search** combining text and vector matching  
- **Large chunk approach** - views don't get split up
- **Dynamic dimension detection** for embeddings

### 👥 Human-in-the-Loop
- **Blocking approval checkpoint** before SQL execution
- **Persistent request state** in MongoDB
- **Resumable workflows** after restarts
- **Approval/rejection with notes**

### 🔄 State Management
- **Session persistence** for resumable flows
- **LangGraph agent** with proper state transitions
- **Error handling** and recovery
- **Audit trail** of all interactions

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker and Docker Compose
- Poetry (for dependency management)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd llamaindex_proj
cp .env.example .env
# Edit .env with your settings
```

### 2. Start Services
```bash
make up          # Start MongoDB, OpenSearch, Redis
make seed        # Seed with mock data  
make reindex     # Build embeddings index

# OR use the quick start script
python quick_start.py
```

### 3. Run the Application
```bash
poetry install
poetry run python src/text_to_sql_rag/api/new_main.py
```

### 4. Test the System
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- OpenSearch Dashboards: http://localhost:5601

## 🛠️ Development Commands

```bash
# Quick setup
make dev-setup   # Complete setup: up + seed + reindex

# Development
make up          # Start all services
make down        # Stop all services  
make clean       # Stop and remove volumes

# Data management
make seed        # Seed mock data
make reindex     # Rebuild OpenSearch index

# Testing
make test        # Run all tests
python tests/run_tests.py  # Detailed test runner

# Code quality
make lint        # Run linting
make format      # Format code
```

## 📡 API Endpoints

### Core Query Processing
- `POST /query` - Process text-to-SQL query with HITL
- `GET /sessions/{session_id}` - Get session state

### HITL Management
- `GET /hitl/requests` - Get pending approval requests
- `POST /hitl/resolve` - Approve/reject requests
- `GET /hitl/requests/{request_id}` - Get specific request

### View Management
- `GET /views` - List all views
- `GET /views/{view_name}` - Get specific view
- `POST /views/search` - Search views by text

### System
- `GET /health` - Health check
- `GET /stats` - System statistics
- `POST /admin/reindex` - Rebuild index
- `POST /admin/cleanup` - Clean old data

## 🔧 Configuration

Key environment variables:

```env
# Bedrock API Gateway
BEDROCK_ENDPOINT_URL=https://8v1n9dbomk.execute-api.us-east-1.amazonaws.com/testaccess
BEDROCK_LLM_MODEL=anthropic.claude-3-haiku-20240307-v1:0
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0

# MongoDB
MONGODB_URL=mongodb://admin:password@localhost:27017
MONGODB_DATABASE=text_to_sql_rag

# OpenSearch
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_INDEX_NAME=view_metadata
OPENSEARCH_VECTOR_FIELD=embedding

# HITL
HITL_TIMEOUT_MINUTES=30
MAX_PENDING_REQUESTS=100
```

## 📋 Example Usage

### 1. Process a Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me all users created in the last 30 days"
  }'
```

### 2. Approve SQL
```bash
curl -X POST http://localhost:8000/hitl/resolve \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "uuid-here",
    "action": "approve",
    "notes": "SQL looks correct"
  }'
```

### 3. Search Views
```bash
curl -X POST http://localhost:8000/views/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "user metrics",
    "limit": 5
  }'
```

## 🧪 Testing

### Automated System Tests
```bash
# Quick validation of core components
python validate_system.py

# Complete automated test suite
python run_comprehensive_tests.py

# Individual test suites
make test-unit          # Unit tests
make test-integration   # Integration tests  
make test-system        # System tests
make test-all          # All test suites
```

### What Gets Tested
- ✅ **MongoDB Operations**: Document upload, retrieval, indexing
- ✅ **OpenSearch Integration**: Embedding storage, vector search
- ✅ **Complete Text-to-SQL Flow**: End-to-end query processing
- ✅ **HITL Workflow**: Approval system with state persistence
- ✅ **User Clarification**: Mid-flow clarification handling
- ✅ **Follow-up Questions**: SQL modification and conversation continuity
- ✅ **Error Handling**: System resilience and recovery

## 📁 Project Structure

```
├── src/text_to_sql_rag/
│   ├── api/
│   │   ├── main.py           # Legacy API
│   │   └── new_main.py       # New simplified API ⭐
│   ├── core/
│   │   └── text_to_sql_agent.py  # Main LangGraph agent ⭐
│   ├── models/
│   │   └── view_models.py    # Simplified models ⭐
│   ├── services/
│   │   ├── view_service.py   # View CRUD operations ⭐
│   │   ├── embedding_service.py  # Embedding & vector ops ⭐
│   │   ├── hitl_service.py   # HITL workflow ⭐
│   │   └── session_service.py    # State persistence ⭐
│   └── config/
│       └── new_settings.py   # Configuration ⭐
├── scripts/
│   ├── seed_mock_data.py     # Database seeding ⭐
│   └── reindex_metadata.py   # Index rebuilding ⭐
├── tests/
│   ├── unit/                 # Unit tests ⭐
│   └── integration/          # Integration tests ⭐
├── meta_documents/
│   ├── views/               # Sample view metadata
│   ├── reports/             # Sample reports  
│   └── lookups/             # Sample lookups
├── docker-compose.yml        # Services setup ⭐
├── Makefile                 # Development commands ⭐
└── .env.example             # Configuration template ⭐
```

⭐ = New/significantly updated files

## 🔍 Mock Data

The system includes comprehensive mock metadata:

### Views
- `V_TRANCHE_SYNDICATES` - Syndicate member participation
- `V_USER_METRICS` - User engagement analytics
- `V_TRANSACTION_SUMMARY` - Financial transactions
- `V_DOCUMENT_ACCESS_LOG` - Audit trail
- `V_PORTFOLIO_PERFORMANCE` - Investment metrics

### Reports  
- Syndicate participation analysis
- User activity dashboards
- Transaction reconciliation

### Lookups
- Transaction status codes
- User status codes  
- Currency codes

## 🚨 Important Changes from V1

### ❌ Removed
- All domain-based architecture
- Business domain models and services
- Domain detection and classification
- Hierarchical domain relationships
- Domain-specific prompting

### ✅ Added  
- One-document-per-view model
- Vector similarity retrieval
- HITL approval workflow
- Session state persistence
- Comprehensive test suite
- Docker development stack
- Makefile automation

### 🔄 Simplified
- Clean API with fewer endpoints
- Streamlined prompting
- Direct Bedrock API Gateway integration
- Environment-based configuration

## 🎯 Demo Readiness

This system is optimized for leadership demos:

- ✅ **One-command setup**: `make dev-setup`
- ✅ **Realistic mock data** with financial/business context
- ✅ **Visual approval workflow** via API endpoints
- ✅ **Health checks and monitoring** at `/health` and `/stats`
- ✅ **Interactive API docs** at `/docs`
- ✅ **Comprehensive test coverage** demonstrating reliability
- ✅ **Clean architecture** showing engineering best practices

## 📞 Support

For questions about this system:

1. Check the API documentation at `/docs`
2. Review test examples in `tests/`
3. Check configuration in `.env.example`
4. Run health checks at `/health`

## 🏆 Success Criteria Met

✅ No domain references anywhere in code, data, or prompts  
✅ MongoDB contains one document per view  
✅ OpenSearch contains matching vectorized documents  
✅ Vector dimension inferred at runtime  
✅ Docker compose starts MongoDB + OpenSearch + dashboards  
✅ Seed scripts populate realistic test data  
✅ RAG retrieval surfaces structured metadata to LLM  
✅ HITL endpoints work with state persistence  
✅ All tests pass via single command  
✅ One-command local setup works  
✅ README provides complete setup instructions

