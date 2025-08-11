# Text-to-SQL RAG System with Hierarchical Retrieval

An advanced text-to-SQL system featuring hierarchical metadata retrieval, Oracle SQL generation, and Human-in-the-Loop (HITL) approval workflow.

## 🎯 Overview

This system transforms natural language questions into Oracle SQL queries using:
- **Hierarchical metadata retrieval** (domains → views → reports → lookups)
- **Business domain intelligence** for context-aware SQL generation
- **Oracle SQL dialect** with schema-qualified table names
- **Human-in-the-Loop approval** for generated SQL
- **Lookup value integration** for accurate WHERE clauses
- **Graph-based metadata relationships** with lookup_id connections

## 🏗️ Architecture

```mermaid
graph TD
    A[User Query] --> B[Hierarchical Context Service]
    B --> C[Step 1: Business Domain Identification]
    C --> D[Step 2: Core View Retrieval]
    D --> E[Step 3: Supporting View Retrieval]
    E --> F[Step 4: Report Examples]
    F --> G[Step 5: Lookup Values]
    G --> H[Oracle SQL Generation]
    H --> I[HITL Approval]
    I --> J[SQL Execution]
    J --> K[Formatted Response]
    
    L[MongoDB] --> M[Business Domains]
    L --> N[View Metadata]
    L --> O[Report Metadata] 
    L --> P[Lookup Metadata]
    L --> Q[Session States]
    L --> R[HITL Requests]
    
    S[OpenSearch] --> T[Vectorized Documents]
    
    U[Bedrock API Gateway] --> V[Claude 3 Haiku]
    U --> W[Titan Embeddings]
```

## 📊 Key Features

### 🔍 Hierarchical Metadata Model
- **Business domains** define knowledge areas with keywords and relationships
- **Views** tagged with business_domains and lookup_id references
- **Reports** provide SQL patterns and examples for specific domains
- **Lookups** connected via lookup_id for accurate value resolution
- **Graph relationships** enable sophisticated context assembly

### 🧠 Intelligent Retrieval Pipeline
- **5-step hierarchical retrieval** starting with domain identification
- **LLM-driven domain classification** using business domain context
- **Semantic similarity ranking** with cosine similarity scoring
- **Parallel execution** for reports and lookups (configurable)
- **Oracle-specific context formatting** with schema qualification

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

## 🔄 Hierarchical Retrieval Process

The system uses a sophisticated 5-step retrieval pipeline:

1. **Domain Identification** (LLM-based)
   - Analyze user query to identify relevant business domains
   - Use domain keywords and descriptions for classification
   - Return prioritized list of domain IDs

2. **Core View Retrieval** (Domain-filtered + Semantic)
   - Filter views by identified domains
   - Rank by semantic similarity to user query
   - Return most relevant core views (configurable limit)

3. **Supporting View Retrieval** (Relationship-based)
   - Find additional views from same domains
   - Exclude already selected core views
   - Rank by semantic relevance

4. **Report Example Retrieval** (Parallel)
   - Find reports related to selected views/domains
   - Provide SQL patterns and examples
   - Include use cases and data patterns

5. **Lookup Value Retrieval** (lookup_id-based)
   - Extract lookup_id references from selected views
   - Retrieve lookup tables and their values
   - Provide context for WHERE clause construction

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

The system includes comprehensive hierarchical metadata:

### Business Domains
- **1. Syndicate Management** - Deal syndication and member participation
- **2. User Analytics** - User behavior and engagement metrics  
- **3. Transaction Processing** - Financial transactions and settlements
- **4. Deal Pipeline** - Deal flow and opportunity management
- **5. Document Management** - Document access and audit trails
- **6. Portfolio Analytics** - Investment performance tracking

### Views (with business_domains tags)
- `V_TRANCHE_SYNDICATES` - Syndicate participation (domains: 1, 2)
- `V_USER_METRICS` - User engagement (domains: 2, 5)
- `V_SYNDICATE_ALLOCATIONS` - Investment allocations (domains: 1, 6)
- `V_DEAL_OVERVIEW` - Deal pipeline overview (domains: 4, 6)

### Reports (domain-specific examples)
- **Syndicate Performance Report** - Participation analysis with SQL examples
- **User Analytics Report** - Engagement metrics with Oracle queries
- **Deal Pipeline Report** - Deal flow analysis with lookups

### Lookups (with lookup_id connections)
- **1. Tranche Statuses** - Connected to syndicate views
- **2. User Roles** - Connected to user management views
- **3. User Statuses** - User state tracking
- **4. Syndicate Roles** - Member role definitions

## 🚀 Advanced Features

### 🎯 Hierarchical Context Retrieval
- **Business Domain Classification**: LLM identifies relevant domains from query
- **Metadata Graph Traversal**: Follows domain → views → reports → lookups relationships
- **Contextual SQL Generation**: Provides comprehensive context for Oracle SQL
- **Lookup Value Integration**: Resolves column values via lookup_id connections

### 🔧 Oracle SQL Optimization
- **Schema-qualified table references** (e.g., SCHEMA.TABLE_NAME)
- **Oracle-specific functions** (SYSDATE, TO_DATE, TO_CHAR, ROWNUM)
- **Lookup ID resolution** for accurate WHERE clauses
- **Case-insensitive matching** with UPPER() functions

### 📊 Advanced Metadata Structure
- **Business domains** with keywords and view relationships
- **View metadata** enhanced with business_domains arrays
- **Report examples** linked to domains and views
- **Lookup tables** with lookup_id cross-references
- **Column-level lookup_id** attributes for value resolution

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

## 🏆 System Capabilities

✅ **Hierarchical metadata retrieval** with 5-step pipeline  
✅ **Business domain intelligence** for context-aware SQL  
✅ **Oracle SQL dialect** with schema-qualified tables  
✅ **Lookup value integration** via lookup_id connections  
✅ **Graph-based metadata relationships** across all types  
✅ **LLM-driven domain classification** with fallback to keywords  
✅ **Parallel report/lookup retrieval** for performance  
✅ **Comprehensive metadata structure** (domains, views, reports, lookups)  
✅ **Vector dimension detection** at runtime  
✅ **HITL approval workflow** with state persistence  
✅ **Comprehensive test coverage** including Oracle SQL validation  
✅ **Docker compose setup** with all required services

