# Text-to-SQL RAG Application

An agentic text-to-SQL RAG (Retrieval-Augmented Generation) solution built with LlamaIndex, Qdrant, and AWS Bedrock. This application enables natural language querying of databases by generating SQL queries using context from uploaded documents and schema information.

## Features

- **Document Management**: Upload and manage two types of documents:
  - **Report Documents**: SQL query examples with descriptions and expected outputs
  - **Schema Documents**: Database schema information with tables, columns, and relationships
- **Hybrid Retrieval**: Uses LlamaIndex's hybrid search combining vector similarity and other retrieval methods
- **SQL Generation**: Generates SQL queries from natural language using RAG with AWS Bedrock LLMs
- **Query Execution**: Integrates with external database execution service via API
- **Query Validation**: Validates generated queries and suggests fixes for errors
- **Session Management**: Tracks user sessions for human-in-the-loop workflows
- **Scalable Architecture**: Vector store with Qdrant, embeddings and LLM via AWS Bedrock

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   LlamaIndex    │    │     Qdrant      │
│                 │    │   RAG Pipeline  │    │  Vector Store   │
│ - Document API  │◄──►│                 │◄──►│                 │
│ - Query API     │    │ - Hybrid Search │    │ - Embeddings    │
│ - Session API   │    │ - SQL Generation│    │ - Metadata      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│  External DB    │    │   AWS Bedrock   │
│ Execution API   │    │                 │
│ - Query Exec    │    │ - Embeddings    │
│ - Validation    │    │ - LLM (Claude)  │
│ - Schema Info   │    │ - Text Gen      │
└─────────────────┘    └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- Poetry for dependency management
- Qdrant instance (local or cloud)
- AWS account with Bedrock access
- External database execution service

### Setup

1. **Clone and install dependencies:**
```bash
git clone <repository-url>
cd llamaindex_proj
poetry install
```

2. **Configure environment variables:**
```bash
# AWS Bedrock
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_EMBEDDING_MODEL=amazon.titan-embed-text-v1
export AWS_LLM_MODEL=anthropic.claude-3-sonnet-20240229-v1:0

# Qdrant
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export QDRANT_COLLECTION_NAME=documents

# External Execution API
export EXECUTION_API_URL=http://localhost:8001

# App Settings
export APP_DEBUG=true
export CHUNK_SIZE=1024
export CHUNK_OVERLAP=200
export SIMILARITY_TOP_K=5
```

3. **Start Qdrant (if running locally):**
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

4. **Run the application:**
```bash
poetry run python -m uvicorn src.text_to_sql_rag.api.main:app --reload --port 8000
```

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

### Key Endpoints

#### Document Management
- `POST /documents/upload` - Upload document (report or schema)
- `GET /documents/{document_id}` - Get document info
- `DELETE /documents/{document_id}` - Delete document
- `POST /documents/search` - Search documents

#### Query Generation & Execution
- `POST /query/generate` - Generate SQL from natural language
- `POST /query/generate-and-execute` - Generate and optionally execute SQL
- `POST /query/execute` - Execute SQL query via external service
- `POST /query/validate` - Validate SQL and get suggestions
- `POST /query/explain` - Explain what a SQL query does

#### Session Management
- `POST /sessions` - Create user session
- `GET /sessions/{session_id}` - Get session details

#### Monitoring
- `GET /health` - Health check
- `GET /health/detailed` - Detailed service status
- `GET /stats` - Application statistics

## Usage Examples

### 1. Upload Schema Document

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@schema.json" \
  -F "title=Customer Database Schema" \
  -F "document_type=schema" \
  -F "description=Main customer and orders tables"
```

### 2. Upload Report Document

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@report_examples.sql" \
  -F "title=Sales Report Queries" \
  -F "document_type=report" \
  -F "description=Example queries for sales analytics"
```

### 3. Generate SQL Query

```bash
curl -X POST "http://localhost:8000/query/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me the top 5 customers by total order value",
    "use_hybrid_retrieval": true,
    "session_id": "optional-session-id"
  }'
```

### 4. Generate and Execute Query

```bash
curl -X POST "http://localhost:8000/query/generate-and-execute" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many orders were placed last month?",
    "auto_execute": true,
    "use_hybrid_retrieval": true
  }'
```

## Configuration

### Document Types

The application supports two document types:

#### Report Documents
- **Purpose**: Provide example SQL queries with context
- **Content**: SQL queries with descriptions and use cases
- **Metadata**: Expected output descriptions, complexity levels, related tables

#### Schema Documents  
- **Purpose**: Provide database structure information
- **Content**: Table definitions, column information, relationships
- **Metadata**: Table names, column details, constraints, indexes

### RAG Configuration

Key settings in `config.yaml`:
- `chunk_size`: Text chunk size for embeddings (default: 1024)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `similarity_top_k`: Number of similar chunks to retrieve (default: 5)

### Hybrid Retrieval

The application uses LlamaIndex's hybrid retrieval which combines:
- Vector similarity search
- Query fusion (multiple query variants)
- Metadata filtering
- Re-ranking strategies

## External Dependencies

### Database Execution Service

The application expects an external service at `/execute/query` endpoint that:
- Accepts SQL queries for execution
- Returns results or error messages
- Provides query validation at `/validate/query`
- Offers schema information at `/schema/info`

Example execution service response:
```json
{
  "success": true,
  "data": [
    {"customer_id": 1, "total_value": 1500.00},
    {"customer_id": 2, "total_value": 1200.00}
  ],
  "execution_time_ms": 45,
  "rows_returned": 2
}
```

### AWS Bedrock Models

Supported models:
- **Embeddings**: Amazon Titan, Cohere Embed
- **LLM**: Anthropic Claude, Amazon Titan, Meta Llama

## Development

### Project Structure

```
src/text_to_sql_rag/
├── api/                 # FastAPI application
│   └── main.py         # Main API endpoints
├── core/               # Core business logic
│   └── rag_pipeline.py # RAG pipeline implementation
├── services/           # External service integrations
│   ├── vector_service.py      # LlamaIndex + Qdrant
│   ├── bedrock_service.py     # AWS Bedrock (legacy)
│   └── query_execution_service.py  # External DB API
├── models/             # Data models
│   └── simple_models.py       # Pydantic models
├── config/             # Configuration
│   └── settings.py     # Settings management
└── utils/              # Utilities
    └── content_processor.py   # Document processing
```

### Running Tests

```bash
poetry run pytest tests/
```

### Code Quality

```bash
# Format code
poetry run black src/

# Sort imports
poetry run isort src/

# Type checking
poetry run mypy src/

# Linting
poetry run flake8 src/
```

## Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev

COPY . .
EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "src.text_to_sql_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables for Production

```bash
# Security
SECRET_KEY=your-secret-key-here
APP_DEBUG=false

# Services
QDRANT_HOST=your-qdrant-host
QDRANT_API_KEY=your-qdrant-api-key
EXECUTION_API_URL=https://your-db-service.com

# AWS
AWS_REGION=us-east-1
# Use IAM roles instead of keys in production
```

## Monitoring

### Health Checks
- `/health` - Basic health status
- `/health/detailed` - Comprehensive service status
- `/stats` - Application metrics

### Logging

Uses structured logging with `structlog`. Configure log level:
```bash
export LOG_LEVEL=INFO
```

## Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**
   - Check if Qdrant is running
   - Verify host/port configuration
   - Check API key if using cloud

2. **AWS Bedrock Access Denied**
   - Verify AWS credentials
   - Check IAM permissions for Bedrock
   - Ensure model access is enabled

3. **Document Upload Fails**
   - Check file size limits
   - Verify file encoding (UTF-8)
   - Check allowed file types

4. **Query Generation Poor Quality**
   - Upload more relevant schema documents
   - Add example queries as report documents
   - Adjust similarity thresholds

### Performance Tuning

- Adjust `chunk_size` and `chunk_overlap` for better retrieval
- Increase `similarity_top_k` for more context
- Use hybrid retrieval for better accuracy
- Monitor vector store performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run code quality checks
5. Submit a pull request

## License

MIT License - see LICENSE file for details.