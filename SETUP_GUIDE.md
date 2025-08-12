# Text-to-SQL RAG Application Setup Guide

## Overview
This guide will walk you through setting up, building, and running the text-to-SQL RAG application with all its dependencies.

## Prerequisites
- Python 3.9-3.12
- Poetry (for dependency management)
- Docker (for running OpenSearch and MongoDB)
- Git

## 1. Environment Setup

### Clone and Navigate to Project
```bash
git clone <your-repo-url>
cd llamaindex_rag
```

### Install Dependencies
```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

## 2. Start Required Services

### Start OpenSearch (Vector Database)
```bash
# Start OpenSearch container
docker run -d \
  --name opensearch \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=YourStrongPassword123!" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  opensearchproject/opensearch:latest

# Verify OpenSearch is running
curl http://localhost:9200
```

### Start MongoDB (Document Storage)
```bash
# Start MongoDB container
docker run -d \
  --name mongodb \
  -p 27017:27017 \
  mongo:latest

# Verify MongoDB is running
docker exec mongodb mongosh --eval "db.adminCommand('ping')"
```

## 3. Configuration

### Environment Variables
Update your `.env` file with the following configuration:

```env
# Application Settings
SECRET_KEY=your-secret-key-here
APP_DEBUG=true
LLM_PROVIDER=bedrock

# AWS Bedrock Configuration
AWS_REGION=us-east-1
AWS_USE_PROFILE=true
AWS_PROFILE=adfs
AWS_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
AWS_LLM_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_ENDPOINT_URL=http://localhost:8080/bedrock

# OpenSearch Configuration
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=admin
OPENSEARCH_INDEX_NAME=documents
OPENSEARCH_VECTOR_SIZE=1024

# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=text_to_sql_rag

# Optional: Redis for session management
REDIS_URL=redis://localhost:6379
```

### Bedrock Endpoint Setup
If you don't have a real Bedrock endpoint, you can create a mock service or use your existing endpoint. Make sure the `BEDROCK_ENDPOINT_URL` points to your actual service.

## 4. Build and Start the Application

### Verify Dependencies are Running
```bash
# Check OpenSearch
curl http://localhost:9200/_cluster/health

# Check MongoDB
docker exec mongodb mongosh --eval "db.runCommand('ping')"
```

### Start the FastAPI Application
```bash
# Using Poetry
poetry run python -m uvicorn src.text_to_sql_rag.api.main:app --host 0.0.0.0 --port 8000 --reload

# Alternative: Direct Python
python -m uvicorn src.text_to_sql_rag.api.main:app --host 0.0.0.0 --port 8000 --reload

poetry run python -m src/text_to_sql_rag/api/main.py
```

### Verify Application is Running
```bash
# Health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed
```

## 5. Testing the Application

### Basic Health Checks
```bash
# Basic health
curl http://localhost:8000/health

# Detailed system status
curl http://localhost:8000/health/detailed

# Application statistics
curl http://localhost:8000/stats
```

### Document Management
```bash
# Create a test document
echo "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100), email VARCHAR(100));" > sample_schema.sql

# Upload document
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_schema.sql" \
  -F "title=User Schema" \
  -F "document_type=schema_documentation" \
  -F "description=Sample user table schema"

# List all documents
curl http://localhost:8000/debug/documents

# Search documents
curl -X POST "http://localhost:8000/search/documents" \
  -H "Content-Type: application/json" \
  -d '{"query": "users table", "limit": 5}'
```

### SQL Generation Testing
```bash
# Generate SQL query
curl -X POST "http://localhost:8000/query/generate" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me all users from the database"}'

# Start a conversation
curl -X POST "http://localhost:8000/conversations/start" \
  -H "Content-Type: application/json" \
  -d '{"query": "What tables are available in the database?"}'

# Validate SQL query
curl -X POST "http://localhost:8000/query/validate" \
  -H "Content-Type: application/json" \
  -d '{"sql_query": "SELECT * FROM users WHERE id = 1"}'

# Explain SQL query
curl -X POST "http://localhost:8000/query/explain" \
  -H "Content-Type: application/json" \
  -d '{"sql_query": "SELECT u.name, COUNT(p.id) FROM users u LEFT JOIN posts p ON u.id = p.user_id GROUP BY u.id"}'
```

### LLM Provider Testing
```bash
# Get LLM provider info
curl http://localhost:8000/llm-provider/info

# Test LLM provider
curl http://localhost:8000/llm-provider/test
```

## 6. Troubleshooting

### Common Issues

#### OpenSearch Connection Failed
```bash
# Check if OpenSearch is running
docker ps | grep opensearch

# Check OpenSearch logs
docker logs opensearch

# Restart OpenSearch
docker restart opensearch
```

#### MongoDB Connection Failed
```bash
# Check if MongoDB is running
docker ps | grep mongodb

# Check MongoDB logs
docker logs mongodb

# Restart MongoDB
docker restart mongodb
```

#### Application Won't Start
```bash
# Check Python/Poetry environment
poetry env info

# Reinstall dependencies
poetry install --no-cache

# Check for missing environment variables
poetry run python -c "from src.text_to_sql_rag.config.settings import settings; print(settings)"
```

#### Bedrock Endpoint Issues
- Verify your `BEDROCK_ENDPOINT_URL` is correct
- Check if your Bedrock service is running and accessible
- Ensure AWS credentials are properly configured if using real Bedrock

### Logs and Monitoring
```bash
# View application logs (if using systemd or similar)
tail -f /var/log/text-to-sql-rag.log

# Check Docker container logs
docker logs opensearch
docker logs mongodb

# Monitor resource usage
docker stats
```

## 7. Development Setup

### Running Tests
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific test file
poetry run pytest tests/test_vector_service.py
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

### Development Server
```bash
# Run with auto-reload for development
poetry run uvicorn src.text_to_sql_rag.api.main:app --reload --host 0.0.0.0 --port 8000

# Run with debug logging
APP_DEBUG=true poetry run uvicorn src.text_to_sql_rag.api.main:app --reload --log-level debug
```

## 8. Production Deployment

### Using Docker
Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "src.text_to_sql_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
# Build Docker image
docker build -t text-to-sql-rag .

# Run application container
docker run -d --name text-to-sql-rag \
  --env-file .env \
  -p 8000:8000 \
  text-to-sql-rag
```

### Environment-Specific Configuration
- **Development**: Use `.env.dev`
- **Testing**: Use `.env.test`
- **Production**: Use environment variables or secrets management

## 9. API Documentation

Once the application is running, you can access:
- **Interactive API Documentation**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **OpenAPI JSON Schema**: http://localhost:8000/openapi.json

## 10. Next Steps

1. **Load Sample Data**: Upload your schema documentation and sample queries
2. **Configure Bedrock**: Set up proper AWS Bedrock access if using real service
3. **Customize Models**: Adjust embedding and LLM models based on your needs
4. **Monitor Performance**: Set up logging and monitoring for production use
5. **Scale Services**: Consider using managed services for OpenSearch and MongoDB in production

For more detailed information about specific components, refer to the individual service documentation in the `docs/` directory.