# Deployment Guide

This guide covers various deployment options for the LlamaIndex RAG Text-to-SQL System with OpenSearch.

## Quick Start with Docker Compose

The easiest way to get the entire system running:

### 1. Prerequisites
- Docker and Docker Compose installed
- AWS credentials configured (for Bedrock access)
- At least 4GB RAM available for containers

### 2. Environment Setup
Create a `.env` file with your AWS credentials:

```env
# AWS Configuration (Required)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_LLM_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
AWS_EMBEDDING_MODEL=amazon.titan-embed-text-v1

# Security (Required for production)
SECRET_KEY=your-very-secure-secret-key-here

# Optional overrides
APP_DEBUG=false
OPENSEARCH_INDEX_NAME=documents
MONGODB_DATABASE=text_to_sql_rag
```

### 3. Start Services
```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f text-to-sql-rag
```

### 4. Access Points
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **OpenSearch Dashboards**: http://localhost:5601
- **Health Check**: http://localhost:8000/health

## Individual Service Setup

### OpenSearch

#### Local Installation
```bash
# Download and run OpenSearch
wget https://artifacts.opensearch.org/releases/bundle/opensearch/2.12.0/opensearch-2.12.0-linux-x64.tar.gz
tar -xzf opensearch-2.12.0-linux-x64.tar.gz
cd opensearch-2.12.0
./bin/opensearch
```

#### Docker
```bash
docker run -d \
  --name opensearch \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  opensearchproject/opensearch:2.12.0
```

#### Configuration
- **Endpoint**: http://localhost:9200
- **Index**: `documents` (configurable)
- **Vector Field**: `vector`
- **Dimensions**: 1536 (for Titan embeddings)

### MongoDB

#### Docker
```bash
docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  mongo:7.0
```

#### Configuration
- **Connection**: `mongodb://admin:password@localhost:27017`
- **Database**: `text_to_sql_rag`

### Redis (Optional)

#### Docker
```bash
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:7.2-alpine
```

## Cloud Deployment

### AWS Deployment

#### Option 1: ECS with Fargate
```yaml
# task-definition.json
{
  "family": "text-to-sql-rag",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "text-to-sql-rag",
      "image": "your-account.dkr.ecr.region.amazonaws.com/text-to-sql-rag:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OPENSEARCH_HOST",
          "value": "your-opensearch-domain.region.es.amazonaws.com"
        },
        {
          "name": "OPENSEARCH_PORT",
          "value": "443"
        },
        {
          "name": "OPENSEARCH_USE_SSL",
          "value": "true"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/text-to-sql-rag",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Option 2: Lambda (API only)
```python
# lambda_handler.py
import json
from mangum import Mangum
from src.text_to_sql_rag.api.main import app

handler = Mangum(app, lifespan="off")
```

### OpenSearch Service (AWS)

#### Setup
1. Create OpenSearch domain
2. Configure security policies
3. Enable fine-grained access control
4. Set up VPC if needed

#### Connection Configuration
```env
OPENSEARCH_HOST=your-domain.region.es.amazonaws.com
OPENSEARCH_PORT=443
OPENSEARCH_USE_SSL=true
OPENSEARCH_VERIFY_CERTS=true
```

### DocumentDB (MongoDB-compatible)

#### Setup
```bash
# Create DocumentDB cluster
aws docdb create-db-cluster \
  --db-cluster-identifier text-to-sql-rag \
  --engine docdb \
  --master-username admin \
  --master-user-password your-password
```

#### Connection
```env
MONGODB_URL=mongodb://admin:password@text-to-sql-rag.cluster-xyz.region.docdb.amazonaws.com:27017/?tls=true&replicaSet=rs0&readPreference=secondaryPreferred&retryWrites=false
```

## Production Considerations

### Security

#### Environment Variables
Never commit secrets to version control. Use:
- AWS Parameter Store
- HashiCorp Vault
- Kubernetes Secrets
- Docker Secrets

#### Network Security
```yaml
# docker-compose.override.yml for production
version: '3.8'
services:
  opensearch:
    environment:
      - "DISABLE_SECURITY_PLUGIN=false"
      - "OPENSEARCH_INITIAL_ADMIN_PASSWORD=your-secure-password"
  
  text-to-sql-rag:
    environment:
      - OPENSEARCH_USERNAME=admin
      - OPENSEARCH_PASSWORD=your-secure-password
      - OPENSEARCH_USE_SSL=true
```

### Monitoring

#### Health Checks
```bash
# Application health
curl http://localhost:8000/health

# OpenSearch health
curl http://localhost:9200/_cluster/health

# MongoDB health
mongo --eval "db.adminCommand('ping')"
```

#### Logging
```yaml
# docker-compose.yml logging configuration
services:
  text-to-sql-rag:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Performance

#### Resource Allocation
```yaml
# docker-compose.yml resource limits
services:
  opensearch:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
  
  text-to-sql-rag:
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
```

#### Scaling
```yaml
# docker-compose.yml scaling
services:
  text-to-sql-rag:
    deploy:
      replicas: 3
    environment:
      - REDIS_HOST=redis  # Required for session sharing
```

### Backup Strategy

#### OpenSearch
```bash
# Create snapshot repository
curl -X PUT "localhost:9200/_snapshot/backup_repo" -H 'Content-Type: application/json' -d'
{
  "type": "fs",
  "settings": {
    "location": "/backup"
  }
}'

# Create snapshot
curl -X PUT "localhost:9200/_snapshot/backup_repo/snapshot_1"
```

#### MongoDB
```bash
# Backup
mongodump --uri="mongodb://admin:password@localhost:27017" --db=text_to_sql_rag

# Restore
mongorestore --uri="mongodb://admin:password@localhost:27017" dump/
```

## Troubleshooting

### Common Issues

#### OpenSearch Connection
```bash
# Check OpenSearch status
curl http://localhost:9200/_cat/health?v

# Check indices
curl http://localhost:9200/_cat/indices?v

# Check cluster settings
curl http://localhost:9200/_cluster/settings?pretty
```

#### Memory Issues
```bash
# Increase OpenSearch heap size
echo 'export OPENSEARCH_JAVA_OPTS="-Xms1g -Xmx1g"' >> ~/.bashrc

# Check container memory usage
docker stats
```

#### SSL/TLS Issues
```env
# Disable SSL for development
OPENSEARCH_USE_SSL=false
OPENSEARCH_VERIFY_CERTS=false

# Enable SSL for production
OPENSEARCH_USE_SSL=true
OPENSEARCH_VERIFY_CERTS=true
```

### Debugging

#### Enable Debug Logging
```env
APP_DEBUG=true
LOG_LEVEL=DEBUG
```

#### Access Container Logs
```bash
# View application logs
docker-compose logs -f text-to-sql-rag

# View OpenSearch logs
docker-compose logs -f opensearch

# Execute commands in container
docker-compose exec text-to-sql-rag bash
```

## Migration from Qdrant

If migrating from a Qdrant-based setup:

### 1. Export Data from Qdrant
```python
# export_qdrant.py
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
points = client.scroll(collection_name="documents", limit=10000)
# Save points data for import to OpenSearch
```

### 2. Import to OpenSearch
```python
# import_opensearch.py
from opensearchpy import OpenSearch

client = OpenSearch([{'host': 'localhost', 'port': 9200}])
# Import the exported data with appropriate mapping
```

### 3. Update Configuration
Replace all Qdrant environment variables with OpenSearch equivalents and restart the application.

---

For additional support, refer to the main README.md or create an issue in the repository.