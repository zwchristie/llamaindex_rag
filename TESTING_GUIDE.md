# Text-to-SQL RAG Application Testing Guide

## System Status ✅
Your application is now running successfully with Docker Compose!

- **Application**: http://localhost:8000
- **OpenSearch**: http://localhost:9200  
- **MongoDB**: localhost:27017
- **Redis**: localhost:6379
- **OpenSearch Dashboards**: http://localhost:5601

## Quick Health Check

```bash
# Basic health check
curl -s http://localhost:8000/health

# Detailed system information
curl -s http://localhost:8000/health/detailed

# Application statistics
curl -s http://localhost:8000/stats
```

## Core API Testing

### 1. Health and System Status

```bash
# Basic health
curl http://localhost:8000/health

# Detailed health with service info
curl http://localhost:8000/health/detailed

# Application statistics
curl http://localhost:8000/stats

# LLM Provider information
curl http://localhost:8000/llm-provider/info

# Test LLM provider connectivity
curl http://localhost:8000/llm-provider/test
```

### 2. Document Management

#### Upload Documents
```bash
# Create a sample SQL schema file
cat > sample_schema.sql << EOF
-- User management tables
CREATE TABLE users (
    id INT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE user_profiles (
    user_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    bio TEXT,
    avatar_url VARCHAR(200),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Product catalog tables
CREATE TABLE categories (
    id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT
);

CREATE TABLE products (
    id INT PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category_id INT,
    price DECIMAL(10,2),
    stock_quantity INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES categories(id)
);

-- Order management tables
CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT NOT NULL,
    total_amount DECIMAL(10,2),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE order_items (
    id INT PRIMARY KEY,
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity INT NOT NULL,
    unit_price DECIMAL(10,2),
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);
EOF

# Upload the schema document
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_schema.sql" \
  -F "title=E-commerce Database Schema" \
  -F "document_type=schema_documentation" \
  -F "description=Complete database schema for e-commerce application"
```

#### Document Operations
```bash
# List all documents
curl http://localhost:8000/debug/documents

# Get specific document info (replace {document_id} with actual ID from upload response)
curl http://localhost:8000/documents/{document_id}

# Search documents by content
curl -X POST "http://localhost:8000/search/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "user table with email and username",
    "limit": 5
  }'

# Search for product-related documents
curl -X POST "http://localhost:8000/search/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "products and categories",
    "limit": 3
  }'
```

### 3. SQL Generation Testing

#### Basic SQL Generation
```bash
# Simple query generation
curl -X POST "http://localhost:8000/query/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me all users with their email addresses"
  }'

# Complex query with joins
curl -X POST "http://localhost:8000/query/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Get all orders with user information and total amount, ordered by date"
  }'

# Aggregation query
curl -X POST "http://localhost:8000/query/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Count how many products are in each category"
  }'

# Query with filters
curl -X POST "http://localhost:8000/query/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find all orders with status pending that have a total amount greater than 100"
  }'
```

#### SQL Query Utilities
```bash
# Validate a SQL query
curl -X POST "http://localhost:8000/query/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "sql_query": "SELECT u.username, u.email FROM users u WHERE u.id = 1"
  }'

# Explain what a SQL query does
curl -X POST "http://localhost:8000/query/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "sql_query": "SELECT u.username, p.name, oi.quantity, oi.unit_price FROM users u JOIN orders o ON u.id = o.user_id JOIN order_items oi ON o.id = oi.order_id JOIN products p ON oi.product_id = p.id"
  }'

# Generate and execute SQL (if execution service is available)
curl -X POST "http://localhost:8000/query/generate-and-execute" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Count total number of users"
  }'
```

### 4. Conversation Management (HITL - Human-in-the-Loop)

#### Start and Manage Conversations
```bash
# Start a new conversation
curl -X POST "http://localhost:8000/conversations/start" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What tables are available in this database?",
    "context": {"user_id": "test_user", "session": "demo"}
  }'

# Continue a conversation (replace {conversation_id} with actual ID from start response)
curl -X POST "http://localhost:8000/conversations/{conversation_id}/continue" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Can you show me how to get user details with their profiles?"
  }'

# Get conversation details
curl http://localhost:8000/conversations/{conversation_id}

# Get conversation status
curl http://localhost:8000/conversations/{conversation_id}/status

# List all conversations
curl http://localhost:8000/conversations

# List conversations with specific status
curl "http://localhost:8000/conversations?status=waiting_for_clarification"

# SQL description within conversation context
curl -X POST "http://localhost:8000/conversations/{conversation_id}/describe-sql" \
  -H "Content-Type: application/json" \
  -d '{
    "sql_query": "SELECT * FROM users u JOIN user_profiles up ON u.id = up.user_id"
  }'
```

### 5. Session Management

#### Session Operations
```bash
# Create a new session
curl -X POST "http://localhost:8000/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_query": "I need help with database queries",
    "user_id": "demo_user",
    "context": {"department": "engineering", "role": "developer"}
  }'

# Get session information (replace {session_id} with actual ID)
curl http://localhost:8000/sessions/{session_id}
```

## Advanced Testing Scenarios

### 1. Complex Business Queries
```bash
# Revenue analysis
curl -X POST "http://localhost:8000/query/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Calculate total revenue by category for the last month, showing category name and revenue amount"
  }'

# Customer behavior analysis
curl -X POST "http://localhost:8000/query/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find customers who have made more than 5 orders and their average order value"
  }'

# Inventory management
curl -X POST "http://localhost:8000/query/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show products that are low in stock (less than 10 items) along with their category"
  }'
```

### 2. Error Handling Tests
```bash
# Invalid query test
curl -X POST "http://localhost:8000/query/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": ""
  }'

# Malformed SQL validation
curl -X POST "http://localhost:8000/query/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "sql_query": "SELECT * FROM non_existent_table WHERE invalid syntax"
  }'

# Non-existent conversation
curl http://localhost:8000/conversations/invalid-id-12345
```

### 3. Load Testing (Optional)
```bash
# Multiple concurrent requests
for i in {1..10}; do
  curl -X POST "http://localhost:8000/query/generate" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"Get user count - request $i\"}" &
done
wait
```

## Monitoring and Debugging

### Check Service Health
```bash
# OpenSearch health
curl http://localhost:9200/_cluster/health

# MongoDB health (requires mongosh in container)
docker exec mongodb mongosh --eval "db.adminCommand('ping')"

# Redis health
docker exec redis redis-cli ping

# Application logs
docker logs text-to-sql-rag-app --tail 50

# All container status
docker compose -f docker-compose.full.yml ps
```

### Performance Monitoring
```bash
# Container resource usage
docker stats

# Application metrics (if available)
curl http://localhost:8000/stats

# OpenSearch index statistics
curl http://localhost:9200/view_metadata/_stats
```

## Integration Testing with External Tools

### Using curl with jq for JSON processing
```bash
# Pretty print health response
curl -s http://localhost:8000/health | jq '.'

# Extract specific fields
curl -s http://localhost:8000/stats | jq '.vector_store'

# Test and extract conversation ID
CONV_ID=$(curl -s -X POST "http://localhost:8000/conversations/start" \
  -H "Content-Type: application/json" \
  -d '{"query": "Test conversation"}' | jq -r '.conversation_id')

echo "Created conversation: $CONV_ID"
```

### Using Python for complex testing
```python
import requests
import json

# Test script example
base_url = "http://localhost:8000"

# Health check
response = requests.get(f"{base_url}/health")
print("Health:", response.json())

# Generate SQL
query_response = requests.post(f"{base_url}/query/generate", 
                              json={"query": "Show all users"})
print("SQL Generation:", query_response.json())
```

## Expected Responses

### Healthy System Response
```json
{
  "status": "healthy",
  "vector_store": "connected",
  "execution_service": "disconnected",
  "mongodb": "healthy",
  "document_sync": {
    "status": "direct_mongodb_sync",
    "method": "mongodb_to_vector_store"
  },
  "version": "2.0.0",
  "timestamp": "2025-08-10T04:23:21.123456"
}
```

### Successful SQL Generation
```json
{
  "conversation_id": "uuid-string",
  "sql": "SELECT username, email FROM users;",
  "explanation": "This query retrieves the username and email from all users in the users table.",
  "confidence": 0.95,
  "response_type": "sql_result",
  "status": "completed"
}
```

## Troubleshooting Common Issues

1. **"Services are still initializing"** - Wait 30-60 seconds for full startup
2. **Vector store not connected** - Check OpenSearch is running on port 9200
3. **MongoDB connection failed** - Ensure MongoDB container is healthy
4. **Bedrock endpoint errors** - Verify your endpoint URL and credentials
5. **Out of memory** - Increase Docker memory limits for large documents

## Next Steps

1. Upload your actual database schema documents
2. Test with your specific business queries
3. Configure proper authentication if needed
4. Set up monitoring and logging for production
5. Scale services based on usage patterns

Your text-to-SQL RAG system is now fully operational! 🚀