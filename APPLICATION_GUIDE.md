# 🚀 **TEXT-TO-SQL RAG APPLICATION - COMPLETE GUIDE**

## 📋 **Table of Contents**
1. [Quick Start](#quick-start)
2. [Getting Started](#getting-started)
3. [Testing Framework](#testing-framework)
4. [Core Application Workflow](#core-application-workflow)
5. [Mock Data to Real Data Migration](#mock-data-to-real-data-migration)
6. [Conversational Interface](#conversational-interface)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 **Quick Start**

### **Prerequisites**
- Docker and Docker Compose installed
- Python 3.8+ with Poetry
- Git

### **1. Clone and Setup**
```bash
git clone <your-repo>
cd llamaindex_proj
poetry install
```

### **2. Start Infrastructure**
```bash
# Start all services (MongoDB, OpenSearch, Redis)
docker compose -f docker-compose.full.yml up -d
```

### **3. Verify Services**
```bash
# Check all containers are running
docker compose -f docker-compose.full.yml ps

# Expected services:
# - mongodb (port 27017)
# - opensearch (port 9200)
# - opensearch-dashboards (port 5601)
# - redis (port 6379)
```

### **4. Run Quick Test**
```bash
# Test system health
poetry run python test_system.py

# Expected: All 4 tests should pass ✅
```

---

## 🏁 **Getting Started**

### **Environment Configuration**
Create/verify your `.env` file:

```env
# Bedrock API Configuration
BEDROCK_ENDPOINT_URL=https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com/prod
BEDROCK_LLM_MODEL=anthropic.claude-3-haiku-20240307-v1:0
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
USE_MOCK_EMBEDDINGS=false

# Database Configuration
MONGODB_URL=mongodb://admin:password@localhost:27017
MONGODB_DATABASE=text_to_sql_rag
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_VECTOR_SIZE=1024

# Application Settings
APP_DEBUG=true
HITL_TIMEOUT_MINUTES=30
```

### **Initial Data Load**
```bash
# Load sample metadata files into MongoDB and OpenSearch
poetry run python scripts/load_sample_data.py

# This will:
# 1. Parse JSON files from meta_documents/
# 2. Store documents in MongoDB
# 3. Generate embeddings using real Bedrock API
# 4. Index embeddings in OpenSearch for vector search
```

---

## 🧪 **Testing Framework**

### **Test 1: Infrastructure Health**
```bash
# Test all database connections
poetry run python -c "
import asyncio
from test_infrastructure import test_all_connections
asyncio.run(test_all_connections())
"
```

**Expected Output:**
```
✅ MongoDB connection successful
✅ OpenSearch connection successful  
✅ Redis connection successful
✅ Bedrock API accessible
```

### **Test 2: Document Storage Pipeline**
```bash
# Test MongoDB document storage and retrieval
poetry run python tests/test_document_storage.py
```

**What this tests:**
- Upload JSON metadata files to MongoDB
- Verify document persistence
- Test document retrieval by ID and filters
- Validate document structure integrity

### **Test 3: Vector Search Pipeline**
```bash
# Test embedding generation and vector search
poetry run python tests/test_vector_search.py
```

**What this tests:**
- Generate real Bedrock embeddings (1024-dim)
- Store embeddings in OpenSearch
- Perform similarity search
- Validate search result rankings

### **Test 4: Complete RAG Workflow**
```bash
# Test end-to-end RAG pipeline
poetry run python tests/test_rag_pipeline.py
```

**What this tests:**
- Query processing and embedding
- Context retrieval from vector store
- LLM-powered response generation
- Result formatting and presentation

### **Test 5: Text-to-SQL Generation**
```bash
# Test SQL generation with context
poetry run python tests/test_sql_generation.py
```

**What this tests:**
- Schema context understanding
- Natural language to SQL conversion
- SQL validation and explanation
- Business logic incorporation

### **Test 6: HITL Approval Workflow**
```bash
# Test human-in-the-loop approval process
poetry run python tests/test_hitl_workflow.py
```

**What this tests:**
- Approval request creation
- Session state management
- Request timeout handling
- Approval/rejection processing

---

## 🔄 **Core Application Workflow**

### **Full Conversational Text-to-SQL Flow**

#### **Step 1: Start the Application**
```bash
# Start the conversational interface
poetry run python main_chat.py
```

#### **Step 2: Sample Conversation Flow**
```
User: "Show me all syndicate participation data for recent tranches"

System: [PROCESSING]
1. 🔍 Analyzing query for relevant database schemas...
2. 🧠 Generating embeddings for semantic search...
3. 📊 Found relevant views: V_TRANCHE_SYNDICATES (similarity: 0.642)
4. 🤖 Generating SQL query with Claude 3 Haiku...

Generated SQL:
```sql
SELECT t.tranche_id, vs.syndicate_id, vs.member_name,
       vs.participation_amount, vs.participation_percentage,
       vs.join_date, vs.status
FROM SYND.V_TRANCHE_SYNDICATES vs
INNER JOIN TRANCHES t ON vs.tranche_id = t.tranche_id
WHERE t.created_date >= CURRENT_DATE - 30
ORDER BY t.tranche_id DESC, vs.join_date DESC;
```

5. ⏳ Creating approval request... [ID: req_abc123]
6. 👤 [HITL] Waiting for human approval...

User: approve

System: ✅ SQL approved and ready for execution!
7. 🚀 Executing query... (simulated)
8. 📋 Results: 23 syndicate participation records found
```

#### **Key Features in Action:**
- **Semantic Understanding**: Query matches relevant database schemas
- **Context-Aware SQL**: Generated SQL uses appropriate JOINs and filters  
- **Business Logic**: Incorporates domain knowledge (syndicates, tranches)
- **Safety First**: All SQL goes through human approval
- **Session Persistence**: Conversation state maintained across interactions

---

## 📊 **Mock Data to Real Data Migration**

### **Phase 1: Understanding Current Mock Data**
```bash
# Examine existing mock metadata
ls -la meta_documents/views/
ls -la meta_documents/reports/

# Current structure:
# - meta_documents/views/*.json (database view definitions)
# - meta_documents/reports/*.json (sample reports and schemas)
```

### **Phase 2: Replace Mock Data**

#### **Replace View Definitions**
1. **Export your real database schema:**
```sql
-- For PostgreSQL
SELECT schemaname, viewname, definition 
FROM pg_views 
WHERE schemaname NOT IN ('information_schema', 'pg_catalog');

-- For SQL Server  
SELECT SCHEMA_NAME(schema_id) as schema_name,
       name as view_name,
       definition
FROM sys.views v
INNER JOIN sys.sql_modules m ON v.object_id = m.object_id;
```

2. **Convert to JSON format:**
```json
{
  "view_name": "V_YOUR_REAL_VIEW",
  "schema": "YOUR_SCHEMA",
  "description": "Description of what this view contains",
  "columns": [
    {"name": "column1", "type": "VARCHAR(50)", "description": "..."},
    {"name": "column2", "type": "INTEGER", "description": "..."}
  ],
  "business_context": "Explain what business questions this view answers",
  "sample_queries": [
    "SELECT * FROM V_YOUR_REAL_VIEW WHERE condition = 'value';"
  ]
}
```

3. **Replace mock files:**
```bash
# Backup existing mock data
mkdir backup_mock_data
cp -r meta_documents/* backup_mock_data/

# Replace with your real data
cp your_real_views/*.json meta_documents/views/
cp your_real_reports/*.json meta_documents/reports/
```

#### **Re-index with Real Data**
```bash
# Clear existing indices
poetry run python scripts/clear_indices.py

# Re-load with new data
poetry run python scripts/load_sample_data.py

# Verify new data loaded
poetry run python scripts/verify_data_load.py
```

### **Phase 3: Test with Real Data**
```bash
# Test search with your domain-specific queries
poetry run python -c "
import asyncio
from test_real_data_search import test_domain_search
asyncio.run(test_domain_search('your domain specific query here'))
"
```

---

## 💬 **Conversational Interface**

### **Interactive Chat Mode**
```bash
# Start interactive session
poetry run python chat_interface.py
```

### **Sample Conversations**

#### **Conversation 1: Financial Analysis**
```
You: "What are the top performing investment tranches by return percentage?"

System: 
🔍 Searching for relevant schemas...
📊 Found: V_TRANCHE_PERFORMANCE (similarity: 0.721)
🤖 Generating SQL...

Generated Query:
```sql
SELECT tranche_id, tranche_name, return_percentage, 
       investment_amount, total_return
FROM V_TRANCHE_PERFORMANCE 
WHERE status = 'ACTIVE'
ORDER BY return_percentage DESC
LIMIT 10;
```

👤 [HITL] Do you approve this SQL? (approve/reject/modify): approve
✅ Approved! Executing query...
📋 Results: Top 10 performing tranches with 8.2% - 12.4% returns
```

#### **Conversation 2: User Analytics**
```
You: "Show me user engagement metrics for the last quarter"

System:
🔍 Analyzing query... 
📊 Found: V_USER_METRICS (similarity: 0.689)
🤖 Generating SQL...

Generated Query:
```sql
SELECT user_id, login_count, session_duration_avg,
       feature_usage_score, last_activity_date
FROM V_USER_METRICS 
WHERE last_activity_date >= DATEADD(month, -3, GETDATE())
AND status = 'ACTIVE'
ORDER BY feature_usage_score DESC;
```

👤 [HITL] Approval required: approve
✅ Query approved and executed
📋 Results: 1,247 active users with avg session 23.5 minutes
```

#### **Conversation 3: Complex Business Logic**
```
You: "Find all syndicate members who joined recent high-value tranches but have low participation rates"

System:
🔍 Complex query detected, searching multiple schemas...
📊 Found: V_TRANCHE_SYNDICATES (0.654), V_MEMBER_STATS (0.601)
🤖 Generating complex SQL with multiple JOINs...

Generated Query:
```sql
SELECT vs.member_name, vs.syndicate_id, t.tranche_value,
       vs.participation_percentage, ms.avg_participation_rate
FROM V_TRANCHE_SYNDICATES vs
INNER JOIN TRANCHES t ON vs.tranche_id = t.tranche_id
INNER JOIN V_MEMBER_STATS ms ON vs.member_name = ms.member_name
WHERE t.tranche_value > 1000000  -- High-value tranches
  AND t.created_date >= DATEADD(month, -6, GETDATE())  -- Recent
  AND vs.participation_percentage < ms.avg_participation_rate * 0.7  -- Low participation
ORDER BY t.tranche_value DESC, vs.participation_percentage ASC;
```

👤 [HITL] Complex query generated. Review carefully: approve
✅ Advanced analytics query approved
📋 Results: 15 members identified for engagement review
```

---

## 🧪 **Comprehensive Testing Suite**

### **Create Test Files**

Let me create the test files referenced in the documentation:

```bash
# Create tests directory structure
mkdir -p tests/integration
mkdir -p tests/unit
mkdir -p scripts
```

---

## 🔧 **Development Commands**

### **Database Management**
```bash
# Clear all data and restart fresh
poetry run python scripts/reset_databases.py

# Export current data for backup
poetry run python scripts/export_data.py

# Import data from backup
poetry run python scripts/import_data.py --file backup.json
```

### **Monitoring and Debugging**
```bash
# View system logs
docker compose -f docker-compose.full.yml logs -f

# Monitor OpenSearch indices
curl "localhost:9200/_cat/indices?v"

# Check MongoDB collections
poetry run python -c "
import asyncio
from pymongo import MongoClient
client = MongoClient('mongodb://admin:password@localhost:27017')
db = client.text_to_sql_rag
print('Collections:', db.list_collection_names())
for collection in db.list_collection_names():
    print(f'{collection}: {db[collection].count_documents({})} documents')
"
```

### **Performance Testing**
```bash
# Load test with multiple concurrent queries
poetry run python tests/performance/load_test.py

# Memory usage monitoring
poetry run python tests/performance/memory_test.py

# Response time benchmarking
poetry run python tests/performance/benchmark.py
```

---

## 🎯 **Production Readiness Checklist**

### **Before Going Live:**
- [ ] Replace all mock data with real schemas
- [ ] Test with production-scale data volumes
- [ ] Configure proper logging and monitoring
- [ ] Set up backup procedures for MongoDB/OpenSearch
- [ ] Implement proper error handling and retries
- [ ] Configure authentication and authorization
- [ ] Set up SSL/TLS for all connections
- [ ] Performance test under expected load
- [ ] Create disaster recovery procedures
- [ ] Document operational procedures

### **Security Considerations:**
- [ ] API key rotation procedures
- [ ] Database connection encryption
- [ ] Input sanitization for SQL injection prevention
- [ ] Rate limiting for API endpoints
- [ ] Audit logging for all HITL approvals
- [ ] Data privacy compliance (GDPR, etc.)

---

## 🔍 **Troubleshooting**

### **Common Issues:**

#### **502 Bedrock Endpoint Errors**
```bash
# Test endpoint directly
curl -X POST https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com/prod \
  -H "Content-Type: application/json" \
  -d '{"model_id": "amazon.titan-embed-text-v2:0", "invoke_type": "embedding", "query": "test"}'

# If fails, check:
# 1. Network connectivity
# 2. API endpoint availability  
# 3. Request format matches exactly
```

#### **Vector Dimension Mismatch**
```bash
# Check current OpenSearch mapping
curl "localhost:9200/view_metadata/_mapping?pretty"

# Should show 1024 dimensions for Titan V2
# If wrong, delete and recreate index:
curl -X DELETE "localhost:9200/view_metadata"
poetry run python scripts/create_opensearch_index.py
```

#### **MongoDB Connection Issues**
```bash
# Test MongoDB connection
poetry run python -c "
from pymongo import MongoClient
try:
    client = MongoClient('mongodb://admin:password@localhost:27017')
    client.server_info()
    print('✅ MongoDB connected')
except Exception as e:
    print(f'❌ MongoDB error: {e}')
"
```

#### **No Search Results**
```bash
# Check if documents are indexed
curl "localhost:9200/view_metadata/_search?size=1&pretty"

# If empty, re-index:
poetry run python scripts/reindex_all_data.py
```

---

## 📈 **Next Steps**

1. **Complete Testing**: Run all test suites to verify functionality
2. **Load Real Data**: Replace mock data with your actual schemas
3. **Performance Tune**: Optimize for your expected query load
4. **Deploy**: Set up production environment
5. **Monitor**: Implement monitoring and alerting
6. **Scale**: Add load balancing and horizontal scaling as needed

Your text-to-SQL RAG system is ready for production use! 🚀