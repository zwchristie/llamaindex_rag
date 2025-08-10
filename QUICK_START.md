# 🚀 **QUICK START GUIDE - Text-to-SQL RAG System**

## ⚡ **5-Minute Setup**

### **1. Start Infrastructure**
```bash
# Start all services (takes ~30 seconds)
docker compose -f docker-compose.full.yml up -d

# Verify all services are running
docker compose -f docker-compose.full.yml ps
```
**Expected:** 4 services running (mongodb, opensearch, opensearch-dashboards, redis)

### **2. Load Sample Data**
```bash
# Load metadata files and create vector embeddings (~2 minutes)
poetry run python scripts/load_sample_data.py
```
**Expected:** "DATA LOADING COMPLETED SUCCESSFULLY!" with document counts

### **3. Test the System**
```bash
# Quick health check (~30 seconds)
poetry run python tests/test_rag_pipeline.py
```
**Expected:** All tests pass ✅

### **4. Start Chatting!**
```bash
# Launch interactive chat interface
poetry run python chat_interface.py
```

## 💬 **Try These Sample Questions:**

```
You: Show me user engagement metrics for active users

System: 🔍 Processing: 'Show me user engagement metrics for active users'
📊 Generating query embedding...
🔎 Searching for relevant database views...
   Found 3 relevant views:
   - V_USER_METRICS (similarity: 0.689)
   - V_USER_ACTIVITY (similarity: 0.542)
   - V_ENGAGEMENT_STATS (similarity: 0.511)
📄 Building database context...
🤖 Generating SQL with Claude 3 Haiku...

Generated SQL:
```sql
SELECT user_id, login_count, session_duration_avg,
       feature_usage_score, last_activity_date
FROM V_USER_METRICS 
WHERE last_activity_date >= DATEADD(month, -3, GETDATE())
AND status = 'ACTIVE'
ORDER BY feature_usage_score DESC;
```

💡 Explanation: This query retrieves user engagement metrics for users who have been active in the last 3 months, ordered by their feature usage score.

⏳ Creating approval request...
📋 Approval request created: req_abc123

👤 Type 'approve' to approve this SQL or 'reject' to reject it.

You: approve

✅ [APPROVED] SQL query has been approved!
🚀 In a production system, this SQL would now be executed.
```

## 🎯 **Other Sample Questions to Try:**

1. **"What are the top performing investment tranches by return percentage?"**
2. **"Show me all syndicate participation data for recent tranches"**  
3. **"List completed transactions with amounts over $100,000"**
4. **"Find users who haven't logged in recently"**
5. **"Get financial performance metrics by quarter"**

## 🔧 **Quick Troubleshooting**

### **If Services Won't Start:**
```bash
# Check Docker is running
docker version

# Restart services
docker compose -f docker-compose.full.yml down
docker compose -f docker-compose.full.yml up -d
```

### **If No Search Results:**
```bash
# Check if data is loaded
curl "localhost:9200/view_metadata/_count?pretty"

# If count is 0, reload data:
poetry run python scripts/load_sample_data.py
```

### **If Bedrock API Errors:**
```bash
# Test endpoint directly
curl -X POST https://8v1n9dbomk.execute-api.us-east-1.amazonaws.com/testaccess \
  -H "Content-Type: application/json" \
  -d '{"model_id": "amazon.titan-embed-text-v2:0", "invoke_type": "embedding", "query": "test"}'
```

### **View Data in Dashboard:**
Open http://localhost:5601 and create an index pattern for `view_metadata*`

## 🎉 **You're Ready!**

Your text-to-SQL RAG system is now running with:
- ✅ Real Amazon Bedrock API integration  
- ✅ MongoDB document storage
- ✅ OpenSearch vector search
- ✅ Human-in-the-loop approval workflow
- ✅ Interactive chat interface

**Next Steps:**
- Replace sample data with your real database schemas
- Customize the system for your specific domain
- Deploy to production environment

**For detailed documentation, see:** `APPLICATION_GUIDE.md`