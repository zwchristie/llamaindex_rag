# 🔄 **REPLACE MOCK DATA WITH YOUR OWN - COMPLETE GUIDE**

## 🎯 **Overview**

This guide walks you through replacing the sample mock data with your real database schemas to make the Text-to-SQL RAG system work with your actual data.

---

## 📊 **Current Mock Data Structure**

The system currently uses sample data in the `meta_documents/` folder:

```
meta_documents/
├── views/                     # Database view definitions (JSON files)
│   ├── v_tranche_syndicates.json
│   ├── v_user_metrics.json  
│   ├── v_transaction_summary.json
│   └── ...
├── reports/                   # Sample reports and contexts
│   └── sample_report.json
└── README.md                  # Description of data structure
```

---

## 🏗️ **Step 1: Export Your Database Schema**

### **For PostgreSQL:**
```sql
-- Export all views with their definitions
SELECT 
    schemaname,
    viewname,
    definition,
    -- Get column information
    (SELECT json_agg(
        json_build_object(
            'name', column_name,
            'type', data_type,
            'description', ''
        )
    ) FROM information_schema.columns 
     WHERE table_schema = schemaname 
       AND table_name = viewname
    ) as columns
FROM pg_views 
WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
ORDER BY schemaname, viewname;
```

### **For SQL Server:**
```sql
-- Export views with column information
SELECT 
    SCHEMA_NAME(v.schema_id) as schema_name,
    v.name as view_name,
    m.definition,
    -- Column details in separate query
    (SELECT 
        c.name as column_name,
        t.name as data_type,
        c.max_length
     FROM sys.columns c
     JOIN sys.types t ON c.user_type_id = t.user_type_id
     WHERE c.object_id = v.object_id
     FOR JSON PATH
    ) as columns_json
FROM sys.views v
JOIN sys.sql_modules m ON v.object_id = m.object_id
ORDER BY schema_name, view_name;
```

### **For MySQL:**
```sql
-- Export view definitions
SELECT 
    TABLE_SCHEMA as schema_name,
    TABLE_NAME as view_name,
    VIEW_DEFINITION as definition,
    -- Get column information
    (SELECT JSON_ARRAYAGG(
        JSON_OBJECT(
            'name', COLUMN_NAME,
            'type', DATA_TYPE,
            'description', ''
        )
    ) FROM INFORMATION_SCHEMA.COLUMNS 
     WHERE TABLE_SCHEMA = v.TABLE_SCHEMA 
       AND TABLE_NAME = v.TABLE_NAME
    ) as columns
FROM INFORMATION_SCHEMA.VIEWS v
WHERE TABLE_SCHEMA NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
ORDER BY schema_name, view_name;
```

---

## 📝 **Step 2: Convert to RAG System Format**

### **Required JSON Structure:**

Each view needs to be saved as a JSON file with this structure:

```json
{
  "view_name": "V_YOUR_VIEW_NAME",
  "schema": "YOUR_SCHEMA",
  "description": "Detailed description of what this view contains and its purpose",
  "columns": [
    {
      "name": "column1",
      "type": "VARCHAR(100)",
      "description": "What this column represents"
    },
    {
      "name": "column2", 
      "type": "INTEGER",
      "description": "What this column represents"
    }
  ],
  "business_context": "Explain what business questions this view answers",
  "sample_queries": [
    "SELECT * FROM V_YOUR_VIEW_NAME WHERE condition = 'value';",
    "SELECT column1, COUNT(*) FROM V_YOUR_VIEW_NAME GROUP BY column1;"
  ],
  "relationships": [
    "Joins with TABLE_A on column_x",
    "Related to V_OTHER_VIEW through shared keys"
  ]
}
```

### **Example Conversion Script:**

Create a Python script to help convert your exported schema:

```python
#!/usr/bin/env python3
"""Convert database schema export to RAG system format."""

import json
import re
from pathlib import Path

def convert_postgres_export_to_json(sql_export_file, output_dir):
    """Convert PostgreSQL schema export to JSON files."""
    
    with open(sql_export_file, 'r') as f:
        lines = f.readlines()
    
    current_view = None
    views = []
    
    for line in lines:
        # Parse your SQL export format here
        # This is a template - adjust based on your export format
        if 'VIEW' in line.upper():
            # Extract view information
            pass
    
    # Convert each view to JSON format
    for view_data in views:
        view_json = {
            "view_name": view_data['name'],
            "schema": view_data['schema'],
            "description": f"Database view: {view_data['name']}",
            "columns": view_data.get('columns', []),
            "business_context": "TODO: Add business context for this view",
            "sample_queries": [
                f"SELECT * FROM {view_data['schema']}.{view_data['name']} LIMIT 10;"
            ],
            "relationships": []
        }
        
        # Save to file
        filename = f"{view_data['name'].lower()}.json"
        output_path = Path(output_dir) / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(view_json, f, indent=2, ensure_ascii=False)
        
        print(f"Created: {filename}")

# Usage
convert_postgres_export_to_json('my_schema_export.sql', 'new_meta_documents/views/')
```

---

## 🗂️ **Step 3: Replace Mock Data Files**

### **Backup Current Mock Data:**
```bash
# Create backup of existing mock data
mkdir -p backup_mock_data
cp -r meta_documents/* backup_mock_data/
echo "Mock data backed up to backup_mock_data/"
```

### **Add Your Real Data:**
```bash
# Create new directory structure
mkdir -p meta_documents/views
mkdir -p meta_documents/reports

# Copy your converted JSON files
cp your_real_views/*.json meta_documents/views/
cp your_real_reports/*.json meta_documents/reports/
```

### **Validate JSON Files:**
```bash
# Check all JSON files are valid
find meta_documents/ -name "*.json" -exec python -m json.tool {} \; > /dev/null
echo "All JSON files validated"
```

---

## 🔄 **Step 4: Re-index with Real Data**

### **Clear Existing Indices:**
```bash
# Clear MongoDB and OpenSearch data
poetry run python -c "
import asyncio
from scripts.clear_all_data import clear_all_data
asyncio.run(clear_all_data())
"
```

### **Load Real Data:**
```bash
# Load your real data into the system
poetry run python scripts/load_sample_data.py
```

**Expected Output:**
```
🚀 SAMPLE DATA LOADER
==================================================
🗑️  Clearing existing data...
📊 Setting up OpenSearch index...
📄 Loading view metadata files...
   📄 Loaded: V_YOUR_REAL_VIEW_1
   📄 Loaded: V_YOUR_REAL_VIEW_2
   ...
🤖 Generating embeddings and indexing...
   🔍 Embedded and indexed: V_YOUR_REAL_VIEW_1 (1024-dim)
   🔍 Embedded and indexed: V_YOUR_REAL_VIEW_2 (1024-dim)
   ...
✅ Verifying data load...
📊 MongoDB documents: 15
🔍 OpenSearch documents: 15
✅ Test search successful: V_YOUR_REAL_VIEW_1 (similarity: 0.734)

🎉 DATA LOADING COMPLETED SUCCESSFULLY!
```

---

## 🧪 **Step 5: Test with Your Real Data**

### **Test Vector Search:**
```bash
# Test search with your domain-specific terms
poetry run python -c "
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

async def test_real_data_search():
    from text_to_sql_rag.services.embedding_service import EmbeddingService, VectorService
    from opensearchpy import AsyncOpenSearch
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize services
    embedding_service = EmbeddingService(
        endpoint_url=os.getenv('BEDROCK_ENDPOINT_URL'),
        embedding_model=os.getenv('BEDROCK_EMBEDDING_MODEL'),
        use_mock=False
    )
    
    opensearch = AsyncOpenSearch([{'host': 'localhost', 'port': 9200}])
    vector_service = VectorService(opensearch, 'view_metadata', 'embedding')
    
    # Test with your domain terms
    test_queries = [
        'your domain specific term 1',
        'your domain specific term 2', 
        'your domain specific term 3'
    ]
    
    for query in test_queries:
        print(f'\n🔍 Testing: \"{query}\"')
        embedding = await embedding_service.get_embedding(query)
        results = await vector_service.search_similar_views(embedding, k=3)
        
        for view, score in results:
            print(f'   📊 {view.view_name} (similarity: {score:.3f})')
    
    await opensearch.close()

asyncio.run(test_real_data_search())
"
```

### **Test Complete Pipeline:**
```bash
# Run full system test with real data
poetry run python tests/test_rag_pipeline.py
```

### **Test Interactive Chat:**
```bash
# Start interactive session
poetry run python chat_interface.py
```

Try queries specific to your domain:
```
You: [Your domain-specific question here]
```

---

## 💡 **Step 6: Optimize for Your Domain**

### **Improve Context Quality:**

1. **Add Business Context:**
   Update each view JSON with detailed business context:
   ```json
   {
     "business_context": "This view is used for monthly financial reporting. It aggregates transaction data by account and provides balance summaries. Key metrics include total_balance, transaction_count, and last_activity_date. Used by finance team for regulatory compliance and customer account management."
   }
   ```

2. **Add Domain-Specific Sample Queries:**
   ```json
   {
     "sample_queries": [
       "SELECT account_id, total_balance FROM V_ACCOUNT_SUMMARY WHERE balance_date = CURRENT_DATE",
       "SELECT customer_type, AVG(total_balance) FROM V_ACCOUNT_SUMMARY WHERE status = 'ACTIVE' GROUP BY customer_type",
       "SELECT * FROM V_ACCOUNT_SUMMARY WHERE total_balance > 100000 ORDER BY balance_date DESC"
     ]
   }
   ```

3. **Define Relationships:**
   ```json
   {
     "relationships": [
       "Joins with ACCOUNTS table on account_id",
       "Related to V_TRANSACTION_HISTORY through shared account_id",
       "Child view of V_CUSTOMER_PORTFOLIO rollup"
     ]
   }
   ```

### **Test Domain-Specific Queries:**

Create test queries that match your users' typical questions:

```python
# test_domain_queries.py
domain_test_queries = [
    "Show me customer account balances for high-value clients",
    "What are the transaction patterns for active accounts this month", 
    "List all accounts with unusual activity patterns",
    # Add your domain-specific queries here
]
```

---

## 📈 **Step 7: Performance Optimization**

### **Monitor Search Quality:**
```bash
# Test search relevance
poetry run python -c "
# Test script to evaluate search quality
test_cases = [
    {'query': 'your test query', 'expected_views': ['V_EXPECTED_VIEW']},
    # Add more test cases
]

for test in test_cases:
    # Run search and check if expected views appear in results
    pass
"
```

### **Tune Embedding Quality:**

If search results aren't accurate enough:

1. **Improve view descriptions** - Add more detailed, searchable text
2. **Add synonyms and keywords** - Include alternative terms users might search for  
3. **Expand business context** - Explain use cases and related concepts
4. **Add example questions** - Include typical user questions this view answers

---

## ✅ **Step 8: Validation Checklist**

Before going live with your real data:

- [ ] All JSON files are valid and properly formatted
- [ ] Data loads successfully without errors
- [ ] Vector search returns relevant results for your domain queries
- [ ] SQL generation produces valid queries for your database
- [ ] HITL approval workflow functions correctly
- [ ] Interactive chat responds appropriately to your domain questions
- [ ] Performance is acceptable for expected query volume
- [ ] All view descriptions include meaningful business context
- [ ] Sample queries are relevant and executable
- [ ] Relationships between views are documented

---

## 🚨 **Troubleshooting Common Issues**

### **Search Results Not Relevant:**
```bash
# Check if your view descriptions are detailed enough
poetry run python -c "
import json
from pathlib import Path

for json_file in Path('meta_documents/views').glob('*.json'):
    with open(json_file) as f:
        data = json.load(f)
    
    desc_len = len(data.get('description', ''))
    context_len = len(data.get('business_context', ''))
    
    if desc_len < 50 or context_len < 100:
        print(f'⚠️  {json_file.name}: Needs more detailed descriptions')
        print(f'   Description: {desc_len} chars, Context: {context_len} chars')
"
```

### **Embeddings Not Generating:**
```bash
# Test Bedrock connection
curl -X POST https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com/prod \
  -H "Content-Type: application/json" \
  -d '{"model_id": "amazon.titan-embed-text-v2:0", "invoke_type": "embedding", "query": "test"}'
```

### **Vector Search Failing:**
```bash
# Check OpenSearch index
curl "localhost:9200/view_metadata/_mapping?pretty"
curl "localhost:9200/view_metadata/_count?pretty"
```

### **SQL Generation Issues:**
- Ensure your view definitions include proper schema names
- Add more sample queries showing expected SQL patterns
- Include column data types and constraints
- Document any special syntax or functions used

---

## 🎉 **Success!**

Once you've completed these steps, your Text-to-SQL RAG system will be powered by your real data and ready for production use with your specific domain knowledge!

The system will now:
- ✅ Understand your business terminology
- ✅ Generate SQL appropriate for your database schemas  
- ✅ Provide relevant context from your actual views
- ✅ Support domain-specific conversations
- ✅ Maintain data governance through HITL approval

**Next Steps:**
- Deploy to production environment
- Set up monitoring and logging
- Train users on the conversational interface
- Gather feedback and iterate on search quality