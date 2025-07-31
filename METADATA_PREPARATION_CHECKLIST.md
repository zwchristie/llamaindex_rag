# Metadata Preparation Checklist

## Pre-Script Execution Checklist

### ✅ **File Preparation**

- [ ] **Create `fi_table_details_demo.json`** with table descriptions
  ```json
  {
    "TRADE": {
      "description": "Core trading transactions with pricing and quantities"
    },
    "USERS": {
      "description": "User accounts and authentication information"
    },
    "LATE_ORDERS": {
      "description": "Orders received after market close requiring approval"
    }
  }
  ```

- [ ] **Verify `main_schema_metadata.json`** exists and has correct structure
  - Located at: `meta_documents/p1-synd/schema/main_schema_metadata.json`
  - Contains `catalog`, `schema`, `models`, and optionally `views`
  - Each model has `table_name` and `columns` array
  - Each column has at minimum `name` and `type`

### ✅ **Directory Structure**

Ensure these directories exist (scripts will create them if missing):
```
meta_documents/p1-synd/
├── schema/
│   ├── ddl/                    # Created by DDL script
│   └── main_schema_metadata.json  # Your existing file
├── descriptions/               # Created by column script
├── business_rules/            # Created by column script  
├── columns/                   # Created by column script
├── lookups/                   # Already exists
└── reports/                   # Already exists
```

### ✅ **Database Connection (for DDL script)**

- [ ] **Update connection string** in `scripts/extract_ddl_statements.py`:
  ```python
  DATABASE_URL = "oracle://username:password@host:port/service_name"
  ```

- [ ] **Verify database permissions** - user needs SELECT access to:
  - `user_tab_columns` or `all_tab_columns`
  - `user_constraints` or `all_constraints`  
  - `user_views` or `all_views`
  - Optionally: `DBMS_METADATA` package

### ✅ **Python Dependencies**

- [ ] **Install SQLAlchemy** (for DDL script):
  ```bash
  pip install sqlalchemy cx_Oracle
  ```

## Execution Steps

### Step 1: Extract DDL Statements
```bash
python scripts/extract_ddl_statements.py
```

**Expected Output:**
- Individual `.sql` files in `meta_documents/p1-synd/schema/ddl/`
- One file per table/view with DDL and description comment

### Step 2: Restructure Column Metadata  
```bash
python scripts/restructure_column_metadata.py
```

**Expected Output:**
- Business description files in `meta_documents/p1-synd/descriptions/`
- Column detail files in `meta_documents/p1-synd/columns/`
- Business rules template in `meta_documents/p1-synd/business_rules/`

### Step 3: Sync New Documents
```bash
# Run your document sync to index the new metadata
python -m src.text_to_sql_rag.services.document_sync_service
```

## Quick Start with Minimal Data

If you want to test the system with minimal data, you can use these minimal files:

### **Minimal `fi_table_details_demo.json`:**
```json
{
  "USERS": {
    "description": "System users and authentication"
  },
  "TRADE": {
    "description": "Trading transactions and orders"
  }
}
```

### **Minimal `main_schema_metadata.json`:**
```json
{
  "catalog": "p1-synd",
  "schema": "main_schema", 
  "models": [
    {
      "table_name": "USERS",
      "columns": [
        {
          "name": "user_id",
          "type": "NUMBER(10)",
          "key": "PRIMARY KEY",
          "nullable": false
        },
        {
          "name": "email", 
          "type": "VARCHAR2(255)",
          "nullable": false
        }
      ]
    }
  ],
  "views": []
}
```

## Troubleshooting Common Issues

### **Issue: "File not found" errors**
- Verify file paths are correct
- Check that `meta_documents/p1-synd/` directory exists
- Ensure JSON files have proper `.json` extension

### **Issue: "JSON decode error"**
- Validate JSON syntax using online JSON validator
- Check for trailing commas or missing quotes
- Ensure UTF-8 encoding without BOM

### **Issue: "No tables found" in DDL script**
- Verify database connection string
- Check database permissions
- Ensure table names in `fi_table_details_demo.json` match database exactly (case-sensitive)

### **Issue: "Empty output files"**
- Check that input `main_schema_metadata.json` has data in `models` array
- Verify column objects have required `name` and `type` fields
- Look at script console output for error messages

## Validation

After running both scripts, you should have:

```
meta_documents/p1-synd/
├── schema/ddl/
│   ├── users.sql              # ✅ DDL with description
│   └── trade.sql              # ✅ DDL with description
├── descriptions/
│   ├── user_management.json   # ✅ Business descriptions
│   └── trading_lifecycle.json # ✅ Business descriptions  
├── business_rules/
│   └── date_and_status_rules.json # ✅ Rules template
└── columns/
    ├── users.json             # ✅ Column details
    └── trade.json             # ✅ Column details
```

Each file should contain valid JSON (for .json files) or valid SQL (for .sql files) with meaningful content.

## Ready for Hierarchical System

Once both scripts complete successfully:
1. Clean up old MongoDB/OpenSearch documents
2. Run document sync to index new hierarchical metadata
3. Test queries with the new HierarchicalContextService
4. Monitor performance improvements (40s → 5-10s expected)