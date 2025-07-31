# Hierarchical Metadata Migration Instructions

This document provides step-by-step instructions for migrating from the old monolithic metadata system to the new hierarchical tiered architecture.

## Overview

The new system uses 5 document types instead of 1 monolithic schema file:
- **DDL**: Core table/view structures (.sql files)
- **BUSINESS_DESC**: Table descriptions by domain (.json files)  
- **BUSINESS_RULES**: Edge cases and special rules (.json files)
- **COLUMN_DETAILS**: Detailed column metadata (.json files)
- **LOOKUP_METADATA**: ID-name lookup mappings (.json files) - already implemented

## Performance Improvements Expected

- **Token Usage**: 80-90% reduction (from 20K+ to 2-5K tokens typical)
- **Response Time**: From 40s to 5-10s for most queries
- **Accuracy**: Improved due to targeted context and less noise

## Migration Steps

### Step 1: Prepare Your Data Files

1. **Place `fi_table_details_demo.json`** in `meta_documents/p1-synd/` directory
   - This file should contain table names as keys with description objects as values
   - Format: `{"TABLE_NAME": {"description": "Table description text"}}`

2. **Update Database Connection** in the DDL extraction script
   - Edit `scripts/extract_ddl_statements.py`
   - Update the `DATABASE_URL` variable with your Oracle connection string

### Step 2: Run Housekeeping Scripts

#### Extract DDL Statements
```bash
cd C:\Users\zack\Documents\llamaindex_proj
python scripts/extract_ddl_statements.py
```

This script will:
- Connect to your Oracle database using SQLAlchemy
- Extract DDL for all tables/views listed in `fi_table_details_demo.json`
- Create individual `.sql` files in `meta_documents/p1-synd/schema/ddl/`
- Add table descriptions as comments in the DDL files

#### Restructure Column Metadata
```bash
python scripts/restructure_column_metadata.py
```

This script will:
- Parse your existing `main_schema_metadata.json`
- Create business description files by domain in `meta_documents/p1-synd/descriptions/`
- Create individual column detail files in `meta_documents/p1-synd/columns/`
- Generate business rules templates in `meta_documents/p1-synd/business_rules/`

### Step 3: Clean Up MongoDB and OpenSearch

**You mentioned you'll handle this manually**, but for reference:
- Remove all documents with `document_type = "schema"` from MongoDB
- Clear the OpenSearch index or remove schema-type documents
- The new sync will repopulate with tiered documents

### Step 4: Sync New Metadata

```bash
# Run your document sync service to index the new tiered metadata
python -m src.text_to_sql_rag.services.document_sync_service
```

This will index:
- DDL files as `document_type = "ddl"`
- Business descriptions as `document_type = "business_desc"`
- Business rules as `document_type = "business_rules"`  
- Column details as `document_type = "column_details"`
- Existing lookups as `document_type = "lookup_metadata"`

### Step 5: Test the New System

```python
# Test the hierarchical context service directly
from src.text_to_sql_rag.services.hierarchical_context_service import HierarchicalContextService
from src.text_to_sql_rag.services.vector_service import LlamaIndexVectorService
from src.text_to_sql_rag.services.llm_service import LLMService

# Initialize services
vector_service = LlamaIndexVectorService(...)
llm_service = LLMService()
hierarchical_service = HierarchicalContextService(vector_service, llm_service)

# Test context building
context = hierarchical_service.build_context("Show me all active trades from last month")
print(f"Selected tables: {context.selected_tables}")
print(f"Total tokens: {context.total_tokens}")
print(f"Tiers: {[tier.name for tier in context.tiers]}")
```

## New Directory Structure

```
meta_documents/p1-synd/
├── schema/
│   ├── ddl/                    # NEW: Individual DDL files
│   │   ├── trade.sql
│   │   ├── users.sql
│   │   └── ...
│   └── main_schema_metadata.json  # LEGACY: Keep for fallback
├── descriptions/               # NEW: Business descriptions by domain
│   ├── trading_lifecycle.json
│   ├── user_management.json
│   └── ...
├── business_rules/            # NEW: Special handling rules
│   ├── date_and_status_rules.json
│   └── ...
├── columns/                   # NEW: Detailed column metadata
│   ├── trade.json
│   ├── users.json
│   └── ...
├── lookups/                   # EXISTING: Lookup metadata
│   ├── tranche_status_lookups.json
│   └── ...
└── reports/                   # EXISTING: Report metadata
    ├── sales_summary_report.txt
    └── ...
```

## Troubleshooting

### DDL Extraction Issues
- **Connection Errors**: Verify Oracle connection string and credentials
- **Permission Errors**: Ensure database user has SELECT permissions on system tables
- **Missing Tables**: Check if tables exist and are spelled correctly in `fi_table_details_demo.json`

### Sync Issues
- **Document Type Detection**: The sync service uses file paths to determine document types
- **Encoding Issues**: Ensure all files are saved as UTF-8
- **JSON Validation**: Verify JSON files are valid before syncing

### Performance Issues
- **Token Limits**: If context still exceeds limits, reduce `max_context_tokens` in hierarchical service
- **Search Quality**: Adjust `similarity_top_k` values in hierarchical context service
- **LLM Timeouts**: Increase timeout settings if table selection takes too long

## Rollback Plan

If you need to rollback:
1. Keep the existing `main_schema_metadata.json` file
2. Change the workflow routing to use `"get_metadata"` instead of `"assess_confidence"`
3. Re-enable the legacy metadata methods by removing `_LEGACY` suffixes

The hierarchical system is designed to coexist with the legacy system during transition.