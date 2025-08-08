# Metadata Management Documentation

This document explains how to manage metadata in the MongoDB-based metadata system. All metadata is now dynamically loaded from MongoDB - there are no hardcoded fallbacks.

## Overview

The system uses MongoDB as the single source of truth for:
- **View Domain Mappings** - Which business domains each view covers
- **View Dependencies** - Relationships between views
- **Query Patterns** - Keywords and patterns for view selection
- **Business Domain Definitions** - Domain hierarchy and relationships
- **Domain Terminology** - Keywords and concepts for each domain
- **Detection Rules** - Rules for identifying domains from queries
- **Classification Rules** - Rules for classifying views as core/supporting

## Quick Start

### 1. Initial Setup
```bash
# Run the discovery script to populate MongoDB from your data files
python scripts/discover_and_migrate_metadata.py --reports-dir meta_documents/reports --views-file meta_documents/view_metadata.json

# This will:
# - Clean MongoDB collections
# - Discover views from view_metadata.json
# - Discover reports from reports/ folder
# - Generate domain mappings, dependencies, and patterns
# - Create business domain hierarchy
# - Set up detection and classification rules
```

### 2. Adding New Views
```bash
# 1. Add your view to view_metadata.json or create a new report JSON file in reports/
# 2. Re-run the discovery script
python scripts/discover_and_migrate_metadata.py

# No code changes needed!
```

### 3. Viewing What Was Discovered
```bash
# Dry run to see what would be migrated without changing MongoDB
python scripts/discover_and_migrate_metadata.py --dry-run
```

## Data Files Structure

### Views File: `meta_documents/view_metadata.json`
```json
{
  "CORE_VIEWS": [
    {
      "view_name": "V_YOUR_VIEW",
      "view_type": "CORE",
      "domains": ["DEAL", "TRANCHE"],
      "entities": ["deals", "tranches"],
      "patterns": ["deal", "tranche", "pricing"],
      "description": "Description of what this view contains",
      "data_returned": "COLUMN1: Description...",
      "use_cases": "Use case descriptions...",
      "example_query": "SELECT * FROM V_YOUR_VIEW"
    }
  ],
  "SUPPORTING_VIEWS": [
    // Similar structure with "view_type": "SUPPORTING"
  ]
}
```

### Reports Files: `meta_documents/reports/*.json`
```json
{
  "domains": ["DEAL", "TRANCHE", "SYNDICATE"],
  "view_name": "V_REPORT_VIEW",
  "report_name": "Your Report Name",
  "report_description": "What this report shows",
  "data_returned": "Column descriptions...",
  "example_sql": "SELECT * FROM V_REPORT_VIEW",
  "use_cases": "How this report is used..."
}
```

## Business Domains

The system recognizes these business domains:
- **ISSUER** - Companies seeking capital through bond issuances
- **DEAL** - Fundraising initiatives created by JPMorgan for issuers
- **TRANCHE** - Individual bond issuances with distinct terms
- **SYNDICATE** - Financial institutions participating in distribution
- **ORDER** - Investment requests from institutional investors
- **INVESTOR** - Primary market investor entities
- **ORDER_BASIS** - Hedge order amounts within orders
- **ORDER_LIMIT** - Bond order amounts within orders
- **TRADE** - Records of final trades that get booked

## MongoDB Collections

### Core Collections

#### `view_domain_mappings`
Maps views to business domains with priority scores:
```json
{
  "view_name": "V_DEAL_SUMMARY",
  "business_domains": ["DEAL", "ISSUER"],
  "view_type": "core",
  "priority_score": 8,
  "description": "High-level deal information",
  "query_patterns": ["deal", "summary", "overview"]
}
```

#### `view_dependencies`
Defines view relationships:
```json
{
  "primary_view": "V_DEAL_SUMMARY",
  "supporting_views": ["V_DEAL_DETAILS", "V_TRANCHE_SUMMARY"],
  "dependency_type": "enhancement"
}
```

#### `business_domains`
Business domain definitions:
```json
{
  "domain_name": "DEAL",
  "summary": "Fundraising initiatives...",
  "parent_domains": ["ISSUER"],
  "child_domains": ["TRANCHE"],
  "key_concepts": ["deal", "fundraising", "initiative"]
}
```

## Advanced Management

### Manual Metadata Updates

#### Adding a New View Mapping
```python
from src.text_to_sql_rag.services.view_metadata_service import ViewMetadataService
from src.text_to_sql_rag.models.view_metadata_models import ViewDomainMapping

service = ViewMetadataService()
mapping = ViewDomainMapping(
    view_name="V_NEW_VIEW",
    business_domains=["DEAL", "TRANCHE"],
    view_type="core",
    priority_score=7,
    description="Your view description",
    query_patterns=["keyword1", "keyword2"]
)
service.add_view_mapping(mapping)
```

#### Adding Business Domain Terminology
```python
from src.text_to_sql_rag.services.business_domain_metadata_service import BusinessDomainMetadataService
from src.text_to_sql_rag.models.business_domain_models import DomainTerminology

service = BusinessDomainMetadataService()
terminology = DomainTerminology(
    domain_name="DEAL",
    term_type="primary",
    terms=["deal", "transaction", "issuance"],
    weight=1.0
)
service.add_domain_terminology(terminology)
```

### View Priority System

Views are prioritized 1-10 scale:
- **10**: Highest priority (critical views like V_TERMSHEET for deal/tranche queries)
- **8-9**: High priority (core business views)
- **6-7**: Medium priority (standard core views)
- **4-5**: Supporting views
- **1-3**: Utility/maintenance views

Priority affects view selection ranking in queries.

### Query Pattern Matching

The system uses multiple matching strategies:
1. **Keyword matching** - Direct term matching
2. **Pattern matching** - Regex patterns (future enhancement)
3. **Context matching** - Business context phrases
4. **Domain matching** - Domain-specific terminology

Weights:
- Context matching: 3.0x weight (highest)
- Pattern matching: 1.5x weight
- Keyword matching: 1.0x weight

### Domain Detection Rules

Rules for detecting business domains from queries are prioritized:
- **TRANCHE detection**: Priority 10 (highest for deal/tranche queries)
- **DEAL detection**: Priority 9
- **ISSUER detection**: Priority 8
- **ORDER detection**: Priority 7
- **TRADE detection**: Priority 6

## Troubleshooting

### View Not Being Selected
1. Check if view exists in MongoDB: `db.view_domain_mappings.find({view_name: "V_YOUR_VIEW"})`
2. Verify domain mapping matches query domains
3. Check priority score - higher priority views are selected first
4. Look for query patterns that match your query terms

### Domain Not Being Detected
1. Check domain detection rules: `db.domain_detection_rules.find({})`
2. Verify terminology mappings: `db.domain_terminology.find({domain_name: "YOUR_DOMAIN"})`
3. Add more keywords to domain terminology if needed

### No Views Returned
1. Verify MongoDB connection and data exists
2. Check logs for "No mappings found in MongoDB" errors
3. Re-run discovery script to populate data
4. Check domain detection - if no domains detected, no views will be selected

### Performance Issues
1. MongoDB indexes are created automatically by the discovery script
2. Caching is enabled with 30-minute TTL
3. Force cache refresh: Call `service.refresh_cache()` on any metadata service

## Monitoring

### Checking Metadata Status
```python
from src.text_to_sql_rag.services.view_metadata_service import ViewMetadataService

service = ViewMetadataService()
mappings = service.get_view_domain_mappings()
print(f"Views in system: {len(mappings)}")

dependencies = service.get_view_dependencies()
print(f"Dependencies: {len(dependencies)}")

# View usage statistics
stats = service.get_usage_stats()
for view_name, stat in stats.items():
    print(f"{view_name}: {stat.usage_count} uses, {stat.success_count} successes")
```

### Log Messages to Monitor
- `"Loaded view domain mappings from MongoDB"` - System loading successfully
- `"No mappings found in MongoDB"` - Data missing, run discovery script
- `"MongoDB connection required"` - Connection issue
- `"Retrieved business domains from MongoDB"` - Domain system working

## System Architecture

```
Application Startup → MongoDB → Vector Store → Hierarchical Query Processing
         ↓              ↓            ↓                      ↓
    Load Metadata   Document     Embedding     Query → Domain Detection → View Selection → SQL
         ↓          Storage      Process              ↓                    ↓              ↓
    Domain Rules  Versioning   Content Hash    Domain Context    View Context    SQL Context
         ↓              ↓            ↓                      ↓                    ↓              ↓
                  MongoDB Collections (Single Source of Truth)        Vector Store (Search)
```

### Key Changes in Architecture

**Before (File-Based):**
- Application reads local JSON/text files on startup
- Files synced to MongoDB and Vector Store
- Hierarchical processing uses vector store

**After (MongoDB-First):**
- MongoDB is populated using `discover_and_migrate_metadata.py`
- Application reads directly from MongoDB on startup
- Documents embedded to Vector Store from MongoDB
- Hierarchical processing maintained with MongoDB metadata

## Best Practices

1. **Always use discovery script** for bulk changes - it's faster and more reliable
2. **Test with --dry-run** before making changes to see what will be affected
3. **Monitor usage statistics** to understand which views are actually being used
4. **Keep view descriptions current** - they're used in LLM selection prompts
5. **Use meaningful priority scores** - they directly affect view selection
6. **Add domain-specific keywords** to improve query matching
7. **Regular backups** of MongoDB metadata collections

## Migration from Hardcoded System

All hardcoded fallbacks have been removed. The system will fail gracefully if MongoDB is unavailable rather than using stale hardcoded data. This ensures consistency and forces proper metadata management.

### Removed Components
- ❌ File-based document sync service
- ❌ Local file dependency on startup  
- ❌ DocumentSyncService initialization
- ❌ Meta documents directory scanning
- ❌ File-to-MongoDB-to-Vector sync chain

### New Components  
- ✅ Direct MongoDB-to-Vector Store synchronization
- ✅ MongoDB-first application startup
- ✅ Simplified startup process (3 services instead of 4)
- ✅ Content hash-based change detection
- ✅ Streamlined embedding workflow
- ✅ Better error handling for missing services

## Testing and Validation

Two testing scripts are available to validate the refactored system:

### 1. MongoDB Document Sync Test
```bash
python scripts/test_mongodb_document_sync.py
```

This script validates:
- MongoDB connection and document retrieval
- Vector Store connection and embedding
- Document synchronization process
- Content hash-based change detection
- Full startup simulation

### 2. Hierarchical Access Pattern Test  
```bash
python scripts/test_metadata_hierarchy.py
```

This script validates:
- ViewMetadataService MongoDB integration
- BusinessDomainMetadataService functionality
- Hierarchical query processing flow
- Metadata consistency across services
- Performance benchmarks

## Migration Steps

To migrate from the old file-based system to the new MongoDB-first system:

1. **Populate MongoDB** (if not already done):
   ```bash
   python scripts/discover_and_migrate_metadata.py
   ```

2. **Test the new system**:
   ```bash
   python scripts/test_mongodb_document_sync.py
   python scripts/test_metadata_hierarchy.py
   ```

3. **Start the application** - it will now:
   - Load metadata directly from MongoDB
   - Embed documents from MongoDB to Vector Store
   - Maintain hierarchical processing patterns
   - Skip local file scanning

4. **Clean up old files** (optional):
   - Old sync service dependencies are removed
   - Unused processing scripts are deleted
   - Local files are no longer required for operation