# Meta Documents Directory

This directory contains metadata documents for the text-to-SQL RAG system.

## Structure

The documents are organized by catalog (application name) and contain:

### Schema Documents (JSON)
- **catalog**: Name of the application the metadata is for
- **schema**: Name of the schema the structures belong to
- **models**: JSON objects of all tables with:
  - Table column names
  - Column data types
  - Keys that belong to columns
  - Column example values
  - Nullable status
- **views**: View information with:
  - View name
  - Query that generates the view
  - Columns and their information
- **relationships**: Relationship information with:
  - Relationship name
  - Tables it connects
  - Example SQL showing column connections
  - Relationship type (one-to-one, one-to-many, many-to-one, many-to-many)

### Report Documents (Text)
- Text files with descriptions of what data the report returns
- SQL for getting the report data

### Lookup Metadata Documents (JSON)
- JSON documents containing lookup data for categorical fields
- ID-to-name mappings for database foreign keys
- Valid values for status, type, category fields
- Helps prevent SQL errors from incorrect literal values

## Example Structure
```
meta_documents/
├── p1-synd/
│   ├── schema/
│   │   └── main_schema_metadata.json
│   ├── reports/
│   │   ├── sales_summary_report.txt
│   │   └── user_analytics_report.txt
│   └── lookups/
│       ├── tranche_statuses.json
│       ├── deal_types.json
│       └── asset_categories.json
└── another_application/
    ├── schema/
    │   └── main_schema_metadata.json
    ├── reports/
    │   └── monthly_report.txt
    └── lookups/
        └── status_codes.json
```

## File Naming Convention
- Schema files: `{schema_name}_metadata.json`
- Report files: `{report_name}.txt`
- Lookup files: `{lookup_category}.json`

## Synchronization
Documents in this folder are automatically synchronized with MongoDB and the vector store on application startup.