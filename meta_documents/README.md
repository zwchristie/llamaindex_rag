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

## Example Structure
```
meta_documents/
├── my_application/
│   ├── schema/
│   │   ├── my_schema_metadata.json
│   │   └── another_schema_metadata.json
│   └── reports/
│       ├── sales_report.txt
│       ├── user_analytics.txt
│       └── financial_summary.txt
└── another_application/
    ├── schema/
    │   └── main_schema_metadata.json
    └── reports/
        └── monthly_report.txt
```

## File Naming Convention
- Schema files: `{schema_name}_metadata.json`
- Report files: `{report_name}.txt`

## Synchronization
Documents in this folder are automatically synchronized with MongoDB and the vector store on application startup.