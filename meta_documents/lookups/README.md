# Lookup Metadata Documents

This folder contains JSON documents that define lookup data used in the Oracle database. These documents help the SQL generation system understand valid values and their corresponding IDs for categorical fields.

## Document Format

Each lookup document should follow this JSON structure:

```json
{
  "name": "lookup_category_name",
  "description": "Description of what this lookup represents",
  "values": [
    {
      "id": 1,
      "name": "display_name",
      "code": "OPTIONAL_CODE",
      "description": "Optional description of this value"
    }
  ]
}
```

## Alternative Formats Supported

### Multiple Lookups in One File
```json
{
  "lookups": [
    {
      "name": "status_types",
      "description": "...",
      "values": [...]
    },
    {
      "name": "category_types", 
      "description": "...",
      "values": [...]
    }
  ]
}
```

### Nested Structure
```json
{
  "tranche_data": {
    "description": "Tranche-related lookups",
    "values": [...]
  }
}
```

## Usage

When users ask questions like:
- "Show me tranches with status announced"
- "Find deals where type equals corporate"
- "Get all items where category is fixed_income"

The system will:
1. Search for relevant lookup documents
2. Find the correct ID mapping (e.g., "announced" → ID 1)
3. Generate SQL using the ID: `WHERE tranche_status_id = 1`
4. Avoid common errors from using string literals incorrectly

## Examples

- `tranche_statuses.json` - Status values for tranches
- `deal_types.json` - Types of financial deals
- `asset_categories.json` - Asset classification categories
- `filing_statuses.json` - FINRA filing status values