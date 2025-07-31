# Column Metadata Restructuring Script - Input Examples

## Overview

The `restructure_column_metadata.py` script transforms your existing `main_schema_metadata.json` into the new hierarchical format. Here are examples of what the script expects and what it creates.

## Required Input Format

### Expected Input: `main_schema_metadata.json`

The script expects your existing schema metadata file to be in this format:

```json
{
  "catalog": "p1-synd",
  "schema": "main_schema",
  "models": [
    {
      "table_name": "TRADE",
      "columns": [
        {
          "name": "trade_id",
          "type": "NUMBER(10)",
          "key": "PRIMARY KEY",
          "example_values": ["1", "2", "3", "4", "5"],
          "nullable": false
        },
        {
          "name": "tranche_id",
          "type": "NUMBER(10)",
          "key": "FOREIGN KEY",
          "example_values": ["101", "102", "103"],
          "nullable": false
        },
        {
          "name": "quantity",
          "type": "NUMBER(15,2)",
          "key": null,
          "example_values": ["1000000.00", "500000.50", "2000000.00"],
          "nullable": false
        },
        {
          "name": "price",
          "type": "NUMBER(10,4)",
          "key": null,
          "example_values": ["99.50", "100.25", "98.75"],
          "nullable": true
        },
        {
          "name": "trade_date",
          "type": "DATE",
          "key": null,
          "example_values": ["2024-01-15", "2024-01-16", "2024-01-17"],
          "nullable": false
        },
        {
          "name": "status_id",
          "type": "NUMBER(2)",
          "key": null,
          "example_values": ["1", "2", "3"],
          "nullable": true
        }
      ]
    },
    {
      "table_name": "USERS",
      "columns": [
        {
          "name": "user_id",
          "type": "NUMBER(10)",
          "key": "PRIMARY KEY",
          "example_values": ["1", "2", "3", "4", "5"],
          "nullable": false
        },
        {
          "name": "email",
          "type": "VARCHAR2(255)",
          "key": "UNIQUE",
          "example_values": ["john@example.com", "jane@company.org", "bob@test.net"],
          "nullable": false
        },
        {
          "name": "first_name",
          "type": "VARCHAR2(100)",
          "key": null,
          "example_values": ["John", "Jane", "Bob", "Alice"],
          "nullable": false
        },
        {
          "name": "created_at",
          "type": "TIMESTAMP",
          "key": null,
          "example_values": ["2023-01-15T10:30:00", "2023-02-20T14:45:00", "2023-03-10T09:15:00"],
          "nullable": false
        },
        {
          "name": "status",
          "type": "VARCHAR2(20)",
          "key": null,
          "example_values": ["active", "inactive", "pending"],
          "nullable": true
        }
      ]
    }
  ],
  "views": [
    {
      "view_name": "V_ACTIVE_TRADES",
      "columns": [
        {
          "name": "trade_id",
          "type": "NUMBER(10)",
          "key": null,
          "example_values": ["1", "2", "3"],
          "nullable": false
        },
        {
          "name": "trader_name",
          "type": "VARCHAR2(200)",
          "key": null,
          "example_values": ["John Smith", "Jane Doe", "Bob Johnson"],
          "nullable": false
        },
        {
          "name": "total_quantity",
          "type": "NUMBER(15,2)",
          "key": null,
          "example_values": ["1000000.00", "2500000.50"],
          "nullable": true
        }
      ]
    }
  ]
}
```

## Generated Output Examples

### 1. Business Descriptions (by Domain)

**File:** `meta_documents/p1-synd/descriptions/trading_lifecycle.json`
```json
{
  "domain": "trading_lifecycle",
  "description": "Core trading and order management tables",
  "tables": {
    "TRADE": "Table containing trade data",
    "V_ACTIVE_TRADES": "View providing active trades information"
  }
}
```

**File:** `meta_documents/p1-synd/descriptions/user_management.json`
```json
{
  "domain": "user_management",
  "description": "User accounts and authentication",
  "tables": {
    "USERS": "Table containing users data"
  }
}
```

### 2. Column Details (per Table)

**File:** `meta_documents/p1-synd/columns/trade.json`
```json
{
  "table": "TRADE",
  "columns": {
    "trade_id": {
      "type": "NUMBER(10)",
      "nullable": false,
      "description": "Unique identifier for trade",
      "constraint": "PRIMARY KEY",
      "example_values": ["1", "2", "3", "4", "5"]
    },
    "tranche_id": {
      "type": "NUMBER(10)",
      "nullable": false,
      "description": "Unique identifier for tranche",
      "constraint": "FOREIGN KEY",
      "example_values": ["101", "102", "103"],
      "join_hint": "Likely references another table's ID field"
    },
    "quantity": {
      "type": "NUMBER(15,2)",
      "nullable": false,
      "description": "Quantity field",
      "example_values": ["1000000.00", "500000.50", "2000000.00"]
    },
    "price": {
      "type": "NUMBER(10,4)",
      "nullable": true,
      "description": "Price field",
      "example_values": ["99.50", "100.25", "98.75"]
    },
    "trade_date": {
      "type": "DATE",
      "nullable": false,
      "description": "Date/time when trade occurred",
      "example_values": ["2024-01-15", "2024-01-16", "2024-01-17"]
    },
    "status_id": {
      "type": "NUMBER(2)",
      "nullable": true,
      "description": "Unique identifier for status",
      "example_values": ["1", "2", "3"],
      "join_hint": "Likely references another table's ID field"
    }
  }
}
```

**File:** `meta_documents/p1-synd/columns/users.json`
```json
{
  "table": "USERS",
  "columns": {
    "user_id": {
      "type": "NUMBER(10)",
      "nullable": false,
      "description": "Unique identifier",
      "constraint": "PRIMARY KEY",
      "example_values": ["1", "2", "3", "4", "5"]
    },
    "email": {
      "type": "VARCHAR2(255)",
      "nullable": false,
      "description": "Email field",
      "constraint": "UNIQUE",
      "example_values": ["john@example.com", "jane@company.org", "bob@test.net"]
    },
    "first_name": {
      "type": "VARCHAR2(100)",
      "nullable": false,
      "description": "Name of the first",
      "example_values": ["John", "Jane", "Bob", "Alice"]
    },
    "created_at": {
      "type": "TIMESTAMP",
      "nullable": false,
      "description": "Timestamp when record was created",
      "example_values": ["2023-01-15T10:30:00", "2023-02-20T14:45:00", "2023-03-10T09:15:00"]
    },
    "status": {
      "type": "VARCHAR2(20)",
      "nullable": true,
      "description": "Status field",
      "example_values": ["active", "inactive", "pending"]
    }
  }
}
```

### 3. Business Rules Template

**File:** `meta_documents/p1-synd/business_rules/date_and_status_rules.json`
```json
{
  "area": "date_handling",
  "description": "Rules for handling date and timestamp columns",
  "rules": [
    {
      "pattern": "T00:00:00 timestamps",
      "columns": ["*_date", "*_time"],
      "rule": "Timestamps ending in T00:00:00 represent date-only values stored as timestamps",
      "sql_guidance": "Use TRUNC() function for date comparisons",
      "example": "WHERE TRUNC(trade_date) = DATE '2023-01-15'"
    },
    {
      "pattern": "status_id columns",
      "columns": ["*_status_id", "status_id"],
      "rule": "Status ID columns reference lookup tables for human-readable values",
      "sql_guidance": "Join with appropriate lookup table or use CASE statements",
      "example": "JOIN tranche_status_lookups tsl ON t.status_id = tsl.id"
    }
  ]
}
```

## Table Categorization Logic

The script uses basic heuristics to categorize tables into domains:

### **Trading Lifecycle Domain**
Tables/views containing: `trade`, `order`, `syndicate`, `hedge`
- TRADE → trading_lifecycle
- V_ACTIVE_TRADES → trading_lifecycle
- SYNDICATE_TRADES → trading_lifecycle

### **User Management Domain**  
Tables/views containing: `user`, `auth`, `login`, `metrics`
- USERS → user_management
- V_USER_METRICS → user_management

### **Syndicate Operations Domain**
Tables/views containing: `tranche`, `syndicate`, `mars`, `termsheet`
- TRANCHE_RATINGS → syndicate_operations
- V_TERMSHEET → syndicate_operations

### **Reporting Domain**
All other tables/views default to this domain

## How to Prepare Your Data

### 1. **Ensure Correct File Location**
Place your existing schema file at:
```
meta_documents/p1-synd/schema/main_schema_metadata.json
```

### 2. **Verify JSON Structure**
Your file should have:
- `catalog` field (string)
- `schema` field (string) 
- `models` array with table objects
- `views` array with view objects (optional)

### 3. **Column Object Requirements**
Each column should have:
- `name` (required)
- `type` (required)
- `nullable` (boolean, optional)
- `key` (string, optional - PRIMARY KEY, FOREIGN KEY, UNIQUE, etc.)
- `example_values` (array, optional but recommended)

### 4. **Run the Script**
```bash
python scripts/restructure_column_metadata.py
```

## Customizing the Output

### **Custom Domain Categorization**
You can modify the categorization logic in the script:

```python
# In create_business_descriptions method
if any(keyword in table_name.lower() for keyword in ["trade", "order", "position"]):
    domain = "trading_lifecycle"
elif any(keyword in table_name.lower() for keyword in ["client", "customer", "investor"]):
    domain = "client_management"  # Custom domain
# ... add your own logic
```

### **Custom Business Rules**
Modify the `create_business_rules_template` method to add your specific rules:

```python
{
    "pattern": "currency columns",
    "columns": ["*_currency", "currency_*"],
    "rule": "Currency columns should use ISO 4217 codes",
    "sql_guidance": "Always validate currency codes against standard list",
    "example": "WHERE currency IN ('USD', 'EUR', 'GBP')"
}
```

The script provides a solid foundation and can be customized for your specific metadata patterns and business rules.