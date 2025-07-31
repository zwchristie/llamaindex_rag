# DDL Extraction Script Fixes

## Issues Identified & Fixed

### 1. **Table DDL Extraction Failures**
**Problem**: The script was failing to extract DDL for normal tables but working for views.

**Root Causes**:
- SQLAlchemy reflection failing for Oracle tables with specific constraints/features
- DBMS_METADATA.GET_DDL() syntax issues
- Insufficient fallback methods for system table queries
- Missing proper handling of Oracle CLOB results

### 2. **Fixes Applied**

#### **Enhanced Table DDL Extraction** (`get_table_ddl`)
1. **Three-tier fallback approach**:
   - **Tier 1**: SQLAlchemy reflection (fastest when it works)
   - **Tier 2**: Oracle DBMS_METADATA.GET_DDL() (most accurate)
   - **Tier 3**: Manual DDL generation from system tables (most reliable)

2. **Improved DBMS_METADATA queries**:
   - Fixed syntax for schema-specific queries
   - Proper CLOB handling for large DDL results
   - Better error handling and cleanup

3. **Robust system table fallbacks**:
   - Try `user_tab_columns` first (current user's tables)
   - Fallback to `all_tab_columns` (all accessible tables)
   - Same approach for constraints with `user_constraints` → `all_constraints`

#### **Enhanced View DDL Extraction** (`get_view_ddl`)
1. **Multi-method approach**:
   - **Primary**: DBMS_METADATA.GET_DDL() for complete view definition
   - **Secondary**: `all_views.text` for view SQL
   - **Fallback**: SQLAlchemy reflection for basic structure

2. **Better CLOB handling** for view definitions

#### **General Improvements**
1. **SQLAlchemy 2.x compatibility**: Added `text()` wrapper for all raw SQL queries
2. **Better error handling**: More specific error messages and graceful degradation
3. **CLOB result handling**: Proper reading of Oracle CLOB data types
4. **Improved logging**: Debug, warning, and info levels for better troubleshooting

## Updated Script Features

### **Multi-Tier Extraction Strategy**
```python
# For Tables:
1. Try SQLAlchemy reflection
2. Try DBMS_METADATA.GET_DDL('TABLE', ...)
3. Build DDL from user_tab_columns/all_tab_columns

# For Views:
1. Try DBMS_METADATA.GET_DDL('VIEW', ...)
2. Try all_views.text
3. Try SQLAlchemy reflection
```

### **Improved Error Handling**
- Specific error messages for each extraction method
- Graceful fallbacks between methods
- Debug logging for troubleshooting permission issues

### **Oracle-Specific Enhancements**
- Proper CLOB handling for large DDL statements
- Schema-aware queries with fallbacks
- Oracle data type handling (NUMBER with precision/scale, VARCHAR2, etc.)
- Primary key constraint extraction

## How to Use the Fixed Script

1. **Update your database connection** in the script:
   ```python
   DATABASE_URL = "oracle://username:password@host:port/service_name"
   ```

2. **Place your `fi_table_details_demo.json`** in the correct location:
   ```
   meta_documents/p1-synd/fi_table_details_demo.json
   ```

3. **Run the script**:
   ```bash
   python scripts/extract_ddl_statements.py
   ```

4. **Check the output** in:
   ```
   meta_documents/p1-synd/schema/ddl/
   ├── table1.sql
   ├── table2.sql
   └── view1.sql
   ```

## Expected Results

The script should now successfully:
- ✅ Extract DDL for both tables and views
- ✅ Handle Oracle-specific data types and constraints
- ✅ Work with different Oracle permission levels
- ✅ Provide detailed error messages for troubleshooting
- ✅ Generate clean, readable DDL statements
- ✅ Include table descriptions as comments

## Troubleshooting

If you still encounter issues:

1. **Check permissions**: Ensure your Oracle user has SELECT access to:
   - `user_tab_columns` / `all_tab_columns`
   - `user_constraints` / `all_constraints`
   - `user_views` / `all_views`

2. **Check DBMS_METADATA permissions**: Some Oracle installations restrict DBMS_METADATA access

3. **Review the logs**: The script now provides detailed logging for each extraction attempt

4. **Test individual tables**: You can modify the script to test specific problematic tables

The script is now much more robust and should handle the Oracle table extraction issues you encountered.