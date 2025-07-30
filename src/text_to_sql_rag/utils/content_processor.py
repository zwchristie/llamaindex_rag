"""Content processing utilities for document management."""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from ..models.simple_models import DocumentType


class ContentProcessor:
    """Utility class for processing document content."""
    
    def __init__(self):
        self.sql_pattern = re.compile(
            r'(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s+.*?;',
            re.IGNORECASE | re.DOTALL
        )
        
        # Oracle SQL Rules for text-to-SQL generation
        self.oracle_sql_rules = """
=== ORACLE SQL GENERATION RULES ===

DIALECT: Oracle SQL (19c)

SECURITY RULES:
- ONLY USE SELECT statements - NO DELETE, UPDATE, INSERT, or DDL statements
- ONLY USE tables and columns mentioned in the database schema
- ALWAYS USE schema qualification (e.g., SYND.DEALS, SYND.USERS)

COLUMN SELECTION RULES:
- ONLY USE "*" if user explicitly asks for all columns
- ONLY CHOOSE columns that belong to tables in the database schema
- ALWAYS QUALIFY column names with table name or alias (e.g., u.username, deals.deal_id)

JOIN RULES:
- YOU MUST USE "JOIN" when selecting columns from multiple tables
- Use proper table aliases to avoid ambiguity
- DON'T USE '.' in table aliases - replace with spaces or underscores

STRING COMPARISON RULES:
- USE case-insensitive comparisons: lower(table.column) = lower('value')
- For pattern matching: lower(table.column) LIKE lower('%pattern%')
- For exact matches: lower(table.column) = lower('exact_value')

DATE/TIME RULES:
- USE SYSDATE for current date/time
- USE efficient date ranges: date_column >= start_date AND date_column < end_date
- DON'T USE EXTRACT(WEEK|DOW|QUARTER|DOY) - use DATE_TRUNC instead
- DON'T USE INTERVAL expressions

DATE TRUNCATION GUIDELINES:
- EXAMINE example values in date/timestamp columns to determine appropriate truncation level
- If example values show timestamps with T00:00:00 (e.g., "2021-10-29T00:00:00"), these are DATE values stored as timestamps - use exact date matching or TRUNC(date_column) for day-level queries
- If example values show actual times (e.g., "2021-10-29T17:32:05"), these are true TIMESTAMP values - consider if user wants day-level precision with TRUNC(date_column) or exact timestamp matching
- If example values show only dates (2023-12-15), user likely wants exact date matching
- If example values show month patterns (2023-12-01), user likely wants month-level: TRUNC(date_column, 'MM')
- For "recent" or "latest" queries, consider if user wants today/this week/this month based on data granularity
- IMPORTANT: T00:00:00 timestamps indicate date-only data - treat as dates, not times
- Always check example values before deciding on date truncation strategy

SORTING RULES:
- USE "NULLS LAST" with ORDER BY: ORDER BY amount DESC NULLS LAST

VIEW USAGE:
- USE views to simplify queries when available
- Use the actual view name from CREATE VIEW statements

PROHIBITED FEATURES:
- DON'T USE FILTER(WHERE expression) clauses
- DON'T USE EXTRACT(EPOCH FROM expression)
- DON'T USE LAX_* functions (LAX_BOOL, LAX_FLOAT64, etc.)

ALLOWED FUNCTIONS:
Aggregation: AVG, COUNT, MAX, MIN, SUM, ARRAY_AGG, BOOL_OR
Math: ABS, CBRT, CEIL, EXP, FLOOR, LN, ROUND, SIGN, GREATEST, LEAST, MOD, POWER  
String: LENGTH, REVERSE, CHR, CONCAT, FORMAT, LOWER, LPAD, LTRIM, POSITION, 
        REPLACE, RPAD, RTRIM, STRPOS, SUBSTR, SUBSTRING, TRANSLATE, TRIM, UPPER
Date: CURRENT_DATE, DATE_TRUNC, EXTRACT
Operators: +, -, *, /, ||, <, >, <=, >=, <>, !=

=== END ORACLE SQL RULES ===
"""
        
    def extract_sql_queries(self, content: str) -> List[str]:
        """Extract SQL queries from content."""
        matches = self.sql_pattern.findall(content)
        return [match.strip() for match in matches if match.strip()]
    
    def parse_report_document(self, content: str) -> Dict[str, Any]:
        """Parse report document content to extract structured information."""
        metadata = {
            "sql_queries": [],
            "table_references": [],
            "complexity_indicators": []
        }
        
        # Extract SQL queries
        sql_queries = self.extract_sql_queries(content)
        metadata["sql_queries"] = sql_queries
        
        # Extract table references from SQL
        table_pattern = re.compile(r'\bFROM\s+(\w+)', re.IGNORECASE)
        join_pattern = re.compile(r'\bJOIN\s+(\w+)', re.IGNORECASE)
        
        tables = set()
        for query in sql_queries:
            tables.update(table_pattern.findall(query))
            tables.update(join_pattern.findall(query))
        
        metadata["table_references"] = list(tables)
        
        # Determine complexity indicators
        complexity_indicators = []
        for query in sql_queries:
            query_lower = query.lower()
            if 'join' in query_lower:
                complexity_indicators.append("uses_joins")
            if 'subquery' in query_lower or '(' in query:
                complexity_indicators.append("has_subqueries")
            if 'group by' in query_lower:
                complexity_indicators.append("uses_aggregation")
            if 'window' in query_lower or 'over(' in query_lower:
                complexity_indicators.append("uses_window_functions")
        
        metadata["complexity_indicators"] = list(set(complexity_indicators))
        
        return metadata
    
    def parse_schema_document(self, content: str) -> Dict[str, Any]:
        """Parse schema document content to extract structured information."""
        metadata = {
            "tables": [],
            "columns": [],
            "relationships": [],
            "data_types": []
        }
        
        try:
            # Try to parse as JSON first
            if content.strip().startswith('{'):
                schema_data = json.loads(content)
                return self._process_json_schema(schema_data)
        except json.JSONDecodeError:
            pass
        
        # Parse as text format
        return self._parse_text_schema(content)
    
    def _process_json_schema(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process JSON schema format."""
        metadata = {
            "tables": [],
            "columns": [],
            "relationships": [],
            "data_types": []
        }
        
        if "tables" in schema_data:
            for table_info in schema_data["tables"]:
                if isinstance(table_info, dict):
                    table_name = table_info.get("name", "")
                    metadata["tables"].append(table_name)
                    
                    if "columns" in table_info:
                        for col in table_info["columns"]:
                            if isinstance(col, dict):
                                col_name = col.get("name", "")
                                data_type = col.get("type", "")
                                metadata["columns"].append(f"{table_name}.{col_name}")
                                if data_type:
                                    metadata["data_types"].append(data_type)
        
        if "relationships" in schema_data:
            metadata["relationships"] = schema_data["relationships"]
        
        return metadata
    
    def _parse_text_schema(self, content: str) -> Dict[str, Any]:
        """Parse text-based schema format."""
        metadata = {
            "tables": [],
            "columns": [],
            "relationships": [],
            "data_types": []
        }
        
        # Extract table names (looking for CREATE TABLE statements or table declarations)
        table_pattern = re.compile(r'CREATE\s+TABLE\s+(\w+)', re.IGNORECASE)
        table_matches = table_pattern.findall(content)
        metadata["tables"] = table_matches
        
        # Extract column definitions
        column_pattern = re.compile(r'(\w+)\s+(VARCHAR|INT|INTEGER|TEXT|DECIMAL|DATE|TIMESTAMP|BOOLEAN)', re.IGNORECASE)
        column_matches = column_pattern.findall(content)
        
        for col_name, data_type in column_matches:
            metadata["columns"].append(col_name)
            metadata["data_types"].append(data_type.upper())
        
        # Look for foreign key relationships
        fk_pattern = re.compile(r'FOREIGN\s+KEY\s*\(\s*(\w+)\s*\)\s*REFERENCES\s+(\w+)\s*\(\s*(\w+)\s*\)', re.IGNORECASE)
        fk_matches = fk_pattern.findall(content)
        
        for fk_col, ref_table, ref_col in fk_matches:
            metadata["relationships"].append({
                "type": "foreign_key",
                "column": fk_col,
                "references_table": ref_table,
                "references_column": ref_col
            })
        
        return metadata
    
    def validate_document_content(
        self, 
        content: str, 
        document_type: DocumentType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """Validate document content based on type."""
        errors = []
        
        if not content.strip():
            errors.append("Content cannot be empty")
            return False, errors
        
        if document_type == DocumentType.REPORT:
            return self._validate_report_content(content, metadata or {})
        elif document_type == DocumentType.SCHEMA:
            return self._validate_schema_content(content, metadata or {})
        
        return True, []
    
    def _validate_report_content(
        self, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate report document content."""
        errors = []
        
        # Check for required metadata fields
        required_fields = ["sql_query", "expected_output_description"]
        for field in required_fields:
            if field not in metadata or not metadata[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate SQL syntax (basic check)
        if "sql_query" in metadata:
            sql_query = metadata["sql_query"]
            if not self._basic_sql_validation(sql_query):
                errors.append("SQL query appears to be invalid")
        
        # Extract and validate SQL from content
        sql_queries = self.extract_sql_queries(content)
        if not sql_queries and "sql_query" not in metadata:
            errors.append("No SQL queries found in content or metadata")
        
        return len(errors) == 0, errors
    
    def _validate_schema_content(
        self, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate schema document content."""
        errors = []
        
        # Check for required metadata fields
        required_fields = ["table_name", "columns"]
        for field in required_fields:
            if field not in metadata or not metadata[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate columns format
        if "columns" in metadata:
            columns = metadata["columns"]
            if not isinstance(columns, list):
                errors.append("Columns must be a list")
            elif len(columns) == 0:
                errors.append("At least one column must be specified")
        
        # Try to parse schema from content
        try:
            parsed_metadata = self.parse_schema_document(content)
            if not parsed_metadata.get("tables") and not parsed_metadata.get("columns"):
                errors.append("Could not extract table or column information from content")
        except Exception as e:
            errors.append(f"Error parsing schema content: {str(e)}")
        
        return len(errors) == 0, errors
    
    def _basic_sql_validation(self, sql: str) -> bool:
        """Basic SQL syntax validation."""
        sql = sql.strip().upper()
        
        # Check for basic SQL keywords
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']
        starts_with_keyword = any(sql.startswith(keyword) for keyword in sql_keywords)
        
        if not starts_with_keyword:
            return False
        
        # Basic bracket matching
        if sql.count('(') != sql.count(')'):
            return False
        
        # Check for common SQL structure
        if sql.startswith('SELECT') and 'FROM' not in sql:
            return False
        
        return True
    
    def extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content for indexing."""
        # Remove SQL queries to avoid noise
        content_without_sql = self.sql_pattern.sub('', content)
        
        # Simple keyword extraction (can be enhanced with NLP)
        words = re.findall(r'\b[A-Za-z]{3,}\b', content_without_sql)
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
            'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 
            'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'
        }
        
        keywords = [word.lower() for word in words if word.lower() not in stop_words]
        
        # Return unique keywords sorted by frequency
        from collections import Counter
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(20)]
    
    def convert_json_to_dolphin_format(self, json_content: str, document_type: DocumentType) -> str:
        """Convert JSON content to Dolphin format for better vectorization."""
        try:
            data = json.loads(json_content)
            return self._format_json_as_dolphin(data, document_type)
        except json.JSONDecodeError:
            # If not valid JSON, return original content
            return json_content
    
    def _format_json_as_dolphin(self, data: Any, document_type: DocumentType, level: int = 0) -> str:
        """Format JSON data as readable Dolphin-style text."""
        indent = "  " * level
        lines = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{indent}{key}:")
                    lines.append(self._format_json_as_dolphin(value, document_type, level + 1))
                else:
                    lines.append(f"{indent}{key}: {value}")
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    lines.append(f"{indent}Item {i + 1}:")
                    lines.append(self._format_json_as_dolphin(item, document_type, level + 1))
                else:
                    lines.append(f"{indent}- {item}")
        
        else:
            return f"{indent}{data}"
        
        result = "\n".join(lines)
        
        # Add context-specific formatting based on document type
        if document_type == DocumentType.SCHEMA:
            result = self._enhance_schema_dolphin_format(result)
        elif document_type == DocumentType.REPORT:
            result = self._enhance_report_dolphin_format(result)
        
        return result
    
    def _enhance_schema_dolphin_format(self, content: str) -> str:
        """Enhance Dolphin format for schema documents."""
        lines = content.split('\n')
        enhanced_lines = []
        
        for line in lines:
            # Enhance table information
            if 'table' in line.lower() and ':' in line:
                enhanced_lines.append(f"DATABASE TABLE: {line}")
            # Enhance column information
            elif 'column' in line.lower() and ':' in line:
                enhanced_lines.append(f"TABLE COLUMN: {line}")
            # Enhance data type information
            elif any(dtype in line.lower() for dtype in ['varchar', 'int', 'decimal', 'date', 'boolean']):
                enhanced_lines.append(f"DATA TYPE: {line}")
            # Enhance relationship information
            elif any(rel in line.lower() for rel in ['foreign', 'primary', 'key', 'reference']):
                enhanced_lines.append(f"RELATIONSHIP: {line}")
            else:
                enhanced_lines.append(line)
        
        # Add semantic headers
        result = "=== DATABASE SCHEMA INFORMATION ===\n"
        result += "\n".join(enhanced_lines)
        result += "\n=== END SCHEMA INFORMATION ==="
        
        return result
    
    def _enhance_report_dolphin_format(self, content: str) -> str:
        """Enhance Dolphin format for report documents."""
        lines = content.split('\n')
        enhanced_lines = []
        
        for line in lines:
            # Enhance SQL query information
            if 'sql' in line.lower() and ':' in line:
                enhanced_lines.append(f"SQL QUERY: {line}")
            # Enhance description information
            elif 'description' in line.lower() and ':' in line:
                enhanced_lines.append(f"QUERY DESCRIPTION: {line}")
            # Enhance result information
            elif 'result' in line.lower() and ':' in line:
                enhanced_lines.append(f"EXPECTED RESULT: {line}")
            # Enhance table references
            elif any(table_ref in line.lower() for table_ref in ['table', 'from', 'join']):
                enhanced_lines.append(f"TABLE REFERENCE: {line}")
            else:
                enhanced_lines.append(line)
        
        # Add semantic headers
        result = "=== SQL QUERY REPORT ===\n"
        result += "\n".join(enhanced_lines)
        result += "\n=== END QUERY REPORT ==="
        
        return result
    
    def is_json_content(self, content: str) -> bool:
        """Check if content is valid JSON."""
        try:
            json.loads(content.strip())
            return True
        except json.JSONDecodeError:
            return False
    
    def create_semantic_chunks(self, json_content: str, document_type: DocumentType) -> List[Dict[str, Any]]:
        """Create semantic chunks that preserve business context and relationships.
        
        Based on WrenAI's approach, this creates entity-focused chunks that maintain
        business meaning and relationships between data elements.
        """
        try:
            data = json.loads(json_content)
            if document_type == DocumentType.SCHEMA:
                return self._create_schema_semantic_chunks(data)
            elif document_type == DocumentType.REPORT:
                return self._create_report_semantic_chunks(data)
            elif document_type == DocumentType.LOOKUP_METADATA:
                return self._create_lookup_semantic_chunks(data)
            else:
                # Fallback to single chunk
                return [{
                    "content": self.convert_json_to_dolphin_format(json_content, document_type),
                    "metadata": {"chunk_type": "full_document"}
                }]
        except json.JSONDecodeError:
            # If not valid JSON, return as single chunk
            return [{
                "content": json_content,
                "metadata": {"chunk_type": "text_document"}
            }]
    
    def _create_schema_semantic_chunks(self, schema_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create semantic chunks for schema documents based on business entities."""
        chunks = []
        
        # Basic schema information chunk
        schema_info = {
            "catalog": schema_data.get("catalog", ""),
            "schema": schema_data.get("schema", ""),
            "version": schema_data.get("version", "")
        }
        
        schema_overview = f"""=== DATABASE SCHEMA OVERVIEW ===
Catalog: {schema_info['catalog']}
Schema: {schema_info['schema']}
Version: {schema_info['version']}

This schema contains information about financial data including deals, tranches, and Fixed Income instruments.
The schema supports queries about deal statuses, tranche information, and asset classification.
=== END OVERVIEW ==="""
        
        chunks.append({
            "content": schema_overview,
            "metadata": {
                "chunk_type": "schema_overview",
                "catalog": schema_info["catalog"],
                "schema": schema_info["schema"],
                "business_domain": "financial_instruments"
            }
        })
        
        # Process models (tables) as separate business entity chunks
        if "models" in schema_data:
            for model in schema_data["models"]:
                table_chunk = self._create_table_semantic_chunk(model, schema_info)
                chunks.append(table_chunk)
        
        # Process views as separate analytical entity chunks
        if "views" in schema_data:
            for view in schema_data["views"]:
                view_chunk = self._create_view_semantic_chunk(view, schema_info)
                chunks.append(view_chunk)
        
        # Create relationship chunks that preserve business logic
        if "relationships" in schema_data:
            relationship_chunks = self._create_relationship_semantic_chunks(
                schema_data["relationships"], schema_info
            )
            chunks.extend(relationship_chunks)
        
        return chunks
    
    def _create_table_semantic_chunk(self, table_data: Dict[str, Any], schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a semantic chunk for a database table with business context."""
        # Handle both name formats
        table_name = table_data.get("name", table_data.get("table_name", "unknown_table"))
        
        # Handle properties structure
        properties = table_data.get("properties", {})
        description = properties.get("description", "")
        
        # Build comprehensive table information in clean Dolphin format
        content_lines = [
            f"DATABASE TABLE: {table_name.upper()}",
            f"Catalog: {schema_info['catalog']}",
            f"Schema: {schema_info['schema']}",
            f"Business Purpose: {description}" if description else "",
            ""
        ]
        
        # Add column information with business context
        if "columns" in table_data:
            content_lines.append("COLUMNS:")
            for col in table_data["columns"]:
                col_name = col.get("name", "")
                col_type = col.get("type", "")
                example_values = col.get("example_values", [])
                
                # Handle different nullable field names
                not_null = not col.get("nullable", True) or col.get("notNull", False)
                
                # Handle relationships
                relationship = col.get("relationship", [])
                col_description = col.get("description", "")
                
                # Handle key information
                key_info = col.get("key", "")
                
                content_lines.append(f"Column: {col_name}")
                content_lines.append(f"  Type: {col_type}")
                if key_info:
                    content_lines.append(f"  Key: {key_info}")
                if col_description:
                    content_lines.append(f"  Description: {col_description}")
                if example_values:
                    content_lines.append(f"  Example values: {', '.join(map(str, example_values[:5]))}")
                if not_null:
                    content_lines.append(f"  Required: Yes (NOT NULL)")
                else:
                    content_lines.append(f"  Required: No (nullable)")
                if relationship:
                    content_lines.append(f"  Related to: {', '.join(relationship)}")
                content_lines.append("")
        
        # Add primary key information
        if "primaryKey" in table_data:
            content_lines.append(f"Primary Key: {table_data['primaryKey']}")
            content_lines.append("")
        
        # Add reference SQL for data access
        if "refSql" in table_data:
            content_lines.append("Query to access all data:")
            content_lines.append(table_data['refSql'])
            content_lines.append("")
        
        # Add business context summary
        content_lines.extend([
            "BUSINESS CONTEXT:",
            f"This table stores {description.lower() if description else 'data'} in the {schema_info['catalog']} catalog.",
            f"Table can be queried as: {schema_info['catalog']}.{schema_info['schema']}.{table_name}",
            ""
        ])
        
        # Extract business terms for metadata
        business_terms = self._extract_business_terms(table_name, description, table_data)
        
        return {
            "content": "\n".join(content_lines),
            "metadata": {
                "chunk_type": "table_entity",
                "table_name": table_name,
                "business_terms": business_terms,
                "catalog": schema_info["catalog"],
                "schema": schema_info["schema"],
                "entity_type": self._classify_entity_type(table_name),
                "columns": [col.get("name", "") for col in table_data.get("columns", [])]
            }
        }
    
    def _create_view_semantic_chunk(self, view_data: Dict[str, Any], schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a semantic chunk for a database view with analytical context."""
        # Handle both name formats
        view_name = view_data.get("name", view_data.get("view_name", "unknown_view"))
        
        # Handle properties structure
        properties = view_data.get("properties", {})
        description = properties.get("description", "")
        view_sql = properties.get("view_sql", view_data.get("query", ""))
        view_type = properties.get("view_type", "")
        security_abstraction = properties.get("security_abstraction", "")
        
        content_lines = [
            f"DATABASE VIEW: {view_name.upper()}",
            f"Catalog: {schema_info['catalog']}",
            f"Schema: {schema_info['schema']}",
            f"View Type: {view_type}" if view_type else "",
            f"Security Abstraction: {security_abstraction}" if security_abstraction else "",
            f"Business Purpose: {description}" if description else "",
            ""
        ]
        
        # Add view definition with business context
        if view_sql:
            content_lines.append("View Definition SQL:")
            content_lines.append(view_sql)
            content_lines.append("")
        
        # Add column information
        if "columns" in view_data:
            content_lines.append("OUTPUT COLUMNS:")
            for col in view_data["columns"]:
                col_name = col.get("name", "")
                col_type = col.get("type", "")
                example_values = col.get("example_values", [])
                
                # Handle different nullable field names
                not_null = not col.get("nullable", True) or col.get("notNull", False)
                
                content_lines.append(f"Column: {col_name}")
                content_lines.append(f"  Type: {col_type}")
                if example_values:
                    content_lines.append(f"  Example values: {', '.join(map(str, example_values[:5]))}")
                if not_null:
                    content_lines.append(f"  Required: Yes (NOT NULL)")
                else:
                    content_lines.append(f"  Required: No (nullable)")
                content_lines.append("")
        
        # Add reference SQL
        if "refSql" in view_data:
            content_lines.append("Query to access view data:")
            content_lines.append(view_data['refSql'])
            content_lines.append("")
        
        # Add business context
        content_lines.extend([
            "BUSINESS CONTEXT:",
            f"This view provides {description.lower() if description else 'analytical data'} from the {schema_info['catalog']} catalog.",
            f"View can be queried as: {schema_info['catalog']}.{schema_info['schema']}.{view_name}",
            ""
        ])
        
        business_terms = self._extract_business_terms(view_name, description, view_data)
        
        return {
            "content": "\n".join(content_lines),
            "metadata": {
                "chunk_type": "view_entity",
                "view_name": view_name,
                "view_type": view_type,
                "business_terms": business_terms,
                "catalog": schema_info["catalog"],
                "schema": schema_info["schema"],
                "entity_type": "analytical_view"
            }
        }
    
    def _create_relationship_semantic_chunks(self, relationships: List[Dict[str, Any]], schema_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create semantic chunks for relationships that preserve business logic."""
        chunks = []
        
        # Group relationships by business domain
        relationship_groups = self._group_relationships_by_domain(relationships)
        
        for domain, domain_relationships in relationship_groups.items():
            content_lines = [
                f"=== {domain.upper()} RELATIONSHIPS ===",
                f"Business Domain: {domain}",
                f"Schema: {schema_info['catalog']}.{schema_info['schema']}",
                ""
            ]
            
            for rel in domain_relationships:
                # Handle both name formats
                rel_name = rel.get("name", rel.get("relationship_name", ""))
                
                # Handle both model formats  
                models = rel.get("models", rel.get("tables", []))
                
                # Handle join type
                join_type = rel.get("joinType", rel.get("type", ""))
                
                # Handle condition
                condition = rel.get("condition", "")
                
                # Handle properties
                properties = rel.get("properties", {})
                constraint_name = properties.get("constraint_name", "")
                relationship_type = properties.get("relationship_type", "")
                example_join_sql = properties.get("example_join_sql", "")
                
                content_lines.append(f"RELATIONSHIP: {rel_name}")
                content_lines.append(f"  Connected Tables: {' <-> '.join(models)}")
                content_lines.append(f"  Join Type: {join_type}")
                if constraint_name:
                    content_lines.append(f"  Constraint Name: {constraint_name}")
                if relationship_type:
                    content_lines.append(f"  Relationship Type: {relationship_type}")
                content_lines.append(f"  Join Condition: {condition}")
                
                # Add example SQL from properties first, then fallback to direct field
                if example_join_sql:
                    content_lines.append("  Example Join SQL (from properties):")
                    content_lines.append(f"  {example_join_sql}")
                elif rel.get("example_sql"):
                    content_lines.append("  Example Join SQL:")
                    content_lines.append(f"  {rel.get('example_sql')}")
                
                # Add description if available
                rel_description = rel.get("description", "")
                if rel_description:
                    content_lines.append(f"  Description: {rel_description}")
                
                content_lines.append("")
            
            content_lines.append(f"=== END {domain.upper()} RELATIONSHIPS ===")
            
            chunks.append({
                "content": "\n".join(content_lines),
                "metadata": {
                    "chunk_type": "relationship_domain",
                    "business_domain": domain,
                    "catalog": schema_info["catalog"],
                    "schema": schema_info["schema"],
                    "related_tables": list(set([table for rel in domain_relationships for table in rel.get("tables", rel.get("models", []))]))
                }
            })
        
        return chunks
    
    def _extract_business_terms(self, entity_name: str, description: str, entity_data: Dict[str, Any]) -> List[str]:
        """Extract business terms and domain concepts from entity information."""
        terms = []
        
        # Extract from entity name
        terms.extend(self._extract_terms_from_name(entity_name))
        
        # Extract from description
        if description:
            terms.extend(self._extract_terms_from_text(description))
        
        # Extract from column names if available
        if "columns" in entity_data:
            for col in entity_data["columns"]:
                col_name = col.get("name", "")
                terms.extend(self._extract_terms_from_name(col_name))
        
        # Extract from column data types and example values
        if "columns" in entity_data:
            for col in entity_data["columns"]:
                col_type = col.get("type", "")
                if col_type:
                    terms.extend(self._extract_terms_from_name(col_type))
                
                # Extract meaningful terms from example values
                example_values = col.get("example_values", [])
                for value in example_values[:3]:  # Limit to first 3 examples
                    if isinstance(value, str) and len(value) > 2:
                        terms.extend(self._extract_terms_from_text(str(value)))
        
        # Domain-specific terms
        domain_terms = [
            # Financial domain
            "deal", "deals", "tranche", "tranches", "fixed_income", "fixed income",
            "announced", "status", "asset_class", "security", "bond", "loan",
            "maturity", "rating", "yield", "coupon", "principal", "issuance",
            # User/Account domain
            "user", "users", "account", "accounts", "customer", "customers",
            "profile", "profiles", "person", "people", "individual", "individuals",
            "authentication", "login", "credential", "credentials",
            # General business terms
            "name", "first", "last", "email", "phone", "address", "contact",
            "created", "updated", "deleted", "modified", "timestamp", "date",
            "active", "inactive", "enabled", "disabled", "team", "role", "permission"
        ]
        
        # Check for domain terms in the content
        text_content = f"{entity_name} {description}".lower()
        for term in domain_terms:
            if term in text_content:
                terms.append(term)
        
        # Add column names as business terms (they're often meaningful)
        if "columns" in entity_data:
            for col in entity_data["columns"]:
                col_name = col.get("name", "")
                if col_name:
                    terms.append(col_name.lower())
        
        return list(set(terms))
    
    def get_sql_rules(self) -> str:
        """Get Oracle SQL rules for use in prompts only."""
        return self.oracle_sql_rules
    
    def get_sql_examples(self) -> str:
        """Get Oracle SQL query examples for use in prompts only."""
        return """=== ORACLE SQL QUERY EXAMPLES ===
-- Select all records from a table (use only when user asks for all columns)
SELECT * FROM schema_name.table_name;

-- Count total records
SELECT COUNT(*) FROM schema_name.table_name;

-- Select specific columns (always qualify with table name)
SELECT table_name.column1, table_name.column2
FROM schema_name.table_name;

-- Case-insensitive search example
SELECT table_name.id, table_name.name
FROM schema_name.table_name
WHERE lower(table_name.column_name) = lower('search_value');

-- Pattern matching example
SELECT table_name.id, table_name.name
FROM schema_name.table_name
WHERE lower(table_name.column_name) LIKE lower('%pattern%');

-- Sorting with nulls last
SELECT * FROM schema_name.table_name
ORDER BY column_name DESC NULLS LAST;

-- Join tables example
SELECT t1.column1, t2.column2
FROM schema_name.table1 t1
JOIN schema_name.table2 t2
  ON t1.join_column = t2.join_column;

-- View query example
SELECT view_name.column1, view_name.column2
FROM schema_name.view_name
WHERE view_name.filter_column = 'value';

-- Aggregate query example
SELECT table_name.category, COUNT(*) as count
FROM schema_name.table_name
GROUP BY table_name.category
ORDER BY count DESC;
"""
    
    def _create_lookup_semantic_chunks(self, lookup_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create semantic chunks for lookup metadata with rich context.
        
        Each chunk contains a specific lookup category with sufficient context
        for the LLM to understand what the lookup data represents and how to use it.
        """
        import structlog
        logger = structlog.get_logger(__name__)
        
        logger.info("Starting lookup semantic chunking", 
                   lookup_data_keys=list(lookup_data.keys()) if isinstance(lookup_data, dict) else str(type(lookup_data)))
        
        chunks = []
        
        # Handle single lookup document
        if "name" in lookup_data and "values" in lookup_data:
            lookup_name = lookup_data.get("name", "")
            description = lookup_data.get("description", "")
            values = lookup_data.get("values", [])
            
            # Create a comprehensive chunk for this lookup
            chunk = self._create_single_lookup_chunk(lookup_name, description, values)
            chunks.append(chunk)
        
        # Handle multiple lookups in one document
        elif "lookups" in lookup_data:
            for lookup in lookup_data["lookups"]:
                lookup_name = lookup.get("name", "")
                description = lookup.get("description", "")
                values = lookup.get("values", [])
                
                chunk = self._create_single_lookup_chunk(lookup_name, description, values)
                chunks.append(chunk)
        
        # Handle other possible structures
        else:
            # Try to find lookup data in any key
            for key, value in lookup_data.items():
                if isinstance(value, dict) and ("values" in value or "data" in value):
                    lookup_name = key
                    description = value.get("description", f"Lookup data for {key}")
                    values = value.get("values", value.get("data", []))
                    
                    chunk = self._create_single_lookup_chunk(lookup_name, description, values)
                    chunks.append(chunk)
        
        if not chunks:
            # Fallback chunk if no structured lookup data found
            logger.warning("No chunks created, using fallback")
            chunks.append({
                "content": f"LOOKUP METADATA\n{json.dumps(lookup_data, indent=2)}",
                "metadata": {
                    "chunk_type": "lookup_fallback",
                    "entity_type": "lookup",
                    "lookup_name": "unknown",
                    "value_count": 0
                }
            })
        
        logger.info("Lookup semantic chunking complete", chunk_count=len(chunks))
        return chunks
    
    def _create_single_lookup_chunk(self, lookup_name: str, description: str, values: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a single lookup chunk with rich context and searchable information."""
        
        # Build comprehensive content with context
        content_lines = [
            f"=== LOOKUP DATA: {lookup_name.upper()} ===",
            f"Lookup Name: {lookup_name}",
        ]
        
        if description:
            content_lines.append(f"Description: {description}")
        
        content_lines.extend([
            f"Purpose: Database lookup values for {lookup_name} field",
            f"Total Values: {len(values)}",
            ""
        ])
        
        # Add value mappings with context
        if values:
            content_lines.append("AVAILABLE VALUES:")
            content_lines.append("Format: ID -> Display Name")
            content_lines.append("")
            
            # Add each value with rich context
            for value in values:
                if isinstance(value, dict):
                    value_id = value.get("id", value.get("value_id", ""))
                    value_name = value.get("name", value.get("display_name", value.get("label", "")))
                    value_code = value.get("code", "")
                    
                    # Create searchable entry
                    if value_id and value_name:
                        entry = f"ID {value_id}: '{value_name}'"
                        if value_code and value_code != value_name:
                            entry += f" (code: {value_code})"
                        content_lines.append(entry)
                        
                        # Add alternative search terms
                        alt_names = value.get("aliases", value.get("alternatives", []))
                        if alt_names:
                            content_lines.append(f"  Aliases: {', '.join(alt_names)}")
        
        # Add usage context
        example_id = "1"
        if values:
            first_value = values[0]
            if isinstance(first_value, dict):
                example_id = first_value.get('id', '1')
            else:
                example_id = "1"
        
        content_lines.extend([
            "",
            "USAGE IN SQL:",
            f"Use the ID value in WHERE clauses: column_name = {example_id}",
            f"Search by name: column_name IN (SELECT id FROM lookup_table WHERE name LIKE '%search_term%')",
            "",
            "=== END LOOKUP DATA ==="
        ])
        
        # Extract search terms for better retrieval
        search_terms = self._extract_lookup_search_terms(lookup_name, description, values)
        
        # Ensure content is not empty
        content = "\n".join(content_lines).strip()
        if not content or len(content) < 10:
            content = f"LOOKUP DATA: {lookup_name}\nPurpose: Database lookup values\nTotal Values: {len(values)}\nContact administrator for more details."
        
        return {
            "content": content,
            "metadata": {
                "chunk_type": "lookup_data",
                "entity_type": "lookup",
                "lookup_name": lookup_name,
                "description": description or f"Lookup values for {lookup_name}",
                "value_count": len(values),
                "search_terms": search_terms,
                "values_preview": [v.get("name", "") if isinstance(v, dict) else str(v) for v in values[:5]] if values else []
            }
        }
    
    def _extract_lookup_search_terms(self, lookup_name: str, description: str, values: List[Dict[str, Any]]) -> List[str]:
        """Extract search terms for better lookup retrieval."""
        terms = set()
        
        # Add terms from lookup name
        terms.update(self._extract_terms_from_name(lookup_name))
        
        # Add terms from description
        if description:
            terms.update(self._extract_terms_from_text(description))
        
        # Add terms from value names (first few)
        for value in values[:10]:  # Limit to avoid too many terms
            if isinstance(value, dict):
                value_name = value.get("name", value.get("display_name", ""))
                if value_name:
                    terms.update(self._extract_terms_from_name(str(value_name)))
            elif isinstance(value, str):
                # Handle string values directly
                terms.update(self._extract_terms_from_name(value))
        
        return list(terms)
    
    def _extract_terms_from_name(self, name: str) -> List[str]:
        """Extract meaningful terms from entity names."""
        # Split on underscores and camelCase
        import re
        parts = re.split(r'[_\s]+|(?=[A-Z])', name)
        return [part.lower() for part in parts if len(part) > 2]
    
    def _extract_terms_from_text(self, text: str) -> List[str]:
        """Extract meaningful terms from descriptive text."""
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        # Filter out common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'this', 'that', 'with', 'have', 'will', 'from'}
        return [word for word in words if word not in stop_words]
    
    def _classify_entity_type(self, table_name: str) -> str:
        """Classify the business entity type based on table name."""
        name_lower = table_name.lower()
        
        # User/Account entities
        if any(term in name_lower for term in ['user', 'users', 'customer', 'customers', 'person', 'people', 'account', 'accounts']):
            return "user_entity"
        # Financial entities
        elif any(term in name_lower for term in ['deal', 'deals']):
            return "deal_entity"
        elif any(term in name_lower for term in ['tranche', 'tranches']):
            return "tranche_entity"
        elif any(term in name_lower for term in ['security', 'securities', 'bond', 'loan']):
            return "security_entity"
        elif any(term in name_lower for term in ['asset', 'class', 'classification']):
            return "classification_entity"
        elif any(term in name_lower for term in ['status', 'state']):
            return "status_entity"
        # Transaction/Order entities
        elif any(term in name_lower for term in ['order', 'orders', 'transaction', 'transactions', 'trade', 'trades']):
            return "transaction_entity"
        # Product/Item entities
        elif any(term in name_lower for term in ['product', 'products', 'item', 'items', 'inventory']):
            return "product_entity"
        # Team/Organization entities
        elif any(term in name_lower for term in ['team', 'teams', 'group', 'groups', 'organization', 'org']):
            return "organizational_entity"
        else:
            return "general_entity"
    
    def _group_relationships_by_domain(self, relationships: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group relationships by business domain for better semantic chunking."""
        domains = {}
        
        for rel in relationships:
            domain = self._determine_relationship_domain(rel)
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(rel)
        
        return domains
    
    def _determine_relationship_domain(self, relationship: Dict[str, Any]) -> str:
        """Determine the business domain of a relationship."""
        models = relationship.get("tables", relationship.get("models", []))
        rel_name = relationship.get("relationship_name", relationship.get("name", "")).lower()
        
        # Check for financial domain patterns
        financial_patterns = ['deal', 'tranche', 'security', 'bond', 'asset']
        if any(pattern in rel_name or any(pattern in model.lower() for model in models) for pattern in financial_patterns):
            return "financial_instruments"
        
        # Check for status/classification patterns  
        status_patterns = ['status', 'classification', 'type', 'category']
        if any(pattern in rel_name or any(pattern in model.lower() for model in models) for pattern in status_patterns):
            return "classification_and_status"
        
        return "general_relationships"
    
    def _create_report_semantic_chunks(self, report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create semantic chunks for report documents (placeholder for future implementation)."""
        # For now, return as single chunk - can be enhanced later
        return [{
            "content": self._format_json_as_dolphin(report_data, DocumentType.REPORT),
            "metadata": {"chunk_type": "report_document"}
        }]
    
    def create_individual_documents(self, json_content: str, document_type: DocumentType) -> List[Dict[str, Any]]:
        """Create individual documents for each model/view/relationship in schema metadata.
        
        This method takes the main_schema_metadata.json and creates separate documents
        for each model, view, and relationship. Each document contains complete information
        about that entity in dolphin format, optimized for semantic search and retrieval.
        
        Args:
            json_content: The main_schema_metadata.json content as string
            document_type: The type of document (should be SCHEMA)
            
        Returns:
            List of individual documents, each containing complete entity information
        """
        import structlog
        logger = structlog.get_logger(__name__)
        
        try:
            logger.info("Creating individual documents", 
                       document_type=document_type.value if hasattr(document_type, 'value') else str(document_type),
                       content_length=len(json_content),
                       content_preview=json_content[:100])
            
            data = json.loads(json_content)
            logger.info("Parsed JSON successfully", 
                       data_keys=list(data.keys()) if isinstance(data, dict) else str(type(data)))
            
            if document_type == DocumentType.SCHEMA:
                result = self._create_individual_schema_documents(data)
                logger.info("Created schema documents", count=len(result))
                return result
            elif document_type == DocumentType.LOOKUP_METADATA:
                logger.info("Processing lookup metadata")
                result = self._create_lookup_semantic_chunks(data)
                logger.info("Created lookup documents", 
                           count=len(result),
                           chunk_types=[r.get("metadata", {}).get("chunk_type") for r in result])
                return result
            else:
                # Fallback for non-schema documents
                logger.info("Using fallback for unknown document type")
                return [{
                    "content": self.convert_json_to_dolphin_format(json_content, document_type),
                    "metadata": {
                        "chunk_type": "full_document",
                        "entity_type": "unknown"
                    }
                }]
        except json.JSONDecodeError as e:
            logger.error("JSON decode error", error=str(e), content_preview=json_content[:200])
            # If not valid JSON, return as single document
            return [{
                "content": json_content,
                "metadata": {
                    "chunk_type": "text_document",
                    "entity_type": "unknown"
                }
            }]
        except Exception as e:
            logger.error("Failed to create individual documents", error=str(e))
            import traceback
            logger.error("Traceback", traceback=traceback.format_exc())
            return []
    
    def _create_individual_schema_documents(self, schema_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create individual documents for each model, view, and relationship in schema data.
        
        This creates completely separate documents for each entity:
        - One document per model (table)
        - One document per view  
        - One document per relationship
        
        Each document contains the complete information for that entity in dolphin format.
        """
        documents = []
        
        # Extract basic schema information
        catalog = schema_data.get("catalog", "")
        schema_name = schema_data.get("schema", "")
        
        # Create individual documents for each model (table)
        if "models" in schema_data:
            for model in schema_data["models"]:
                doc = self._create_individual_model_document(model, catalog, schema_name)
                documents.append(doc)
        
        # Create individual documents for each view
        if "views" in schema_data:
            for view in schema_data["views"]:
                doc = self._create_individual_view_document(view, catalog, schema_name)
                documents.append(doc)
        
        # Create individual documents for each relationship
        if "relationships" in schema_data:
            for relationship in schema_data["relationships"]:
                doc = self._create_individual_relationship_document(relationship, catalog, schema_name)
                documents.append(doc)
        
        return documents
    
    def _create_individual_model_document(self, model_data: Dict[str, Any], catalog: str, schema_name: str) -> Dict[str, Any]:
        """Create a complete document for a single database model/table."""
        table_name = model_data.get("table_name", model_data.get("name", "unknown_table"))
        
        # Handle nested description in properties structure
        description = ""
        if "properties" in model_data and "description" in model_data["properties"]:
            description = model_data["properties"]["description"]
        else:
            description = model_data.get("description", "")
        
        # Build comprehensive model document in dolphin format
        content_lines = [
            "=== DATABASE TABLE DOCUMENT ===",
            f"Table Name: {table_name.upper()}",
            f"Catalog: {catalog}",
            f"Schema: {schema_name}",
            f"Schema Qualified Name: {schema_name}.{table_name}",
            ""
        ]
        
        if description:
            content_lines.extend([
                "Table Description:",
                description,
                ""
            ])
        
        # Add detailed column information
        if "columns" in model_data:
            content_lines.append("=== TABLE COLUMNS ===")
            for col in model_data["columns"]:
                col_name = col.get("name", "")
                col_type = col.get("type", "")
                col_key = col.get("key", "")
                example_values = col.get("example_values", [])
                
                # Handle both nullable and notNull fields
                nullable = True  # Default to nullable
                if "nullable" in col:
                    nullable = col["nullable"]
                elif "notNull" in col:
                    nullable = not col["notNull"]
                
                # Handle default values
                default_value = col.get("default", "")
                
                content_lines.extend([
                    f"Column: {col_name}",
                    f"  Data Type: {col_type}",
                ])
                
                if col_key:
                    content_lines.append(f"  Key Type: {col_key}")
                
                content_lines.append(f"  Required: {'No (nullable)' if nullable else 'Yes (NOT NULL)'}")
                
                if default_value:
                    content_lines.append(f"  Default Value: {default_value}")
                
                if example_values:
                    example_str = ", ".join([f"'{val}'" for val in example_values[:5]])
                    content_lines.append(f"  Example Values: {example_str}")
                
                content_lines.append("")
        
        # Add primary key information
        if "primaryKey" in model_data:
            content_lines.extend([
                "=== PRIMARY KEY ===",
                f"Primary Key: {model_data['primaryKey']}",
                ""
            ])
        
        # Add reference SQL if available
        if "refSql" in model_data:
            content_lines.extend([
                "=== REFERENCE SQL ===",
                "Query to access all data:",
                model_data['refSql'],
                ""
            ])
        
        # Add business context and usage information
        content_lines.extend([
            "=== BUSINESS CONTEXT ===",
            f"This table '{table_name}' contains {description.lower() if description else 'data'} in the database.",
            f"The table can be accessed using: SELECT * FROM {schema_name}.{table_name}",
            ""
        ])
        
        # Get primary key before using it
        primary_key = model_data.get("primaryKey", "")
        
        # Oracle SQL rules are now only added in prompts, not in document content
        
        # Extract business terms and metadata
        business_terms = self._extract_business_terms(table_name, description, model_data)
        entity_type = self._classify_entity_type(table_name)
        
        content_lines.append("=== END TABLE DOCUMENT ===")
        
        # Create comprehensive searchable content
        column_names = [col.get("name", "") for col in model_data.get("columns", [])]
        column_types = [col.get("type", "") for col in model_data.get("columns", [])]
        
        # Build rich searchable content that includes all important terms
        searchable_parts = [
            table_name,
            "table",
            "model", 
            description,
            ' '.join(business_terms),
            ' '.join(column_names),
            ' '.join(column_types),
            primary_key,
            catalog,
            schema_name,
            "oracle sql",
            "select",
            "join",
            "where",
            "order by"
        ]
        
        # Add example values for better searchability
        for col in model_data.get("columns", []):
            example_values = col.get("example_values", [])
            if example_values:
                searchable_parts.extend([str(val) for val in example_values[:3]])
        
        searchable_content = ' '.join([part for part in searchable_parts if part]).lower()
        
        return {
            "content": "\n".join(content_lines),
            "metadata": {
                "chunk_type": "individual_model",
                "entity_type": "model",
                "table_name": table_name,
                "catalog": catalog,
                "schema": schema_name,
                "business_terms": business_terms,
                "classified_entity_type": entity_type,
                "columns": column_names,
                "column_types": column_types,
                "primary_key": primary_key,
                "searchable_content": searchable_content,
                "sql_dialect": "Oracle SQL 19c",
                "supports_joins": True,
                "case_sensitive": False,
                "requires_schema_qualification": True,
                "allowed_operations": ["SELECT"],
                "prohibited_operations": ["DELETE", "UPDATE", "INSERT", "DROP", "CREATE", "ALTER"]
            }
        }
    
    def _create_individual_view_document(self, view_data: Dict[str, Any], catalog: str, schema_name: str) -> Dict[str, Any]:
        """Create a complete document for a single database view."""
        view_name = view_data.get("view_name", view_data.get("name", "unknown_view"))
        description = view_data.get("description", "")
        view_sql = view_data.get("query", view_data.get("view_sql", ""))
        
        # Build comprehensive view document in dolphin format
        content_lines = [
            "=== DATABASE VIEW DOCUMENT ===",
            f"View Name: {view_name.upper()}",
            f"Catalog: {catalog}",
            f"Schema: {schema_name}",
            f"Schema Qualified Name: {schema_name}.{view_name}",
            ""
        ]
        
        if description:
            content_lines.extend([
                "View Description:",
                description,
                ""
            ])
        
        # Add view definition SQL
        if view_sql:
            content_lines.extend([
                "=== VIEW DEFINITION ===",
                "SQL Definition:",
                view_sql,
                ""
            ])
        
        # Add detailed column information
        if "columns" in view_data:
            content_lines.append("=== VIEW COLUMNS ===")
            for col in view_data["columns"]:
                col_name = col.get("name", "")
                col_type = col.get("type", "")
                nullable = col.get("nullable", True)
                
                content_lines.extend([
                    f"Column: {col_name}",
                    f"  Data Type: {col_type}",
                    f"  Nullable: {'Yes' if nullable else 'No'}",
                    ""
                ])
        
        # Add business context
        content_lines.extend([
            "=== BUSINESS CONTEXT ===",
            f"This view '{view_name}' provides {description.lower() if description else 'analytical data'} from the database.",
            f"The view can be queried using: SELECT * FROM {schema_name}.{view_name}",
            ""
        ])
        
        # Oracle SQL examples are now added once in prompts, not duplicated in every document
        
        # Oracle SQL rules are now only added in prompts, not in document content
        
        business_terms = self._extract_business_terms(view_name, description, view_data)
        
        content_lines.append("=== END VIEW DOCUMENT ===")
        
        return {
            "content": "\n".join(content_lines),
            "metadata": {
                "chunk_type": "individual_view",
                "entity_type": "view",
                "view_name": view_name,
                "catalog": catalog,
                "schema": schema_name,
                "business_terms": business_terms,
                "columns": [col.get("name", "") for col in view_data.get("columns", [])],
                "searchable_content": f"{view_name} view {description} {' '.join(business_terms)}"
            }
        }
    
    def _create_individual_relationship_document(self, relationship_data: Dict[str, Any], catalog: str, schema_name: str) -> Dict[str, Any]:
        """Create a complete document for a single database relationship."""
        rel_name = relationship_data.get("relationship_name", relationship_data.get("name", "unknown_relationship"))
        tables = relationship_data.get("tables", relationship_data.get("models", []))
        rel_type = relationship_data.get("type", relationship_data.get("joinType", ""))
        description = relationship_data.get("description", "")
        example_sql = relationship_data.get("example_sql", "")
        
        # Build comprehensive relationship document in dolphin format
        content_lines = [
            "=== DATABASE RELATIONSHIP DOCUMENT ===",
            f"Relationship Name: {rel_name.upper()}",
            f"Catalog: {catalog}",
            f"Schema: {schema_name}",
            ""
        ]
        
        if description:
            content_lines.extend([
                "Relationship Description:",
                description,
                ""
            ])
        
        # Add relationship details
        content_lines.extend([
            "=== RELATIONSHIP DETAILS ===",
            f"Connected Tables: {' <-> '.join(tables)}",
            f"Relationship Type: {rel_type}",
            f"Schema Table References: {' <-> '.join([f'{schema_name}.{table}' for table in tables])}",
            ""
        ])
        
        # Add join information
        if example_sql:
            content_lines.extend([
                "=== JOIN EXAMPLE ===",
                "Example SQL to join these tables:",
                example_sql,
                ""
            ])
        
        # Add business context
        content_lines.extend([
            "=== BUSINESS CONTEXT ===",
            f"This relationship '{rel_name}' defines how {' and '.join(tables)} tables are connected.",
            f"It represents a {rel_type} relationship between the tables.",
            ""
        ])
        
        # Add usage patterns
        content_lines.extend([
            "=== USAGE PATTERNS ===",
            f"-- Query data from related tables using this relationship",
            f"-- Use JOIN operations to combine data from {' and '.join(tables)}",
            f"-- Relationship type: {rel_type}",
            ""
        ])
        
        business_terms = []
        for table in tables:
            business_terms.extend(self._extract_terms_from_name(table))
        business_terms.extend(self._extract_terms_from_text(description))
        business_terms = list(set(business_terms))  # Remove duplicates
        
        content_lines.append("=== END RELATIONSHIP DOCUMENT ===")
        
        return {
            "content": "\n".join(content_lines),
            "metadata": {
                "chunk_type": "individual_relationship",
                "entity_type": "relationship",
                "relationship_name": rel_name,
                "catalog": catalog,
                "schema": schema_name,
                "tables": tables,
                "relationship_type": rel_type,
                "business_terms": business_terms,
                "searchable_content": f"{rel_name} relationship {rel_type} {' '.join(tables)} {description} {' '.join(business_terms)}"
            }
        }