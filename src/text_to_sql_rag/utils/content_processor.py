"""Content processing utilities for document management."""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from ..models.document import DocumentType


class ContentProcessor:
    """Utility class for processing document content."""
    
    def __init__(self):
        self.sql_pattern = re.compile(
            r'(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s+.*?;',
            re.IGNORECASE | re.DOTALL
        )
        
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