#!/usr/bin/env python3
"""
Housekeeping script to restructure existing main_schema_metadata.json into new 4-tier metadata format.
This script creates column metadata files with business domain classification for robust RAG retrieval.
Uses: DDL, COLUMN_DETAILS, LOOKUP_METADATA, REPORTS (no BUSINESS_DESC/BUSINESS_RULES)
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ColumnMetadataRestructurer:
    """Restructure existing schema metadata into new column-focused format."""
    
    def __init__(self, input_file: str, output_dir: str):
        """
        Initialize restructurer.
        
        Args:
            input_file: Path to main_schema_metadata.json
            output_dir: Directory to save new column metadata files
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
    
    def load_existing_metadata(self) -> Dict[str, Any]:
        """Load existing schema metadata."""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded metadata from {self.input_file}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise
    
    def classify_business_domain(self, table_name: str) -> Dict[str, Any]:
        """Classify table into business domain using financial instruments knowledge."""
        
        name_lower = table_name.lower()
        
        # Core business entities classification
        if any(term in name_lower for term in ['issuer', 'company']):
            return {
                "business_domain": "issuer_management",
                "entity_type": "issuer",
                "core_table": True,
                "description": "Issuer management - companies seeking capital through bond issuances",
                "business_terms": ["issuer", "company", "capital_raise", "bonds"]
            }
        
        elif any(term in name_lower for term in ['deal', 'fundrais']):
            return {
                "business_domain": "deal_management", 
                "entity_type": "deal",
                "core_table": True,
                "description": "Deal management - fundraising initiatives created by JPMorgan for issuers",
                "business_terms": ["deal", "fundraising", "capital_raise", "bond_issuance"]
            }
        
        elif any(term in name_lower for term in ['tranche', 'bond']):
            return {
                "business_domain": "tranche_management",
                "entity_type": "tranche", 
                "core_table": True,
                "description": "Tranche management - individual bond issuances with distinct terms and conditions",
                "business_terms": ["tranche", "bond", "issuance", "pricing", "maturity", "ratings"]
            }
        
        elif any(term in name_lower for term in ['syndicate', 'bank']):
            return {
                "business_domain": "syndicate_operations",
                "entity_type": "syndicate_bank",
                "core_table": True, 
                "description": "Syndicate operations - financial institutions participating in bond distribution",
                "business_terms": ["syndicate", "bank", "distribution", "lead_manager", "co_manager"]
            }
        
        elif any(term in name_lower for term in ['order', 'allocation']):
            return {
                "business_domain": "order_management",
                "entity_type": "order",
                "core_table": True,
                "description": "Order management - investment requests from institutional investors",
                "business_terms": ["order", "allocation", "investor", "ioi", "indication_of_interest"]
            }
        
        elif any(term in name_lower for term in ['limit', 'conditional', 'reoffer']):
            return {
                "business_domain": "order_management", 
                "entity_type": "order_limit",
                "core_table": True,
                "description": "Order limits - investment amounts and conditions within orders",
                "business_terms": ["order_limit", "reoffer", "conditional", "investment_amount", "threshold"]
            }
        
        elif any(term in name_lower for term in ['investor', 'institution']):
            return {
                "business_domain": "investor_management",
                "entity_type": "investor",
                "core_table": True,
                "description": "Investor management - institutional investors placing orders",
                "business_terms": ["investor", "institutional", "investment", "portfolio"]
            }
        
        elif any(term in name_lower for term in ['status', 'lookup', 'ref']):
            return {
                "business_domain": "reference_data",
                "entity_type": "lookup",
                "core_table": False,
                "description": "Reference data - lookup tables for status codes and business values",
                "business_terms": ["status", "lookup", "reference", "code", "value"]
            }
        
        elif any(term in name_lower for term in ['audit', 'log', 'history']):
            return {
                "business_domain": "audit_trail",
                "entity_type": "audit",
                "core_table": False,
                "description": "Audit trail - tracking changes and system activities",
                "business_terms": ["audit", "log", "history", "tracking", "changes"]
            }
        
        elif any(term in name_lower for term in ['user', 'auth', 'login', 'permission']):
            return {
                "business_domain": "system_administration",
                "entity_type": "user",
                "core_table": False,
                "description": "System administration - user accounts and permissions",
                "business_terms": ["user", "authentication", "authorization", "permission", "role"]
            }
        
        else:
            return {
                "business_domain": "supporting_data",
                "entity_type": "support",
                "core_table": False,
                "description": "Supporting data - additional tables supporting core business operations", 
                "business_terms": ["supporting", "data", "auxiliary"]
            }
    
    def create_column_metadata_file(self, table_data: Dict[str, Any], table_name: str) -> Optional[Dict[str, Any]]:
        """Create enhanced column metadata with business domain classification."""
        
        try:
            logger.info(f"Processing {table_name} - Input data structure check")
            
            # Detailed input validation and logging
            if not table_data:
                logger.error(f"Empty table_data for {table_name}")
                return None
                
            if not isinstance(table_data, dict):
                logger.error(f"Invalid table_data type for {table_name}: {type(table_data)}")
                return None
            
            # Handle different data structures - columns might be directly on table_data or in properties
            columns = table_data.get("columns", [])
            if not columns and "properties" in table_data:
                # Try to get columns from properties wrapper
                properties = table_data.get("properties", {})
                columns = properties.get("columns", [])
            
            if not columns:
                logger.error(f"No columns found for {table_name}. Available keys: {list(table_data.keys())}")
                if "properties" in table_data:
                    logger.error(f"Properties keys: {list(table_data.get('properties', {}).keys())}")
                return None
                
            if not isinstance(columns, list):
                logger.error(f"Invalid columns type for {table_name}: {type(columns)}")
                return None
            
            logger.info(f"✓ {table_name}: {len(columns)} columns found. Keys: {list(table_data.keys())}")
            
            # Get business domain classification
            domain_info = self.classify_business_domain(table_name)
            logger.info(f"✓ {table_name}: Business domain = {domain_info['business_domain']}")
            
        except Exception as e:
            logger.error(f"❌ Failed during initial setup for {table_name}: {e}", exc_info=True)
            return None
        
        # Extract catalog and schema from metadata root level
        catalog = table_data.get("catalog", "unknown")
        schema = table_data.get("schema", "unknown")
        
        # If not found at root level, they might be in the parent metadata
        # We'll pass these from the parent call in restructure_all
        
        column_metadata = {
            "table_name": table_name.upper(),
            "catalog": catalog,
            "schema": schema,
            "business_domain": domain_info["business_domain"],
            "entity_type": domain_info["entity_type"],
            "core_table": domain_info["core_table"],
            "description": domain_info["description"],
            "business_terms": domain_info["business_terms"],
            "relationships": self._analyze_relationships(table_data, table_name),
            "columns": {}
        }
        
        # Process columns with enhanced metadata
        columns_processed = 0
        columns_failed = 0
        
        logger.info(f"Processing {len(columns)} columns for {table_name}...")
        
        for i, column in enumerate(columns):
            if not isinstance(column, dict):
                logger.error(f"Column {i} in {table_name} is not a dict: {type(column)}")
                columns_failed += 1
                continue
                
            col_name = column.get("name")
            if not col_name:
                logger.warning(f"Column {i} in {table_name} missing name. Keys: {list(column.keys())}")
                columns_failed += 1
                continue
                
            try:
                logger.debug(f"Processing column {col_name} in {table_name}")
                
                col_info = {
                    "name": col_name,
                    "type": column.get("type", "UNKNOWN"),
                    "nullable": column.get("nullable", True),
                    "description": self._generate_enhanced_column_description(col_name, column, domain_info),
                    "business_significance": self._determine_business_significance(col_name, domain_info)
                }
                columns_processed += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to process column {col_name} in {table_name}: {e}", exc_info=True)
                columns_failed += 1
                # Create minimal column info as fallback
                col_info = {
                    "name": col_name,
                    "type": column.get("type", "UNKNOWN"),
                    "nullable": column.get("nullable", True),
                    "description": f"Column {col_name} (processing failed)",
                    "business_significance": "business_attribute"
                }
            
            # Add key information if present
            if column.get("key"):
                col_info["constraint"] = column["key"]
                col_info["is_key"] = True
            else:
                col_info["is_key"] = False
            
            # Add example values if present and useful
            example_values = column.get("example_values", [])
            if example_values and len(example_values) > 0:
                col_info["example_values"] = example_values[:5]  # Limit to 5 examples
            
            # Enhanced relationship hints
            if col_name.lower().endswith('_id') and col_name.lower() != table_name.lower() + '_id':
                col_info["relationship_hint"] = self._generate_relationship_hint(col_name, domain_info)
            
            # Financial domain specific hints
            if domain_info["business_domain"] in ["tranche_management", "order_management"]:
                col_info["financial_context"] = self._add_financial_context(col_name, domain_info)
            
            column_metadata["columns"][col_name] = col_info
        
        logger.info(f"✓ Completed {table_name}: {columns_processed} columns processed, {columns_failed} failed")
        
        if columns_processed == 0:
            logger.error(f"❌ No columns successfully processed for {table_name}")
            return None
            
        return column_metadata
    
    def _analyze_relationships(self, table_data: Dict[str, Any], table_name: str) -> List[str]:
        """Analyze potential relationships for this table."""
        relationships = []
        
        # Get columns from either direct location or properties wrapper
        columns = table_data.get("columns", [])
        if not columns and "properties" in table_data:
            columns = table_data.get("properties", {}).get("columns", [])
        
        for column in columns:
            col_name = column.get("name", "").lower()
            
            # Foreign key detection
            if col_name.endswith('_id') and col_name != 'id' and col_name != f"{table_name.lower()}_id":
                base_name = col_name.replace('_id', '')
                relationships.append(f"References {base_name} table via {col_name}")
        
        return relationships
    
    def _generate_enhanced_column_description(self, col_name: str, col_data: Dict[str, Any], domain_info: Dict[str, Any]) -> str:
        """Generate enhanced description with business context."""
        
        col_lower = col_name.lower()
        business_domain = domain_info["business_domain"]
        
        # Financial domain specific descriptions
        if business_domain == "tranche_management":
            if col_lower in ['pricing', 'yield', 'spread']:
                return f"Bond {col_name.lower()} - key financial metric for tranche valuation"
            elif col_lower in ['maturity', 'maturity_date']:
                return f"Bond maturity information - when the tranche reaches full term"
            elif col_lower in ['rating', 'credit_rating']:
                return f"Credit rating assessment for this tranche"
        
        elif business_domain == "order_management":
            if col_lower in ['ioi', 'indication_of_interest']:
                return f"Indication of Interest - total amount investor wants to purchase"
            elif col_lower in ['allocation', 'final_allocation']:
                return f"Final allocation - actual amount assigned by syndicate banks"
            elif col_lower in ['limit_amount', 'conditional_amount']:
                return f"Investment amount with specific conditions or thresholds"
        
        # Standard pattern matching
        if col_lower.endswith('_id'):
            if col_lower == 'id':
                return f"Primary key identifier for {domain_info['entity_type']} records"
            else:
                base_name = col_lower.replace('_id', '').replace('_', ' ')
                return f"Foreign key reference to {base_name} entity"
        
        elif col_lower.endswith('_date') or col_lower.endswith('_time'):
            event = col_lower.replace('_date', '').replace('_time', '').replace('_', ' ')
            return f"Timestamp when {event} occurred in the {business_domain.replace('_', ' ')} process"
        
        elif col_lower.endswith('_status'):
            entity = col_lower.replace('_status', '').replace('_', ' ')
            return f"Current status of the {entity} in the {business_domain.replace('_', ' ')} lifecycle"
        
        elif col_lower.endswith('_name'):
            return f"Name identifier for the {col_lower.replace('_name', '').replace('_', ' ')}"
        
        elif col_lower in ['created_at', 'updated_at', 'modified_at']:
            return f"Audit timestamp when record was {col_lower.replace('_at', '').replace('_', ' ')}"
        
        elif col_lower in ['created_by', 'updated_by', 'modified_by']:
            return f"User who {col_lower.replace('_by', '').replace('_', ' ')} the record"
        
        else:
            return f"{col_name.replace('_', ' ').title()} - {domain_info['description'].split(' - ')[0].lower()} attribute"
    
    def _determine_business_significance(self, col_name: str, domain_info: Dict[str, Any]) -> str:
        """Determine the business significance of a column."""
        
        col_lower = col_name.lower()
        
        # Key business identifiers
        if col_lower == 'id':
            return "primary_key"
        elif col_lower.endswith('_id'):
            return "foreign_key"
        
        # Financial metrics
        elif col_lower in ['pricing', 'yield', 'spread', 'amount', 'value', 'price']:
            return "financial_metric"
        
        # Status and lifecycle
        elif col_lower.endswith('_status') or col_lower == 'status':
            return "lifecycle_status"
        
        # Dates and timing
        elif col_lower.endswith('_date') or col_lower.endswith('_time'):
            return "temporal_marker"
        
        # Names and identifiers
        elif col_lower.endswith('_name') or col_lower == 'name':
            return "business_identifier"
        
        # Audit fields
        elif col_lower in ['created_at', 'updated_at', 'created_by', 'updated_by']:
            return "audit_field"
        
        else:
            return "business_attribute"
    
    def _generate_relationship_hint(self, col_name: str, domain_info: Dict[str, Any]) -> str:
        """Generate relationship hint based on column name and business domain."""
        
        base_name = col_name.lower().replace('_id', '')
        business_domain = domain_info["business_domain"]
        
        # Domain-specific relationship hints
        if business_domain == "tranche_management":
            if base_name == "deal":
                return "References deal table - each tranche belongs to one deal"
            elif base_name == "issuer":
                return "References issuer table - ultimate parent of the deal"
        
        elif business_domain == "order_management":
            if base_name == "tranche":
                return "References tranche table - orders are placed for specific tranches"
            elif base_name == "investor":
                return "References investor table - who is placing the order"
        
        elif business_domain == "syndicate_operations":
            if base_name == "tranche":
                return "References tranche table - syndicate participates in tranche distribution"
            elif base_name == "bank":
                return "References bank table - syndicate member institution"
        
        return f"References {base_name} table - foreign key relationship"
    
    def _add_financial_context(self, col_name: str, domain_info: Dict[str, Any]) -> str:
        """Add financial domain specific context."""
        
        col_lower = col_name.lower()
        
        if col_lower in ['ioi', 'indication_of_interest']:
            return "IOI represents total investment interest before final allocation decisions"
        elif col_lower in ['allocation', 'final_allocation']:
            return "Final allocation often differs from IOI based on syndicate distribution strategy"
        elif col_lower in ['reoffer', 'reoffer_amount']:
            return "Reoffer amount is unconditional investment regardless of final terms"
        elif col_lower in ['conditional', 'conditional_amount']:
            return "Conditional amount depends on meeting specific price, yield, or spread thresholds"
        elif col_lower in ['pricing', 'yield', 'spread']:
            return "Key financial terms that determine bond attractiveness to investors"
        
        return "Financial instrument context applies"
    
    def create_business_context_report(self, metadata: Dict[str, Any]) -> str:
        """Create a business context report for the REPORTS document type."""
        
        # Analyze the schema and create business context
        core_tables = []
        supporting_tables = []
        
        for model in metadata.get("models", []):
            table_name = model.get("table_name", "")
            domain_info = self.classify_business_domain(table_name)
            
            if domain_info["core_table"]:
                core_tables.append({
                    "name": table_name,
                    "domain": domain_info["business_domain"],
                    "entity": domain_info["entity_type"],
                    "description": domain_info["description"]
                })
            else:
                supporting_tables.append({
                    "name": table_name,
                    "domain": domain_info["business_domain"],
                    "entity": domain_info["entity_type"],
                    "description": domain_info["description"]
                })
        
        # Create business context report
        report = f"""# Fixed Income Syndication Business Context

## Overview
This database supports the fixed income syndication platform for bond issuances and trading operations.

## Core Business Entities Hierarchy

### 1. ISSUER → DEAL → TRANCHE → ORDERS → ORDER LIMITS

**ISSUER**: Companies seeking capital through bond issuances
- Top-level entity initiating fundraising
- One issuer can have multiple deals

**DEAL**: Fundraising initiatives created by JPMorgan for issuers  
- Container for all bond issuances for specific capital raise
- Each deal belongs to one issuer, contains multiple tranches

**TRANCHE**: Individual bond issuances with distinct terms
- Core object with pricing, maturity, ratings information
- Multiple tranches per deal allow different risk/return profiles
- Each tranche belongs to one deal

**SYNDICATE BANK**: Financial institutions participating in distribution
- Multiple banks per tranche with different roles (lead, co-manager)
- Handle distribution and allocation decisions

**ORDER**: Investment requests from institutional investors
- Contains IOI (Indication of Interest) and Final Allocation
- Multiple orders per tranche from different investors

**ORDER LIMIT**: Investment components within orders
- Reoffer Order Limit: Unconditional investment amount
- Conditional Order Limit: Investment with price/yield thresholds

## Core Tables ({len(core_tables)} tables)
"""
        
        for table in core_tables:
            report += f"- **{table['name']}**: {table['description']} [{table['domain']}]\n"
        
        report += f"\n## Supporting Tables ({len(supporting_tables)} tables)\n"
        
        for table in supporting_tables:
            report += f"- **{table['name']}**: {table['description']} [{table['domain']}]\n"
        
        report += """
## Common Query Patterns

### Core Entity Relationships
```sql
-- Deal to Tranche hierarchy
SELECT d.deal_name, t.tranche_name, t.pricing
FROM deals d
JOIN tranches t ON d.id = t.deal_id

-- Order allocation analysis  
SELECT t.tranche_name, o.ioi_amount, o.final_allocation
FROM tranches t
JOIN orders o ON t.id = o.tranche_id

-- Syndicate participation
SELECT t.tranche_name, sb.bank_name, sb.role
FROM tranches t
JOIN syndicate_banks sb ON t.id = sb.tranche_id
```

### Status and Lookup Joins
Status fields typically use lookup tables for human-readable values:
```sql
-- Join with status lookup
SELECT t.*, tsl.status_name
FROM tranches t
JOIN tranche_status_lookups tsl ON t.status_id = tsl.id
```

### Date Handling
Timestamps ending in T00:00:00 represent date-only values:
```sql
-- Date comparisons
WHERE TRUNC(trade_date) = DATE '2023-01-15'
```

## Business Rules
1. **Entity Hierarchy**: Issuer → Deal → Tranche → Order → Order Limit
2. **Status Lookups**: Most status fields reference lookup tables
3. **Financial Metrics**: Pricing, yield, spread are key tranche valuation metrics
4. **Allocation Logic**: Final allocation often differs from IOI based on distribution strategy
5. **Date Handling**: Use TRUNC() for date-only timestamp comparisons
"""
        
        return report
    
    def restructure_all(self):
        """Main method to restructure metadata for 4-tier system (DDL, COLUMN_DETAILS, LOOKUP_METADATA, REPORTS)."""
        
        logger.info("Starting 4-tier metadata restructuring...")
        
        # Load existing metadata
        try:
            metadata = self.load_existing_metadata()
            models_count = len(metadata.get("models", []))
            views_count = len(metadata.get("views", []))
            relationships_count = len(metadata.get("relationships", []))
            logger.info(f"Successfully loaded metadata: {models_count} models, {views_count} views, {relationships_count} relationships")
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            raise
        
        # 1. Create enhanced column metadata files (COLUMN_DETAILS document type)
        logger.info("Creating enhanced column metadata files...")
        
        models_processed = 0
        models_failed = 0
        
        # Process tables
        total_models = len(metadata.get("models", []))
        logger.info(f"Starting to process {total_models} models...")
        
        for i, model in enumerate(metadata.get("models", [])):
            # Handle different naming conventions for table name
            table_name = model.get("table_name") or model.get("name")
            if not table_name:
                logger.warning(f"Model {i} missing table_name/name. Keys: {list(model.keys())}")
                models_failed += 1
                continue
                
            try:
                logger.info(f"📋 Processing model {i+1}/{total_models}: {table_name}")
                
                # Debug the model structure
                logger.debug(f"Model keys for {table_name}: {list(model.keys())}")
                if "columns" in model:
                    logger.debug(f"Columns count for {table_name}: {len(model['columns'])}")
                else:
                    logger.error(f"❌ No 'columns' key found in model {table_name}")
                    models_failed += 1
                    continue
                
                # Add root-level catalog and schema to model data
                model_with_context = model.copy()
                if "catalog" not in model_with_context:
                    model_with_context["catalog"] = metadata.get("catalog", "unknown")
                if "schema" not in model_with_context:
                    model_with_context["schema"] = metadata.get("schema", "unknown")
                
                col_metadata = self.create_column_metadata_file(model_with_context, table_name)
                
                if not col_metadata:
                    logger.error(f"❌ Failed to create column metadata for {table_name} - returned None")
                    models_failed += 1
                    continue
                
                # Validate metadata before writing
                if not isinstance(col_metadata, dict) or not col_metadata.get("columns"):
                    logger.error(f"❌ Invalid metadata structure for {table_name}")
                    models_failed += 1
                    continue
                
                output_file = self.output_dir / f"{table_name.lower()}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(col_metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"✅ Created column metadata: {output_file}")
                models_processed += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to process model {table_name}: {e}", exc_info=True)
                models_failed += 1
                continue
        
        logger.info(f"📊 Models processing complete: {models_processed} succeeded, {models_failed} failed")
        
        if models_failed > 0:
            logger.warning(f"⚠️ {models_failed} models failed processing")
        if models_processed == 0:
            logger.error(f"❌ No models were successfully processed!")
        
        views_processed = 0
        views_failed = 0
        
        # Process views
        total_views = len(metadata.get("views", []))
        logger.info(f"Starting to process {total_views} views...")
        
        for i, view in enumerate(metadata.get("views", [])):
            view_name = view.get("view_name")
            if not view_name:
                logger.warning(f"View {i} missing view_name. Keys: {list(view.keys())}")
                views_failed += 1
                continue
                
            try:
                logger.info(f"👁️ Processing view {i+1}/{total_views}: {view_name}")
                
                # Debug the view structure
                logger.debug(f"View keys for {view_name}: {list(view.keys())}")
                if "columns" in view:
                    logger.debug(f"Columns count for {view_name}: {len(view['columns'])}")
                else:
                    logger.error(f"❌ No 'columns' key found in view {view_name}")
                    views_failed += 1
                    continue
                
                col_metadata = self.create_column_metadata_file(view, view_name)
                
                if not col_metadata:
                    logger.error(f"❌ Failed to create column metadata for {view_name} - returned None")
                    views_failed += 1
                    continue
                
                # Validate metadata before writing
                if not isinstance(col_metadata, dict) or not col_metadata.get("columns"):
                    logger.error(f"❌ Invalid metadata structure for {view_name}")
                    views_failed += 1
                    continue
                
                output_file = self.output_dir / f"{view_name.lower()}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(col_metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"✅ Created view metadata: {output_file}")
                views_processed += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to process view {view_name}: {e}", exc_info=True)
                views_failed += 1
                continue
        
        logger.info(f"📊 Views processing complete: {views_processed} succeeded, {views_failed} failed")
        
        if views_failed > 0:
            logger.warning(f"⚠️ {views_failed} views failed processing")
        if views_processed == 0 and total_views > 0:
            logger.error(f"❌ No views were successfully processed!")
        
        # 2. Create business context report (REPORTS document type)
        logger.info("Creating business context report...")
        business_report = self.create_business_context_report(metadata)
        reports_dir = self.output_dir.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / "business_context_and_query_patterns.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(business_report)
        logger.info(f"Created business context report: {report_file}")
        
        # 3. Create summary statistics
        logger.info("Creating summary statistics...")
        core_tables = sum(1 for model in metadata.get("models", []) if self.classify_business_domain(model.get("table_name", ""))["core_table"])
        supporting_tables = len(metadata.get("models", [])) - core_tables
        total_views = len(metadata.get("views", []))
        
        summary = {
            "restructuring_completed": True,
            "timestamp": "2024-01-01",  # Will be updated when script runs
            "statistics": {
                "total_tables": len(metadata.get("models", [])),
                "core_tables": core_tables,
                "supporting_tables": supporting_tables,
                "total_views": total_views,
                "column_detail_files": core_tables + supporting_tables + total_views
            },
            "document_types_created": [
                "COLUMN_DETAILS - Enhanced column metadata with business domain classification",
                "REPORTS - Business context and query patterns documentation"
            ],
            "notes": [
                "DDL files should be created using extract_ddl_statements.py script",
                "LOOKUP_METADATA files already exist in lookups/ directory",
                "System uses 4-tier architecture: DDL + COLUMN_DETAILS + LOOKUP_METADATA + REPORTS"
            ]
        }
        
        summary_file = self.output_dir.parent / "restructuring_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Created restructuring summary: {summary_file}")
        
        logger.info("4-tier metadata restructuring completed successfully!")


def main():
    """Main function to run metadata restructuring."""
    
    # Configuration
    INPUT_FILE = "meta_documents/p1-synd/schema/main_schema_metadata.json"
    OUTPUT_DIR = "meta_documents/p1-synd/columns"
    
    try:
        restructurer = ColumnMetadataRestructurer(INPUT_FILE, OUTPUT_DIR)
        restructurer.restructure_all()
        
        print("\n4-Tier Metadata Restructuring completed!")
        print("Files created in:")
        print(f"  - Column metadata (COLUMN_DETAILS): {OUTPUT_DIR}")
        print(f"  - Business context report (REPORTS): meta_documents/p1-synd/reports/")
        print(f"  - Summary: meta_documents/p1-synd/restructuring_summary.json")
        print("\nNext steps:")
        print("  1. Run extract_ddl_statements.py to create DDL files")
        print("  2. Ensure LOOKUP_METADATA files exist in lookups/ directory")
        print("  3. Sync documents to populate vector store with 4-tier metadata")
        
    except Exception as e:
        logger.error(f"Restructuring failed: {e}")


if __name__ == "__main__":
    print("4-Tier Metadata Restructuring Script")
    print("====================================")
    print("This script restructures your existing main_schema_metadata.json")
    print("into the new 4-tier metadata format with:")
    print("- DDL: Table structures (use extract_ddl_statements.py)")
    print("- COLUMN_DETAILS: Enhanced column metadata with business domains")
    print("- LOOKUP_METADATA: ID-name mappings (existing files)")
    print("- REPORTS: Business context and query patterns")
    print()
    
    # Ask for confirmation  
    response = input("Do you want to proceed? (y/N): ")
    if response.lower() == 'y':
        main()
    else:
        print("Cancelled by user.")