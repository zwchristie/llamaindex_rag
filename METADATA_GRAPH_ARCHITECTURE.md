# Metadata Graph Architecture for Hierarchical RAG Retrieval

## Overview

This document describes the metadata graph architecture that enables sophisticated hierarchical retrieval for text-to-SQL generation. The system uses a multi-step retrieval pattern that combines LLM reasoning, semantic search, and graph-based filtering to provide highly relevant context.

## Architecture Principles

### 1. Graph-Based Metadata Structure
The metadata is organized as a connected graph where:
- **Nodes**: Different metadata types (Business Domains, Views, Reports, Lookups)
- **Edges**: Relationships between metadata objects (domain membership, view dependencies, lookup references)
- **Attributes**: Properties that enable filtering and retrieval (business_domains, lookup_id, related_views)

### 2. Hierarchical Retrieval Pipeline
Retrieval follows a structured pipeline:
1. **Business Domain Identification** → LLM identifies relevant domains
2. **Core View Retrieval** → Get primary tables for identified domains
3. **Supporting View Retrieval** → Get additional related tables
4. **Report Example Retrieval** → Get example queries from reports
5. **Lookup Value Retrieval** → Get lookup values for columns with lookup_id

### 3. Multi-Modal Search Strategy
- **LLM Reasoning**: Domain classification and intent understanding
- **Semantic Search**: Vector similarity for content matching
- **Graph Filtering**: Relationship-based filtering using metadata connections

## Metadata Types and Structure

### Business Domains (`business_domains.json`)
```json
{
  "domain_id": 1,
  "domain_name": "Deal Management",
  "description": "Management of financial deals and transactions",
  "keywords": ["deal", "transaction", "financial"],
  "core_views": ["V_DEAL_OVERVIEW", "V_DEAL_DETAILS"],
  "supporting_views": ["V_DEAL_PARTICIPANTS", "V_DEAL_HISTORY"],
  "related_lookups": [1, 2, 5]
}
```

**Purpose**: Define business areas and their associated metadata objects
**Connections**: Links to views and lookups through IDs and names
**Usage**: First-level filtering based on user query intent

### View Metadata (`views/*.json`)
```json
{
  "view_name": "V_TRANCHE_SYNDICATES",
  "view_type": "CORE",
  "business_domains": [1, 2],
  "columns": [
    {
      "name": "tranche_status_id",
      "type": "number",
      "lookup_id": 1
    }
  ]
}
```

**Purpose**: Define database views/tables with column metadata
**Connections**: 
- `business_domains[]` → Links to business domain IDs
- `columns[].lookup_id` → Links to lookup table IDs
- `related_views[]` → Links to other related views
**Usage**: Primary source of SQL generation context

### Report Metadata (`reports/*.json`)
```json
{
  "report_name": "Deal Pipeline Analysis Report",
  "view_name": "V_DEAL_OVERVIEW",
  "business_domains": [1, 5],
  "related_views": ["V_DEAL_OVERVIEW", "V_TRANCHE_SYNDICATES"],
  "example_sql": "SELECT deal_type, COUNT(*) FROM V_DEAL_OVERVIEW..."
}
```

**Purpose**: Provide example queries and use cases
**Connections**:
- `business_domains[]` → Links to business domain IDs
- `view_name` → Primary view dependency
- `related_views[]` → Additional view dependencies
**Usage**: SQL pattern examples and query templates

### Lookup Metadata (`lookups/*.json`)
```json
{
  "lookup_id": 1,
  "lookup_name": "tranche_statuses",
  "business_domains": [1, 2],
  "table_column_references": [
    "V_TRANCHE_SYNDICATES.tranche_status_id"
  ],
  "values": [{"id": 1, "name": "active", "code": "ACTIVE"}]
}
```

**Purpose**: Define lookup values for foreign key columns
**Connections**:
- `lookup_id` → Referenced by view columns
- `business_domains[]` → Links to business domain IDs
- `table_column_references[]` → Explicit column connections
**Usage**: Provide valid values for WHERE clauses and filtering

## Hierarchical Retrieval Process

### Step 1: Business Domain Identification
**Input**: User query
**Process**: 
1. Send user query to LLM with business domain definitions
2. LLM analyzes query and returns relevant domain IDs
3. Filter subsequent searches to identified domains

**Example**:
```
Query: "Show me syndicate participation for recent deals"
LLM Response: [1, 2] // Deal Management, Syndicate Operations
```

### Step 2: Core View Retrieval
**Input**: Identified business domains + user query
**Process**:
1. Filter views where `business_domains` intersects with identified domains
2. Rank by semantic similarity to user query
3. Select top-k core views (`view_type = "CORE"`)

**Example**:
```
Domains: [1, 2]
Core Views Retrieved: ["V_DEAL_OVERVIEW", "V_TRANCHE_SYNDICATES"]
```

### Step 3: Supporting View Retrieval
**Input**: Core views + business domains
**Process**:
1. For each core view, find related supporting views
2. Filter supporting views by business domain overlap
3. Rank by relevance to core views and user query

**Example**:
```
Core Views: ["V_TRANCHE_SYNDICATES"]
Supporting Views: ["V_SYNDICATE_ALLOCATIONS", "V_PARTICIPANT_ROLES"]
```

### Step 4: Report Example Retrieval (Parallel)
**Input**: Retrieved views + business domains
**Process**:
1. Find reports where `view_name` matches retrieved views
2. Find reports where `related_views` intersects retrieved views
3. Filter by business domain overlap
4. Extract example SQL patterns

### Step 5: Lookup Value Retrieval (Parallel)
**Input**: Retrieved views + their columns
**Process**:
1. Extract all columns with `lookup_id` from retrieved views
2. Fetch corresponding lookup metadata by `lookup_id`
3. Include lookup values in context for SQL generation

**Example**:
```
Column: V_TRANCHE_SYNDICATES.tranche_status_id (lookup_id: 1)
Lookup Retrieved: tranche_statuses with values [ACTIVE, PENDING, SETTLED]
```

## Context Assembly

The final context includes:
1. **Primary Views**: Core tables with full column definitions
2. **Supporting Views**: Additional related tables
3. **Example Patterns**: SQL examples from reports
4. **Lookup Values**: Valid values for lookup columns
5. **Business Context**: Domain descriptions and use cases

## Implementation Architecture

### Service Classes

**BusinessDomainService**
- Load and manage business domain definitions
- LLM-based domain identification from user queries
- Domain filtering and relationship traversal

**HierarchicalRetrievalService**
- Orchestrate multi-step retrieval pipeline
- Coordinate between different retrieval steps
- Assemble final context from all sources

**MetadataGraphService**
- Manage graph relationships between metadata objects
- Efficient traversal of metadata connections
- Caching and optimization of graph operations

### Pipeline Configuration

```python
retrieval_config = {
    "max_core_views": 3,
    "max_supporting_views": 5,
    "max_reports": 2,
    "enable_parallel_lookup": True,
    "domain_confidence_threshold": 0.7
}
```

## Benefits

1. **Precision**: Multi-step filtering reduces irrelevant context
2. **Completeness**: Systematic coverage of related metadata
3. **Efficiency**: Graph-based filtering avoids expensive semantic searches
4. **Scalability**: Hierarchical approach scales with metadata growth
5. **Interpretability**: Clear reasoning chain through retrieval steps

## Example Flow

```
User Query: "What are the active syndicate participants for high-value deals?"

1. Domain Identification: [1: Deal Management, 2: Syndicate Operations]
2. Core Views: [V_DEAL_OVERVIEW, V_TRANCHE_SYNDICATES]
3. Supporting Views: [V_SYNDICATE_ALLOCATIONS]
4. Reports: [Syndicate Performance Analysis Report]
5. Lookups: [tranche_statuses (id:1), syndicate_roles (id:3)]

Final Context:
- V_DEAL_OVERVIEW: deal filtering and high-value identification
- V_TRANCHE_SYNDICATES: syndicate participant information
- V_SYNDICATE_ALLOCATIONS: detailed allocation data
- Example SQL: Role-based grouping patterns
- Lookup Values: ACTIVE status (id:3), participant roles
```

This architecture enables sophisticated, contextually-aware SQL generation while maintaining clear reasoning chains and efficient retrieval performance.