# 4-Tier System Workflow Examples

## Overview of Business Context Integration

Your business domain knowledge is embedded in multiple layers:

1. **Column Metadata Enhancement** - Business significance classification
2. **Business Context Reports** - Entity relationships and query patterns  
3. **LLM Table Selection** - Uses business context for intelligent reasoning
4. **Financial Term Recognition** - IOI, allocation, reoffer, conditional, etc.

---

## Example 1: Core Business Query
**User Query**: "Show me all active deals with their tranches and current status"

### Step-by-Step Workflow:

#### Phase 1: Table Selection (Hierarchical Context Service)
1. **RAG Search - Reports Tier**
   - Query: "Show me all active deals with their tranches and current status"
   - Searches: `document_type=REPORT` files
   - Returns: Business context about DEAL → TRANCHE hierarchy
   - **Business Context Used**: Your entity relationship explanation

2. **RAG Search - DDL Tier** 
   - Query: "Show me all active deals with their tranches and current status"
   - Searches: `document_type=DDL` files  
   - Returns: 20 candidate DDL files
   - Extracts: Table names like `DEALS`, `TRANCHES`, `DEAL_STATUS`, etc.

3. **LLM Reasoning - Table Selection**
   - **Input**: Business context + candidate tables + user query
   - **Prompt Engineering**: Uses your business hierarchy knowledge
   - **LLM Call**: Select most relevant tables using business context
   - **Output**: `["DEALS", "TRANCHES", "DEAL_STATUS_LOOKUPS"]`

#### Phase 2: Context Building (Progressive Enhancement)
4. **Tier 1 - DDL Content**
   - **RAG Search**: Get DDL for selected tables
   - **No LLM**: Direct retrieval of CREATE TABLE statements
   - **Result**: Core schema for DEALS, TRANCHES, DEAL_STATUS_LOOKUPS

5. **Tier 2 - Business Context (Reports)**
   - **RAG Search**: Query-relevant sections from business reports
   - **Filtering**: Sections about deal status, entity relationships
   - **Business Knowledge**: Your ISSUER→DEAL→TRANCHE hierarchy used
   - **Result**: Entity relationship patterns and query examples

6. **Tier 3 - Column Details** (if tokens allow)
   - **RAG Search**: Enhanced column metadata for core tables
   - **Business Classification**: Focuses on core_table=True entities
   - **Financial Context**: Your IOI/allocation concepts applied
   - **Result**: Key columns with business significance

7. **Tier 4 - Lookups** (if needed)
   - **RAG Search**: Status lookup mappings
   - **Result**: DEAL_STATUS_ID → human-readable mappings

#### Phase 3: SQL Generation
8. **LLM SQL Generation**
   - **Input**: Combined context from all tiers + user query
   - **Business Context**: Uses your entity relationships
   - **Output**: SQL with proper joins and status lookups

---

## Example 2: Financial Metrics Query  
**User Query**: "What are the pricing and yield details for all tranches in announced deals?"

### Step-by-Step Workflow:

#### Phase 1: Table Selection
1. **RAG Search - Reports**
   - Finds sections about tranche pricing, financial metrics
   - **Business Context**: Your pricing/yield/spread explanation used

2. **RAG Search - DDL**
   - Candidate tables: `TRANCHES`, `DEALS`, `PRICING_DATA`, etc.

3. **LLM Table Selection**
   - **Financial Context**: Recognizes "pricing", "yield" as financial metrics
   - **Business Logic**: Knows tranches belong to deals (your hierarchy)
   - **Selected**: `["TRANCHES", "DEALS", "TRANCHE_STATUS_LOOKUPS"]`

#### Phase 2: Context Building
4. **DDL Tier**: Core table structures
5. **Reports Tier**: 
   - **Business Context**: Your tranche financial metrics explanation
   - Query patterns for pricing/yield analysis
6. **Column Details Tier**:
   - **Financial Classification**: Columns marked as `financial_metric`
   - **Business Significance**: pricing, yield, spread prioritized
   - **Your Context**: "Key financial terms that determine bond attractiveness"

---

## Example 3: Report/Documentation Query
**User Query**: "How do order limits work? What's the difference between reoffer and conditional?"

### Step-by-Step Workflow:

#### Phase 1: Query Type Detection
1. **Query Analysis**: Not a SQL generation request - documentation query
2. **Route Decision**: Should use Reports tier primarily

#### Phase 2: Knowledge Retrieval  
3. **RAG Search - Reports Tier**
   - Query: "order limits reoffer conditional"
   - **Finds Your Business Context**:
     - "Reoffer Order Limit: Unconditional investment amount"
     - "Conditional Order Limit: Investment with price/yield thresholds"
   - **Business Knowledge**: Your order limit explanation directly used

4. **Optional Column Details**
   - If more detail needed, searches column metadata
   - **Financial Context**: Your reoffer/conditional definitions applied

#### Phase 3: Response Generation
5. **LLM Response**
   - **Input**: Your business explanations from reports
   - **Output**: Explanation using your domain knowledge
   - **No SQL**: Documentation response, not query generation

---

## Example 4: Complex Multi-Entity Query
**User Query**: "Show investor allocation details for the top 5 tranches by total orders"

### Step-by-Step Workflow:

#### Phase 1: Complex Table Selection
1. **RAG - Reports**: Multi-entity relationship context
   - **Your Hierarchy**: INVESTORS → ORDERS → TRANCHES
   - **Allocation Context**: Your IOI vs final allocation explanation

2. **LLM Reasoning**: 
   - **Business Logic**: Understands investor→order→tranche chain
   - **Financial Context**: "allocation" triggers investor management domain
   - **Selected Tables**: `["INVESTORS", "ORDERS", "TRANCHES", "ORDER_LIMITS"]`

#### Phase 2: Enhanced Context Building
3. **All 4 Tiers Activated** (complex query):
   - **DDL**: All entity structures
   - **Reports**: Your allocation process explanation
   - **Column Details**: IOI vs final_allocation columns highlighted
   - **Lookups**: Investor status, order status lookups

#### Phase 3: Advanced SQL Generation
4. **LLM SQL Generation**:
   - **Business Context**: Uses your allocation concepts
   - **Financial Logic**: Understands IOI vs final allocation difference
   - **Complex Joins**: Based on your entity relationship hierarchy

---

## Key System Characteristics

### 🎯 Adaptive Flow
- **Query Type Detection**: SQL vs documentation automatically routed
- **Tier Activation**: Complex queries get more tiers
- **Token Management**: Progressive enhancement based on available tokens

### 🧠 Business Intelligence Usage
- **Entity Classification**: Your core vs supporting table logic
- **Financial Context**: Your IOI/allocation/reoffer concepts embedded
- **Relationship Understanding**: Your ISSUER→DEAL→TRANCHE hierarchy used
- **Domain Terminology**: Your business terms recognized and prioritized

### ⚡ Performance Optimizations
- **Smart Filtering**: Business-significant columns prioritized
- **Context Relevance**: Only query-relevant business context included
- **Progressive Loading**: Start with core DDL, add detail as needed

### 🔄 RAG vs LLM Usage Points

**RAG Used For**:
- Finding relevant DDL files
- Retrieving business context sections  
- Getting column metadata
- Lookup table mappings

**LLM Used For**:
- Intelligent table selection (with business context)
- Final SQL generation
- Documentation responses
- Query complexity analysis

**Prompt Engineering**:
- Table selection uses your business hierarchy
- Column significance uses your financial context
- Query patterns include your relationship explanations

The system truly leverages your business domain expertise throughout the entire workflow, from initial table selection through final SQL generation.