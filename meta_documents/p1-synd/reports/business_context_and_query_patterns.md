# Fixed Income Syndication Business Context

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

## Core Tables (2 tables)
- **orders**: Order management - investment requests from institutional investors [order_management]
- **order_items**: Order management - investment requests from institutional investors [order_management]

## Supporting Tables (1 tables)
- **users**: System administration - user accounts and permissions [system_administration]

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
