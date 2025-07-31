"""
Hierarchical Context Service for efficient text-to-SQL metadata retrieval.
Implements multi-tiered context building with progressive enhancement.
"""

import json
import structlog
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .vector_service import LlamaIndexVectorService
from .llm_service import LLMService
from ..models.simple_models import DocumentType

logger = structlog.get_logger(__name__)


@dataclass
class ContextTier:
    """Represents a tier of context information."""
    name: str
    tokens_estimate: int
    content: str
    sources: List[str]
    confidence: float = 1.0


@dataclass
class HierarchicalContext:
    """Complete hierarchical context for SQL generation."""
    query: str
    selected_tables: List[str]
    tiers: List[ContextTier]
    total_tokens: int
    retrieval_time_ms: float
    
    def get_combined_context(self) -> str:
        """Combine all tiers into a single context string."""
        context_parts = []
        for tier in self.tiers:
            context_parts.append(f"## {tier.name.upper()}\n{tier.content}")
        return "\n\n".join(context_parts)


class HierarchicalContextService:
    """Service for building hierarchical context using multiple metadata tiers."""
    
    def __init__(
        self,
        vector_service: LlamaIndexVectorService,
        llm_service: LLMService,
        max_context_tokens: int = 15000
    ):
        """
        Initialize hierarchical context service.
        
        Args:
            vector_service: Vector search service
            llm_service: LLM service for reasoning
            max_context_tokens: Maximum tokens to use for context
        """
        self.vector_service = vector_service
        self.llm_service = llm_service
        self.max_context_tokens = max_context_tokens
    
    def build_context(
        self,
        query: str,
        include_advanced_rules: bool = None,
        debug: bool = False
    ) -> HierarchicalContext:
        """
        Build hierarchical context using progressive enhancement.
        
        Args:
            query: User's natural language query
            include_advanced_rules: Force include/exclude advanced rules
            debug: Enable debug logging
            
        Returns:
            HierarchicalContext with all relevant metadata tiers
        """
        import time
        start_time = time.time()
        
        logger.info("Building hierarchical context", query=query)
        
        # Phase 1: Fast Table Selection
        selected_tables = self._select_relevant_tables(query, debug)
        
        # Phase 2: Build Progressive Context Tiers
        tiers = []
        remaining_tokens = self.max_context_tokens
        
        # Tier 1: Core DDL (always included)
        ddl_tier = self._build_ddl_tier(selected_tables, remaining_tokens)
        if ddl_tier:
            tiers.append(ddl_tier)
            remaining_tokens -= ddl_tier.tokens_estimate
        
        # Tier 2: Lookup Metadata (if query involves lookups)
        if self._query_needs_lookups(query):
            lookup_tier = self._build_lookup_tier(query, remaining_tokens)
            if lookup_tier:
                tiers.append(lookup_tier)
                remaining_tokens -= lookup_tier.tokens_estimate
        
        # Tier 3: Business Rules (conditional)
        if include_advanced_rules or (include_advanced_rules is None and self._query_needs_rules(query)):
            rules_tier = self._build_rules_tier(query, remaining_tokens)
            if rules_tier:
                tiers.append(rules_tier)
                remaining_tokens -= rules_tier.tokens_estimate
        
        # Tier 4: Column Details (only for complex queries with remaining tokens)
        if remaining_tokens > 2000 and self._query_needs_column_details(query, selected_tables):
            column_tier = self._build_column_details_tier(selected_tables, remaining_tokens)
            if column_tier:
                tiers.append(column_tier)
                remaining_tokens -= column_tier.tokens_estimate
        
        retrieval_time = (time.time() - start_time) * 1000
        total_tokens = sum(tier.tokens_estimate for tier in tiers)
        
        context = HierarchicalContext(
            query=query,
            selected_tables=selected_tables,
            tiers=tiers,
            total_tokens=total_tokens,
            retrieval_time_ms=retrieval_time
        )
        
        logger.info(
            "Hierarchical context built",
            selected_tables=len(selected_tables),
            tiers_count=len(tiers),
            total_tokens=total_tokens,
            retrieval_time_ms=retrieval_time
        )
        
        return context
    
    def _select_relevant_tables(self, query: str, debug: bool = False) -> List[str]:
        """Phase 1: Fast table selection using business descriptions + LLM reasoning."""
        
        # Search business descriptions for relevant domains
        business_results = self.vector_service.search_similar(
            query=query,
            retriever_type="hybrid",
            similarity_top_k=8,
            document_type=DocumentType.BUSINESS_DESC.value
        )
        
        if not business_results:
            logger.warning("No business descriptions found, falling back to DDL search")
            # Fallback: search DDL directly
            ddl_results = self.vector_service.search_similar(
                query=query,
                retriever_type="hybrid", 
                similarity_top_k=15,
                document_type=DocumentType.DDL.value
            )
            return [result.metadata.get("table_name", result.metadata.get("file_path", "").split("/")[-1].replace(".sql", "")) 
                   for result in ddl_results[:10]]
        
        # Extract table names from business descriptions
        candidate_tables = []
        business_context = []
        
        for result in business_results:
            try:
                # Parse business description content
                if hasattr(result, 'text'):
                    content = result.text
                else:
                    content = str(result)
                    
                business_context.append(content)
                
                # Extract table names from content (assuming JSON format)
                if content.strip().startswith('{'):
                    data = json.loads(content)
                    if "tables" in data:
                        candidate_tables.extend(data["tables"].keys())
                        
            except Exception as e:
                logger.warning("Failed to parse business description", error=str(e))
                continue
        
        # Use LLM to select most relevant tables
        if candidate_tables:
            selected_tables = self._llm_select_tables(query, candidate_tables, business_context)
            if debug:
                logger.info("Table selection", 
                           candidates=len(candidate_tables),
                           selected=len(selected_tables),
                           tables=selected_tables)
            return selected_tables
        
        logger.warning("No candidate tables found from business descriptions")
        return []
    
    def _llm_select_tables(self, query: str, candidate_tables: List[str], business_context: List[str]) -> List[str]:
        """Use LLM to intelligently select relevant tables from candidates."""
        
        context = "\n\n".join(business_context)
        
        prompt = f"""Given the user query and available table descriptions, select the 3-7 most relevant tables for generating a SQL query.

User Query: {query}

Available Table Information:
{context}

Available Tables: {', '.join(candidate_tables)}

Instructions:
1. Select tables that are directly relevant to answering the user's question
2. Include primary tables (main entities) and supporting tables (lookups, relationships)
3. Avoid tables that don't contribute to the query
4. Prioritize tables mentioned in the business descriptions
5. Return ONLY a JSON list of selected table names (uppercase)

Example: ["TRADE", "USERS", "TRANCHE_STATUS"]

Selected Tables:"""

        try:
            response = self.llm_service.generate_response(prompt)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                selected = json.loads(json_match.group())
                # Filter to ensure only valid candidates are returned
                valid_selected = [table.upper() for table in selected if table.upper() in [c.upper() for c in candidate_tables]]
                return valid_selected[:8]  # Limit to 8 tables max
                
        except Exception as e:
            logger.warning("LLM table selection failed", error=str(e))
        
        # Fallback: return first 5 candidates
        return candidate_tables[:5]
    
    def _build_ddl_tier(self, selected_tables: List[str], max_tokens: int) -> Optional[ContextTier]:
        """Build DDL tier with core table structures."""
        
        if not selected_tables or max_tokens < 500:
            return None
        
        ddl_content = []
        sources = []
        
        for table_name in selected_tables:
            # Search for DDL file
            ddl_results = self.vector_service.search_similar(
                query=f"table {table_name}",
                retriever_type="keyword",
                similarity_top_k=1,
                document_type=DocumentType.DDL.value
            )
            
            if ddl_results:
                ddl_content.append(ddl_results[0].text)
                sources.append(f"{table_name}.sql")
        
        if not ddl_content:
            return None
        
        content = "\n\n".join(ddl_content)
        tokens_estimate = len(content) // 4  # Rough token estimate
        
        # Truncate if too long
        if tokens_estimate > max_tokens - 100:
            content = content[:max_tokens * 4]
            tokens_estimate = max_tokens - 100
        
        return ContextTier(
            name="Core Schema (DDL)",
            tokens_estimate=tokens_estimate,
            content=content,
            sources=sources,
            confidence=1.0
        )
    
    def _build_lookup_tier(self, query: str, max_tokens: int) -> Optional[ContextTier]:
        """Build lookup metadata tier for status/category mappings."""
        
        if max_tokens < 300:
            return None
        
        lookup_results = self.vector_service.search_similar(
            query=query,
            retriever_type="hybrid",
            similarity_top_k=5,
            document_type=DocumentType.LOOKUP_METADATA.value
        )
        
        if not lookup_results:
            return None
        
        lookup_content = []
        sources = []
        
        for result in lookup_results:
            lookup_content.append(result.text)
            sources.append(result.metadata.get('file_path', 'unknown'))
        
        content = "\n\n".join(lookup_content)
        tokens_estimate = len(content) // 4
        
        if tokens_estimate > max_tokens - 100:
            content = content[:max_tokens * 4]
            tokens_estimate = max_tokens - 100
        
        return ContextTier(
            name="Lookup Metadata",
            tokens_estimate=tokens_estimate,
            content=content,
            sources=sources,
            confidence=0.9
        )
    
    def _build_rules_tier(self, query: str, max_tokens: int) -> Optional[ContextTier]:
        """Build business rules tier for edge cases and special handling."""
        
        if max_tokens < 200:
            return None
        
        rules_results = self.vector_service.search_similar(
            query=query,
            retriever_type="hybrid",
            similarity_top_k=3,
            document_type=DocumentType.BUSINESS_RULES.value
        )
        
        if not rules_results:
            return None
        
        rules_content = []
        sources = []
        
        for result in rules_results:
            rules_content.append(result.text)
            sources.append(result.metadata.get('file_path', 'unknown'))
        
        content = "\n\n".join(rules_content)
        tokens_estimate = len(content) // 4
        
        if tokens_estimate > max_tokens - 100:
            content = content[:max_tokens * 4]
            tokens_estimate = max_tokens - 100
        
        return ContextTier(
            name="Business Rules",
            tokens_estimate=tokens_estimate,
            content=content,
            sources=sources,
            confidence=0.8
        )
    
    def _build_column_details_tier(self, selected_tables: List[str], max_tokens: int) -> Optional[ContextTier]:
        """Build detailed column metadata tier (only for complex queries)."""
        
        if max_tokens < 500:
            return None
        
        column_content = []
        sources = []
        
        for table_name in selected_tables[:3]:  # Limit to top 3 tables
            column_results = self.vector_service.search_similar(
                query=f"columns {table_name}",
                retriever_type="keyword",
                similarity_top_k=1,
                document_type=DocumentType.COLUMN_DETAILS.value
            )
            
            if column_results:
                column_content.append(column_results[0].text)
                sources.append(f"{table_name}_columns.json")
        
        if not column_content:
            return None
        
        content = "\n\n".join(column_content)
        tokens_estimate = len(content) // 4
        
        if tokens_estimate > max_tokens - 100:
            content = content[:max_tokens * 4]
            tokens_estimate = max_tokens - 100
        
        return ContextTier(
            name="Column Details",
            tokens_estimate=tokens_estimate,
            content=content,
            sources=sources,
            confidence=0.7
        )
    
    def _query_needs_lookups(self, query: str) -> bool:
        """Determine if query likely needs lookup metadata."""
        lookup_keywords = [
            "status", "type", "category", "state", "code", "name",
            "active", "pending", "cancelled", "approved", "rejected"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in lookup_keywords)
    
    def _query_needs_rules(self, query: str) -> bool:
        """Determine if query likely needs business rules."""
        rule_keywords = [
            "date", "time", "between", "range", "before", "after",
            "month", "year", "day", "week", "timestamp"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in rule_keywords)
    
    def _query_needs_column_details(self, query: str, selected_tables: List[str]) -> bool:
        """Determine if query needs detailed column information."""
        # Complex queries with many tables or specific column references
        complex_keywords = [
            "join", "relationship", "connect", "link", "reference",
            "detailed", "specific", "exact", "precise"
        ]
        query_lower = query.lower()
        
        # Needs details if: complex keywords OR many tables selected
        return (any(keyword in query_lower for keyword in complex_keywords) or 
                len(selected_tables) > 4)