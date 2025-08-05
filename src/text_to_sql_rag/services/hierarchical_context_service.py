"""
Business Domain-First Cascading RAG Hierarchical Context Service.
Implements 3-stage cascading query refinement with 5-tier progressive context building.
"""

import json
import structlog
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .vector_service import LlamaIndexVectorService
from .llm_service import LLMService
from .business_domain_service import BusinessDomainService
from .view_selection_service import ViewSelectionService
from .report_pattern_service import ReportPatternService
from ..models.simple_models import (
    DocumentType, DomainContext, ViewContext, ReportContext
)

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
class CascadingContext:
    """Complete cascading context for SQL generation with domain intelligence."""
    query: str
    domain_context: DomainContext
    view_context: ViewContext
    report_context: ReportContext
    tiers: List[ContextTier]
    total_tokens: int
    retrieval_time_ms: float
    
    def get_combined_context(self) -> str:
        """Combine all tiers into a single context string."""
        context_parts = []
        for tier in self.tiers:
            context_parts.append(f"## {tier.name.upper()}\n{tier.content}")
        return "\n\n".join(context_parts)
    
    def get_selected_views(self) -> List[str]:
        """Get all selected views (core + supporting)."""
        return self.view_context.core_views + self.view_context.supporting_views


class HierarchicalContextService:
    """Business Domain-First Cascading RAG Context Service."""
    
    def __init__(
        self,
        vector_service: LlamaIndexVectorService,
        llm_service: LLMService,
        max_context_tokens: int = 15000
    ):
        """
        Initialize cascading hierarchical context service.
        
        Args:
            vector_service: Vector search service
            llm_service: LLM service for reasoning
            max_context_tokens: Maximum tokens to use for context
        """
        self.vector_service = vector_service
        self.llm_service = llm_service
        self.max_context_tokens = max_context_tokens
        
        # Initialize cascading services
        self.domain_service = BusinessDomainService(vector_service, llm_service)
        self.view_service = ViewSelectionService(vector_service, llm_service)
        self.report_service = ReportPatternService(vector_service, llm_service)
    
    def build_context(
        self,
        query: str,
        debug: bool = False
    ) -> CascadingContext:
        """
        Build cascading context using business domain-first 3-stage refinement.
        
        Args:
            query: User's natural language query
            debug: Enable debug logging
            
        Returns:
            CascadingContext with all relevant metadata tiers
        """
        import time
        start_time = time.time()
        
        logger.info("Building business domain-first cascading context", query=query)
        
        # STAGE 1: Business Domain Identification
        domain_context = self.domain_service.identify_business_domains(query, debug)
        
        # STAGE 2: Domain-Filtered View Selection  
        view_context = self.view_service.select_domain_views(domain_context, debug=debug)
        
        # STAGE 3: Report Pattern Extraction
        report_context = self.report_service.extract_report_patterns(
            query, view_context, debug=debug
        )
        
        # PHASE 2: Build 5-Tier Progressive Context
        tiers = []
        remaining_tokens = self.max_context_tokens
        
        # Tier 1: Business Domain Context (always included)
        domain_tier = self._build_domain_tier(domain_context, remaining_tokens)
        if domain_tier:
            tiers.append(domain_tier)
            remaining_tokens -= domain_tier.tokens_estimate
        
        # Tier 2: View DDL (filtered by domains)
        if remaining_tokens > 1000:
            ddl_tier = self._build_view_ddl_tier(view_context, remaining_tokens)
            if ddl_tier:
                tiers.append(ddl_tier)
                remaining_tokens -= ddl_tier.tokens_estimate
        
        # Tier 3: Report Examples & Patterns (targeted)
        if remaining_tokens > 1500:
            report_tier = self._build_report_tier(report_context, remaining_tokens)
            if report_tier:
                tiers.append(report_tier)
                remaining_tokens -= report_tier.tokens_estimate
        
        # Tier 4: Core View Details (if complex query)
        if remaining_tokens > 2000 and self._query_needs_view_details(query, view_context):
            view_tier = self._build_view_details_tier(view_context, remaining_tokens)
            if view_tier:
                tiers.append(view_tier)
                remaining_tokens -= view_tier.tokens_estimate
        
        # Tier 5: Lookup Metadata (if status/reference queries)
        if self._query_needs_lookups(query) and remaining_tokens > 500:
            lookup_tier = self._build_lookup_tier(query, remaining_tokens)
            if lookup_tier:
                tiers.append(lookup_tier)
                remaining_tokens -= lookup_tier.tokens_estimate
        
        retrieval_time = (time.time() - start_time) * 1000
        total_tokens = sum(tier.tokens_estimate for tier in tiers)
        
        context = CascadingContext(
            query=query,
            domain_context=domain_context,
            view_context=view_context,
            report_context=report_context,
            tiers=tiers,
            total_tokens=total_tokens,
            retrieval_time_ms=retrieval_time
        )
        
        logger.info(
            "Cascading context built",
            domains=len(domain_context.identified_domains),
            core_views=len(view_context.core_views),
            supporting_views=len(view_context.supporting_views),
            reports=len(report_context.relevant_reports),
            tiers_count=len(tiers),
            total_tokens=total_tokens,
            retrieval_time_ms=retrieval_time
        )
        
        return context
    
    # New 5-Tier Context Builders
    
    def _build_domain_tier(self, domain_context: DomainContext, max_tokens: int) -> Optional[ContextTier]:
        """Build Tier 1: Business Domain Context."""
        if max_tokens < 500:
            return None
        
        content_parts = []
        
        # Business domain descriptions
        content_parts.append("# Business Domain Context")
        content_parts.append(domain_context.business_context)
        
        # Domain relationships
        if domain_context.domain_relationships:
            content_parts.append("\n## Domain Relationships:")
            for domain, relations in domain_context.domain_relationships.items():
                if relations.get("parent"):
                    content_parts.append(f"- {domain} → child of {relations['parent']}")
                if relations.get("children"):
                    children = ", ".join(relations["children"])
                    content_parts.append(f"- {domain} → parent of {children}")
        
        # Enhanced query terms
        content_parts.append(f"\n## Domain-Enhanced Query: {domain_context.enhanced_query}")
        
        content = "\n".join(content_parts)
        tokens_estimate = len(content) // 4
        
        # Truncate if too long
        if tokens_estimate > max_tokens - 100:
            content = content[:max_tokens * 4]
            tokens_estimate = max_tokens - 100
        
        return ContextTier(
            name="Business Domain Context",
            tokens_estimate=tokens_estimate,
            content=content,
            sources=["business_domains"],
            confidence=domain_context.confidence
        )
    
    def _build_view_ddl_tier(self, view_context: ViewContext, max_tokens: int) -> Optional[ContextTier]:
        """Build Tier 2: View DDL (filtered by domains)."""
        if max_tokens < 500:
            return None
        
        selected_views = view_context.core_views + view_context.supporting_views
        if not selected_views:
            return None
        
        ddl_content = []
        sources = []
        
        for view_name in selected_views:
            # Search for DDL file
            try:
                ddl_results = self.vector_service.search_similar(
                    query=f"view {view_name}",
                    retriever_type="keyword",
                    similarity_top_k=1,
                    document_type=DocumentType.DDL.value
                )
                
                if ddl_results:
                    result = ddl_results[0]
                    content = result.text if hasattr(result, 'text') else str(result)
                    ddl_content.append(f"-- {view_name}\n{content}")
                    sources.append(f"{view_name.lower()}.sql")
            except Exception as e:
                logger.warning(f"Failed to get DDL for {view_name}", error=str(e))
        
        if not ddl_content:
            return None
        
        content = "\n\n".join(ddl_content)
        tokens_estimate = len(content) // 4
        
        # Truncate if too long
        if tokens_estimate > max_tokens - 100:
            content = content[:max_tokens * 4]
            tokens_estimate = max_tokens - 100
        
        return ContextTier(
            name="View DDL Structures",
            tokens_estimate=tokens_estimate,
            content=content,
            sources=sources,
            confidence=0.9
        )
    
    def _build_report_tier(self, report_context: ReportContext, max_tokens: int) -> Optional[ContextTier]:
        """Build Tier 3: Report Examples & Patterns (targeted)."""
        if max_tokens < 500 or not report_context.relevant_reports:
            return None
        
        content_parts = []
        
        # SQL patterns
        if report_context.sql_patterns:
            content_parts.append("# Query Patterns")
            for i, pattern in enumerate(report_context.sql_patterns[:3], 1):
                content_parts.append(f"\n## Pattern {i}:\n```sql\n{pattern}\n```")
        
        # Use case matches
        if report_context.use_case_matches:
            content_parts.append("\n# Similar Use Cases")
            for use_case in report_context.use_case_matches[:3]:
                content_parts.append(f"- {use_case}")
        
        # Query examples
        if report_context.query_examples:
            content_parts.append("\n# Query Examples")
            for example in report_context.query_examples[:2]:
                content_parts.append(f"```sql\n{example}\n```")
        
        if not content_parts:
            return None
        
        content = "\n".join(content_parts)
        tokens_estimate = len(content) // 4
        
        # Truncate if too long
        if tokens_estimate > max_tokens - 100:
            content = content[:max_tokens * 4]
            tokens_estimate = max_tokens - 100
        
        return ContextTier(
            name="Report Patterns & Examples",
            tokens_estimate=tokens_estimate,
            content=content,
            sources=report_context.relevant_reports,
            confidence=report_context.pattern_confidence
        )
    
    def _build_view_details_tier(self, view_context: ViewContext, max_tokens: int) -> Optional[ContextTier]:
        """Build Tier 4: Core View Details (for complex queries)."""
        if max_tokens < 500:
            return None
        
        view_content = []
        sources = []
        
        # Focus on core views first
        for view_name in view_context.core_views[:3]:  # Limit to top 3 core views
            try:
                view_results = self.vector_service.search_similar(
                    query=f"view {view_name} metadata details",
                    retriever_type="hybrid",
                    similarity_top_k=1,
                    document_type=DocumentType.CORE_VIEW.value
                )
                
                if view_results:
                    result = view_results[0]
                    content = self._format_view_metadata(result, view_name)
                    if content:
                        view_content.append(content)
                        sources.append(f"{view_name.lower()}.json")
            except Exception as e:
                logger.warning(f"Failed to get details for {view_name}", error=str(e))
        
        if not view_content:
            return None
        
        content = "\n\n".join(view_content)
        tokens_estimate = len(content) // 4
        
        # Truncate if too long
        if tokens_estimate > max_tokens - 100:
            content = content[:max_tokens * 4]
            tokens_estimate = max_tokens - 100
        
        return ContextTier(
            name="Core View Details",
            tokens_estimate=tokens_estimate,
            content=content,
            sources=sources,
            confidence=0.8
        )
    
    def _format_view_metadata(self, result: Any, view_name: str) -> str:
        """Format view metadata for better readability."""
        try:
            if isinstance(result, dict):
                content = result.get("content", "")
            else:
                content = getattr(result, 'text', str(result))
            
            # Parse JSON content if available
            if content.strip().startswith('{'):
                data = json.loads(content)
                
                formatted = f"View: {view_name}\n"
                formatted += f"Type: {data.get('view_type', 'Unknown')}\n"
                formatted += f"Description: {data.get('description', '')}\n"
                
                if data.get('use_cases'):
                    formatted += f"Use Cases: {', '.join(data['use_cases'][:3])}\n"
                
                if data.get('data_returned'):
                    formatted += f"Data Returned: {data['data_returned'][:200]}...\n"
                
                return formatted
                
        except Exception as e:
            logger.warning("Failed to parse view metadata", error=str(e))
        
        # Fallback: return raw content truncated
        if isinstance(result, dict):
            content = result.get("content", "")
        else:
            content = getattr(result, 'text', str(result))
        
        return content[:500] if len(content) > 500 else content
    
    def _query_needs_view_details(self, query: str, view_context: ViewContext) -> bool:
        """Determine if query needs detailed view information."""
        # Complex queries with many views or specific detail requests
        complex_keywords = [
            "detailed", "specific", "exact", "precise", "columns", "fields",
            "structure", "metadata", "definition"
        ]
        query_lower = query.lower()
        
        # Needs details if: complex keywords OR many views selected OR supporting views present
        return (any(keyword in query_lower for keyword in complex_keywords) or 
                len(view_context.core_views) > 3 or
                len(view_context.supporting_views) > 0)
    
    def _build_lookup_tier(self, query: str, max_tokens: int) -> Optional[ContextTier]:
        """Build Tier 5: Lookup Metadata (if status/reference queries)."""
        if max_tokens < 300:
            return None
        
        try:
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
                if isinstance(result, dict):
                    content = result.get("content", "")
                    metadata = result.get("metadata", {})
                else:
                    content = getattr(result, 'text', str(result))
                    metadata = getattr(result, 'metadata', {})
                
                if content:
                    lookup_content.append(content)
                    source = metadata.get('file_path', 'unknown_lookup')
                    sources.append(source)
            
            if not lookup_content:
                return None
            
            content = "\n\n".join(lookup_content)
            tokens_estimate = len(content) // 4
            
            # Truncate if too long
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
            
        except Exception as e:
            logger.warning("Failed to build lookup tier", error=str(e))
            return None
    
    def _query_needs_lookups(self, query: str) -> bool:
        """Determine if query likely needs lookup metadata."""
        lookup_keywords = [
            "status", "type", "category", "state", "code", "name",
            "active", "pending", "cancelled", "approved", "rejected",
            "lookup", "reference", "mapping"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in lookup_keywords)
    
    # Legacy method for backward compatibility
    def build_hierarchical_context(self, query: str, debug: bool = False) -> CascadingContext:
        """Legacy method name - redirects to build_context.""" 
        return self.build_context(query, debug)