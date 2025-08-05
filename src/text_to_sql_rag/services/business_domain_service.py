"""
Business Domain Identification Service for Stage 1 of Cascading RAG.
Identifies relevant business domains from user queries and enhances queries with domain terminology.
"""

import structlog
from typing import Dict, Any, List, Optional
import json

from .vector_service import LlamaIndexVectorService
from .llm_service import LLMService
from ..models.simple_models import DocumentType, DomainContext, BusinessDomainMetadata

logger = structlog.get_logger(__name__)


class BusinessDomainService:
    """Service for business domain identification and query enhancement."""
    
    def __init__(
        self,
        vector_service: LlamaIndexVectorService,
        llm_service: LLMService
    ):
        """
        Initialize business domain service.
        
        Args:
            vector_service: Vector search service for domain retrieval
            llm_service: LLM service for domain analysis and query enhancement
        """
        self.vector_service = vector_service
        self.llm_service = llm_service
        
        # Business domain hierarchy based on user's definitions
        self.domain_hierarchy = {
            "ISSUER": {
                "description": "Companies seeking capital through bond issuances",
                "children": ["DEAL"],
                "key_concepts": ["issuer", "company", "capital", "bond", "issuance", "fundraising"]
            },
            "DEAL": {
                "description": "Fundraising initiatives created by JPMorgan for issuers",
                "parent": "ISSUER",
                "children": ["TRANCHE"],
                "key_concepts": ["deal", "fundraising", "initiative", "capital_raise", "bond_issuance"]
            },
            "TRANCHE": {
                "description": "Individual bond issuances with distinct terms",
                "parent": "DEAL", 
                "children": ["ORDER", "SYNDICATE"],
                "key_concepts": ["tranche", "bond", "pricing", "maturity", "ratings", "yield", "spread", "financial_metrics"]
            },
            "SYNDICATE": {
                "description": "Financial institutions participating in distribution",
                "parent": "TRANCHE",
                "key_concepts": ["syndicate", "bank", "distribution", "lead_manager", "co_manager", "allocation"]
            },
            "ORDER": {
                "description": "Investment requests from institutional investors",
                "parent": "TRANCHE",
                "children": ["ORDER_LIMIT"],
                "key_concepts": ["order", "ioi", "indication_of_interest", "final_allocation", "investor", "investment"]
            },
            "ORDER_LIMIT": {
                "description": "Bond order amount within orders",
                "parent": "ORDER",
                "key_concepts": ["order_limit", "reoffer", "conditional", "investment_amount", "threshold"]
            },
            "INVESTOR": {
                "description": "Primary market investor entity that invests in deals",
                "key_concepts": ["investor", "institutional", "investment", "portfolio", "allocation"]
            },
            "TRADES": {
                "description": "Final record of actual trade execution",
                "key_concepts": ["trade", "execution", "trade_date", "price", "yield", "dealer", "settlement"]
            }
        }
    
    def identify_business_domains(self, user_query: str, debug: bool = False) -> DomainContext:
        """
        Stage 1: Identify relevant business domains and enhance query.
        
        Args:
            user_query: Original user query
            debug: Enable debug logging
            
        Returns:
            DomainContext with identified domains and enhanced query
        """
        logger.info("Starting business domain identification", query=user_query)
        
        # Step 1: Search business domain definitions using RAG
        domain_results = self._search_business_domains(user_query)
        
        # Step 2: Use LLM to analyze query and identify relevant domains
        identified_domains = self._llm_identify_domains(user_query, domain_results)
        
        # Step 3: Enhance query with domain-specific terminology
        enhanced_query = self._enhance_query_with_domain_terms(user_query, identified_domains)
        
        # Step 4: Extract business context for identified domains
        business_context = self._extract_business_context(identified_domains)
        
        # Step 5: Get domain relationships for hierarchical context
        domain_relationships = self._get_domain_relationships(identified_domains)
        
        # Calculate confidence based on domain matching
        confidence = self._calculate_domain_confidence(user_query, identified_domains)
        
        context = DomainContext(
            identified_domains=identified_domains,
            enhanced_query=enhanced_query,
            business_context=business_context,
            confidence=confidence,
            domain_relationships=domain_relationships
        )
        
        if debug:
            logger.info(
                "Domain identification complete",
                domains=identified_domains,
                enhanced_query=enhanced_query,
                confidence=confidence
            )
        
        return context
    
    def _search_business_domains(self, query: str) -> List[Dict[str, Any]]:
        """Search for relevant business domain definitions using vector search."""
        try:
            # Search business domain documents
            results = self.vector_service.search_similar(
                query=query,
                retriever_type="hybrid",
                similarity_top_k=5,
                document_type=DocumentType.BUSINESS_DOMAIN.value
            )
            
            # Convert results to consistent format
            domain_results = []
            for result in results:
                if isinstance(result, dict):
                    domain_results.append(result)
                else:
                    # Handle LlamaIndex node format
                    domain_results.append({
                        "content": getattr(result, 'text', str(result)),
                        "metadata": getattr(result, 'metadata', {})
                    })
            
            return domain_results
            
        except Exception as e:
            logger.warning("Failed to search business domains, using fallback", error=str(e))
            return []
    
    def _llm_identify_domains(self, user_query: str, domain_results: List[Dict[str, Any]]) -> List[str]:
        """Use LLM to identify relevant business domains from query and search results."""
        
        # Prepare domain context from search results
        domain_context = ""
        for result in domain_results:
            content = result.get("content", "")
            domain_context += content + "\n\n"
        
        # If no RAG results, use built-in domain hierarchy
        if not domain_context.strip():
            domain_context = self._format_domain_hierarchy_for_llm()
        
        prompt = f"""Analyze the user query and identify which business domains are most relevant.

User Query: {user_query}

Available Business Domains:
{domain_context}

Financial Services Context:
This is a fixed income syndication platform for bond issuances. The main business entities are:
- ISSUER: Companies issuing bonds
- DEAL: Fundraising initiatives 
- TRANCHE: Individual bond issuances
- SYNDICATE: Banks handling distribution
- ORDER: Investment requests from investors
- INVESTOR: Institutional investors
- TRADES: Final trade execution records

Instructions:
1. Identify which business domains are most relevant to the user's query
2. Consider the entity hierarchy: ISSUER → DEAL → TRANCHE → ORDER/SYNDICATE → TRADES
3. Include parent/child domains if the query spans multiple levels
4. Focus on domains that would contain the data needed to answer the query
5. Return ONLY a JSON list of domain names (uppercase)

Example: ["TRANCHE", "DEAL"] for pricing queries
Example: ["ORDER", "INVESTOR"] for allocation queries

Relevant Domains:"""

        try:
            response = self.llm_service.generate_response(prompt)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                domains = json.loads(json_match.group())
                # Validate domains exist in our hierarchy
                valid_domains = [d.upper() for d in domains if d.upper() in self.domain_hierarchy]
                return valid_domains[:4]  # Limit to 4 domains max
                
        except Exception as e:
            logger.warning("LLM domain identification failed", error=str(e))
        
        # Fallback: keyword-based domain detection
        return self._keyword_based_domain_detection(user_query)
    
    def _keyword_based_domain_detection(self, query: str) -> List[str]:
        """Fallback domain detection using keyword matching."""
        query_lower = query.lower()
        detected_domains = []
        
        for domain, info in self.domain_hierarchy.items():
            key_concepts = info.get("key_concepts", [])
            if any(concept in query_lower for concept in key_concepts):
                detected_domains.append(domain)
        
        # If no matches, try basic financial terms
        if not detected_domains:
            if any(term in query_lower for term in ["price", "pricing", "yield", "bond"]):
                detected_domains.append("TRANCHE")
            elif any(term in query_lower for term in ["order", "allocation", "investor"]):
                detected_domains.append("ORDER")
            elif any(term in query_lower for term in ["deal", "issuance"]):
                detected_domains.append("DEAL")
        
        return detected_domains[:3]  # Limit to 3 domains
    
    def _enhance_query_with_domain_terms(self, original_query: str, domains: List[str]) -> str:
        """Enhance query with domain-specific terminology for better semantic matching."""
        if not domains:
            return original_query
        
        # Collect key concepts from identified domains
        domain_terms = set()
        for domain in domains:
            if domain in self.domain_hierarchy:
                domain_info = self.domain_hierarchy[domain]
                domain_terms.update(domain_info.get("key_concepts", []))
        
        # Add domain context to original query
        enhanced_terms = " ".join(domain_terms)
        enhanced_query = f"{original_query} {enhanced_terms}"
        
        return enhanced_query
    
    def _extract_business_context(self, domains: List[str]) -> str:
        """Extract business context for identified domains."""
        if not domains:
            return "General financial services query"
        
        context_parts = []
        for domain in domains:
            if domain in self.domain_hierarchy:
                domain_info = self.domain_hierarchy[domain]
                context_parts.append(f"{domain}: {domain_info['description']}")
        
        return ". ".join(context_parts)
    
    def _get_domain_relationships(self, domains: List[str]) -> Dict[str, List[str]]:
        """Get hierarchical relationships for identified domains."""
        relationships = {}
        
        for domain in domains:
            if domain in self.domain_hierarchy:
                domain_info = self.domain_hierarchy[domain]
                relationships[domain] = {
                    "parent": domain_info.get("parent"),
                    "children": domain_info.get("children", [])
                }
        
        return relationships
    
    def _calculate_domain_confidence(self, query: str, domains: List[str]) -> float:
        """Calculate confidence in domain identification."""
        if not domains:
            return 0.0
        
        query_lower = query.lower()
        total_matches = 0
        total_concepts = 0
        
        for domain in domains:
            if domain in self.domain_hierarchy:
                key_concepts = self.domain_hierarchy[domain].get("key_concepts", [])
                total_concepts += len(key_concepts)
                for concept in key_concepts:
                    if concept in query_lower:
                        total_matches += 1
        
        if total_concepts == 0:
            return 0.5  # Default confidence if no concepts
        
        confidence = min(total_matches / total_concepts * 2, 1.0)  # Scale to 0-1
        return max(confidence, 0.3)  # Minimum confidence of 0.3
    
    def _format_domain_hierarchy_for_llm(self) -> str:
        """Format domain hierarchy for LLM prompt when RAG search fails."""
        formatted = ""
        for domain, info in self.domain_hierarchy.items():
            formatted += f"{domain}: {info['description']}\n"
            if "key_concepts" in info:
                formatted += f"  Key terms: {', '.join(info['key_concepts'])}\n"
            formatted += "\n"
        return formatted
    
    def get_domain_hierarchy(self) -> Dict[str, Any]:
        """Get the complete business domain hierarchy."""
        return self.domain_hierarchy
    
    def add_custom_domain(self, domain_name: str, domain_info: Dict[str, Any]) -> None:
        """Add a custom business domain to the hierarchy."""
        self.domain_hierarchy[domain_name.upper()] = domain_info
        logger.info("Added custom business domain", domain=domain_name)