"""
Report Pattern Extraction Service for Stage 3 of Cascading RAG.
Extracts relevant query patterns and examples from report documents based on selected views.
"""

import structlog
from typing import Dict, Any, List, Optional
import json
import re

from .vector_service import LlamaIndexVectorService
from .llm_service import LLMService
from ..models.simple_models import DocumentType, ReportContext, ViewContext

logger = structlog.get_logger(__name__)


class ReportPatternService:
    """Service for extracting query patterns and examples from reports."""
    
    def __init__(
        self,
        vector_service: LlamaIndexVectorService,
        llm_service: LLMService
    ):
        """
        Initialize report pattern service.
        
        Args:
            vector_service: Vector search service for report retrieval
            llm_service: LLM service for pattern extraction and analysis
        """
        self.vector_service = vector_service
        self.llm_service = llm_service
    
    def extract_report_patterns(
        self,
        original_query: str,
        view_context: ViewContext,
        max_reports: int = 5,
        debug: bool = False
    ) -> ReportContext:
        """
        Stage 3: Extract relevant query patterns from reports based on selected views.
        
        Args:
            original_query: Original user query
            view_context: Result from view selection stage
            max_reports: Maximum number of reports to analyze
            debug: Enable debug logging
            
        Returns:
            ReportContext with extracted patterns and examples
        """
        logger.info(
            "Starting report pattern extraction",
            query=original_query,
            core_views=len(view_context.core_views),
            supporting_views=len(view_context.supporting_views)
        )
        
        # Step 1: Generate context-aware query for report search
        context_query = self._generate_context_aware_query(original_query, view_context)
        
        # Step 2: Search for relevant reports
        relevant_reports = self._search_relevant_reports(context_query, max_reports)
        
        # Step 3: Extract SQL patterns from reports
        sql_patterns = self._extract_sql_patterns(relevant_reports, view_context)
        
        # Step 4: Find matching use cases
        use_case_matches = self._find_use_case_matches(original_query, relevant_reports)
        
        # Step 5: Extract query examples
        query_examples = self._extract_query_examples(relevant_reports, view_context)
        
        # Step 6: Calculate pattern confidence
        pattern_confidence = self._calculate_pattern_confidence(
            original_query,
            relevant_reports,
            view_context
        )
        
        context = ReportContext(
            relevant_reports=[r.get("report_name", "unknown") for r in relevant_reports],
            sql_patterns=sql_patterns,
            use_case_matches=use_case_matches,
            query_examples=query_examples,
            pattern_confidence=pattern_confidence
        )
        
        if debug:
            logger.info(
                "Report pattern extraction complete",
                reports_found=len(relevant_reports),
                patterns_extracted=len(sql_patterns),
                examples_found=len(query_examples),
                confidence=pattern_confidence
            )
        
        return context
    
    def _generate_context_aware_query(self, original_query: str, view_context: ViewContext) -> str:
        """Generate a context-aware query based on selected views and business domains."""
        
        # Include view names for better semantic matching
        view_terms = []
        for view in view_context.core_views + view_context.supporting_views:
            # Extract meaningful terms from view names
            view_parts = view.replace("V_", "").replace("_", " ").lower()
            view_terms.append(view_parts)
        
        # Include business domain terms
        domain_terms = " ".join(view_context.business_domains).lower()
        
        # Combine original query with context
        context_query = f"{original_query} {domain_terms} {' '.join(view_terms)}"
        
        return context_query
    
    def _search_relevant_reports(self, context_query: str, max_reports: int) -> List[Dict[str, Any]]:
        """Search for relevant reports using context-aware query."""
        try:
            # Search report documents
            results = self.vector_service.search_similar(
                query=context_query,
                retriever_type="hybrid",
                similarity_top_k=max_reports * 2,  # Get more for filtering
                document_type=DocumentType.REPORT.value
            )
            
            # Convert results to consistent format
            reports = []
            for result in results:
                if isinstance(result, dict):
                    report_data = result
                else:
                    # Handle LlamaIndex node format
                    content = getattr(result, 'text', str(result))
                    metadata = getattr(result, 'metadata', {})
                    
                    # Try to parse content as JSON if it looks like report metadata
                    try:
                        if content.strip().startswith('{'):
                            parsed_content = json.loads(content)
                            report_data = {**parsed_content, "metadata": metadata}
                        else:
                            report_data = {"content": content, "metadata": metadata}
                    except json.JSONDecodeError:
                        report_data = {"content": content, "metadata": metadata}
                
                reports.append(report_data)
                if len(reports) >= max_reports:
                    break
            
            return reports
            
        except Exception as e:
            logger.warning("Report search failed", error=str(e))
            return []
    
    def _extract_sql_patterns(self, reports: List[Dict[str, Any]], view_context: ViewContext) -> List[str]:
        """Extract SQL patterns from reports that use selected views."""
        patterns = []
        selected_views_lower = [v.lower() for v in view_context.core_views + view_context.supporting_views]
        
        for report in reports:
            # Look for SQL in various fields
            sql_fields = ["example_sql", "view_sql", "query", "sql_example"]
            
            for field in sql_fields:
                sql_content = report.get(field, "")
                if sql_content and isinstance(sql_content, str):
                    # Check if SQL uses any of our selected views
                    sql_lower = sql_content.lower()
                    if any(view.lower() in sql_lower for view in selected_views_lower):
                        # Clean and format SQL
                        cleaned_sql = self._clean_sql_pattern(sql_content)
                        if cleaned_sql and cleaned_sql not in patterns:
                            patterns.append(cleaned_sql)
        
        return patterns[:5]  # Limit to 5 patterns
    
    def _clean_sql_pattern(self, sql: str) -> str:
        """Clean and format SQL pattern for better readability."""
        if not sql:
            return ""
        
        # Remove excessive whitespace and newlines
        cleaned = re.sub(r'\s+', ' ', sql.strip())
        
        # Remove comments
        cleaned = re.sub(r'--.*?$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        
        # Basic SQL formatting
        cleaned = cleaned.replace(' FROM ', '\nFROM ')
        cleaned = cleaned.replace(' WHERE ', '\nWHERE ')
        cleaned = cleaned.replace(' JOIN ', '\nJOIN ')
        cleaned = cleaned.replace(' ORDER BY ', '\nORDER BY ')
        cleaned = cleaned.replace(' GROUP BY ', '\nGROUP BY ')
        
        return cleaned.strip()
    
    def _find_use_case_matches(self, original_query: str, reports: List[Dict[str, Any]]) -> List[str]:
        """Find use cases from reports that match the original query intent."""
        matches = []
        query_lower = original_query.lower()
        
        for report in reports:
            use_cases = report.get("use_cases", [])
            if isinstance(use_cases, list):
                for use_case in use_cases:
                    if isinstance(use_case, str):
                        # Simple keyword matching for now
                        use_case_lower = use_case.lower()
                        if self._calculate_text_similarity(query_lower, use_case_lower) > 0.3:
                            if use_case not in matches:
                                matches.append(use_case)
            elif isinstance(use_cases, str):
                # Handle single use case as string
                if self._calculate_text_similarity(query_lower, use_cases.lower()) > 0.3:
                    if use_cases not in matches:
                        matches.append(use_cases)
        
        return matches[:5]  # Limit to 5 matches
    
    def _extract_query_examples(self, reports: List[Dict[str, Any]], view_context: ViewContext) -> List[str]:
        """Extract query examples from reports that are relevant to selected views."""
        examples = []
        
        for report in reports:
            # Look for example queries in various fields
            example_fields = ["example_query", "query_example", "sample_query"]
            
            for field in example_fields:
                example = report.get(field, "")
                if example and isinstance(example, str):
                    # Check if example is relevant to our views
                    if self._is_example_relevant(example, view_context):
                        if example not in examples:
                            examples.append(example)
            
            # Also check if report description contains query patterns
            description = report.get("report_description", "")
            if description and "SELECT" in description.upper():
                # Extract SQL from description
                sql_matches = re.findall(r'SELECT.*?(?:;|$)', description, re.IGNORECASE | re.DOTALL)
                for match in sql_matches:
                    cleaned = match.strip().rstrip(';')
                    if self._is_example_relevant(cleaned, view_context):
                        if cleaned not in examples:
                            examples.append(cleaned)
        
        return examples[:5]  # Limit to 5 examples
    
    def _is_example_relevant(self, example: str, view_context: ViewContext) -> bool:
        """Check if a query example is relevant to selected views."""
        example_lower = example.lower()
        all_views = view_context.core_views + view_context.supporting_views
        
        # Check if example mentions any of our selected views
        for view in all_views:
            if view.lower() in example_lower:
                return True
        
        # Check if example mentions business domains
        for domain in view_context.business_domains:
            if domain.lower() in example_lower:
                return True
        
        return False
    
    def _calculate_pattern_confidence(
        self,
        original_query: str,
        reports: List[Dict[str, Any]],
        view_context: ViewContext
    ) -> float:
        """Calculate confidence in extracted patterns based on relevance."""
        if not reports:
            return 0.0
        
        total_score = 0.0
        max_score = 0.0
        
        for report in reports:
            score = 0.0
            max_report_score = 4.0  # Maximum possible score per report
            
            # Score based on view matches
            if self._report_mentions_views(report, view_context):
                score += 1.0
            
            # Score based on business domain matches
            if self._report_mentions_domains(report, view_context.business_domains):
                score += 1.0
            
            # Score based on use case similarity
            use_cases = report.get("use_cases", [])
            if isinstance(use_cases, list) and use_cases:
                for use_case in use_cases:
                    if isinstance(use_case, str):
                        similarity = self._calculate_text_similarity(
                            original_query.lower(), 
                            use_case.lower()
                        )
                        if similarity > 0.5:
                            score += 1.0
                            break
            
            # Score based on SQL pattern availability
            if any(field in report for field in ["example_sql", "view_sql", "example_query"]):
                score += 1.0
            
            total_score += score
            max_score += max_report_score
        
        if max_score == 0:
            return 0.0
        
        confidence = total_score / max_score
        return min(max(confidence, 0.1), 1.0)  # Clamp between 0.1 and 1.0
    
    def _report_mentions_views(self, report: Dict[str, Any], view_context: ViewContext) -> bool:
        """Check if report mentions any of the selected views."""
        all_views = view_context.core_views + view_context.supporting_views
        
        # Check various fields for view mentions
        text_fields = ["description", "report_description", "data_returned", "view_name"]
        
        for field in text_fields:
            content = report.get(field, "")
            if isinstance(content, str):
                content_lower = content.lower()
                for view in all_views:
                    if view.lower() in content_lower:
                        return True
        
        return False
    
    def _report_mentions_domains(self, report: Dict[str, Any], domains: List[str]) -> bool:
        """Check if report mentions any of the business domains."""
        text_fields = ["description", "report_description", "use_cases"]
        
        for field in text_fields:
            content = report.get(field, "")
            if isinstance(content, str):
                content_lower = content.lower()
                for domain in domains:
                    if domain.lower() in content_lower:
                        return True
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        item_lower = item.lower()
                        for domain in domains:
                            if domain.lower() in item_lower:
                                return True
        
        return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on common words."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0