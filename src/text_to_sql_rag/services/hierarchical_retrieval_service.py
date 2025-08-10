"""
Hierarchical Retrieval Service for multi-step metadata retrieval.

This service implements the hierarchical RAG retrieval pipeline:
1. Business Domain Identification
2. Core View Retrieval
3. Supporting View Retrieval 
4. Report Example Retrieval
5. Lookup Value Retrieval
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalContext:
    """Complete context assembled from hierarchical retrieval."""
    identified_domains: List[int]
    core_views: List[Dict[str, Any]]
    supporting_views: List[Dict[str, Any]]
    reports: List[Dict[str, Any]]
    lookups: List[Dict[str, Any]]
    retrieval_steps: Dict[str, Any]
    confidence: float


@dataclass
class RetrievalConfig:
    """Configuration for hierarchical retrieval."""
    max_core_views: int = 3
    max_supporting_views: int = 5
    max_reports: int = 2
    max_lookups: int = 10
    enable_parallel_lookup: bool = True
    domain_confidence_threshold: float = 0.7
    semantic_search_threshold: float = 0.5


class HierarchicalRetrievalService:
    """Service for hierarchical metadata retrieval."""
    
    def __init__(self, 
                 meta_docs_path: Path,
                 embedding_service,
                 vector_service, 
                 llm_service,
                 config: Optional[RetrievalConfig] = None):
        self.meta_docs_path = meta_docs_path
        self.embedding_service = embedding_service
        self.vector_service = vector_service
        self.llm_service = llm_service
        self.config = config or RetrievalConfig()
        
        # Cached metadata
        self.business_domains: Dict[int, Dict[str, Any]] = {}
        self.view_metadata: Dict[str, Dict[str, Any]] = {}
        self.report_metadata: Dict[str, Dict[str, Any]] = {}
        self.lookup_metadata: Dict[int, Dict[str, Any]] = {}
        
        # Load all metadata
        self._load_all_metadata()
    
    def _load_all_metadata(self):
        """Load all metadata types from files."""
        try:
            self._load_business_domains()
            self._load_view_metadata()
            self._load_report_metadata()
            self._load_lookup_metadata()
            
            logger.info("Loaded hierarchical metadata", 
                       domains=len(self.business_domains),
                       views=len(self.view_metadata),
                       reports=len(self.report_metadata),
                       lookups=len(self.lookup_metadata))
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
    
    def _load_business_domains(self):
        """Load business domain definitions."""
        domains_file = self.meta_docs_path / "business_domains.json"
        if domains_file.exists():
            with open(domains_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for domain in data.get('business_domains', []):
                    self.business_domains[domain['domain_id']] = domain
    
    def _load_view_metadata(self):
        """Load view metadata from files."""
        views_dir = self.meta_docs_path / "views"
        if views_dir.exists():
            for json_file in views_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        view_data = json.load(f)
                        view_name = view_data.get('view_name', json_file.stem)
                        self.view_metadata[view_name] = view_data
                except Exception as e:
                    logger.warning(f"Failed to load view {json_file.name}: {e}")
    
    def _load_report_metadata(self):
        """Load report metadata from files."""
        reports_dir = self.meta_docs_path / "reports"
        if reports_dir.exists():
            for json_file in reports_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                        report_name = report_data.get('report_name', json_file.stem)
                        self.report_metadata[report_name] = report_data
                except Exception as e:
                    logger.warning(f"Failed to load report {json_file.name}: {e}")
    
    def _load_lookup_metadata(self):
        """Load lookup metadata from files."""
        lookups_dir = self.meta_docs_path / "lookups"
        if lookups_dir.exists():
            for json_file in lookups_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        lookup_data = json.load(f)
                        lookup_id = lookup_data.get('lookup_id')
                        if lookup_id:
                            self.lookup_metadata[lookup_id] = lookup_data
                except Exception as e:
                    logger.warning(f"Failed to load lookup {json_file.name}: {e}")
    
    async def retrieve_hierarchical_context(self, user_query: str) -> HierarchicalContext:
        """Execute the full hierarchical retrieval pipeline."""
        logger.info(f"Starting hierarchical retrieval for: {user_query}")
        
        retrieval_steps = {}
        
        try:
            # Step 1: Business Domain Identification
            identified_domains = await self._identify_business_domains(user_query)
            retrieval_steps['domain_identification'] = {
                'domains': identified_domains,
                'method': 'llm_analysis'
            }
            
            # Step 2: Core View Retrieval
            core_views = await self._retrieve_core_views(user_query, identified_domains)
            retrieval_steps['core_views'] = {
                'count': len(core_views),
                'views': [v.get('view_name') for v in core_views]
            }
            
            # Step 3: Supporting View Retrieval
            supporting_views = await self._retrieve_supporting_views(user_query, core_views, identified_domains)
            retrieval_steps['supporting_views'] = {
                'count': len(supporting_views),
                'views': [v.get('view_name') for v in supporting_views]
            }
            
            # Steps 4 & 5: Report and Lookup Retrieval (parallel)
            all_views = core_views + supporting_views
            
            if self.config.enable_parallel_lookup:
                reports, lookups = await asyncio.gather(
                    self._retrieve_report_examples(user_query, all_views, identified_domains),
                    self._retrieve_lookup_values(all_views)
                )
            else:
                reports = await self._retrieve_report_examples(user_query, all_views, identified_domains)
                lookups = await self._retrieve_lookup_values(all_views)
            
            retrieval_steps['reports'] = {
                'count': len(reports),
                'reports': [r.get('report_name') for r in reports]
            }
            retrieval_steps['lookups'] = {
                'count': len(lookups),
                'lookup_ids': [l.get('lookup_id') for l in lookups]
            }
            
            # Calculate overall confidence
            confidence = self._calculate_retrieval_confidence(identified_domains, core_views, supporting_views)
            
            context = HierarchicalContext(
                identified_domains=identified_domains,
                core_views=core_views,
                supporting_views=supporting_views,
                reports=reports,
                lookups=lookups,
                retrieval_steps=retrieval_steps,
                confidence=confidence
            )
            
            logger.info("Hierarchical retrieval completed",
                       domains=len(identified_domains),
                       core_views=len(core_views),
                       supporting_views=len(supporting_views),
                       reports=len(reports),
                       lookups=len(lookups),
                       confidence=confidence)
            
            return context
            
        except Exception as e:
            logger.error(f"Error in hierarchical retrieval: {e}")
            # Return minimal fallback context
            return self._create_fallback_context(user_query, retrieval_steps)
    
    async def _identify_business_domains(self, user_query: str) -> List[int]:
        """Step 1: Identify relevant business domains."""
        try:
            # Create domain context for LLM
            domain_context = self._create_domain_context_for_llm()
            
            prompt = f"""Analyze the following user query and identify which business domains are most relevant.

User Query: {user_query}

Available Business Domains:
{domain_context}

Instructions:
1. Consider which domains would contain the data needed to answer the query
2. Include related domains that might have supporting information
3. Focus on domains whose views and reports would be relevant
4. Return ONLY the domain IDs as a comma-separated list (e.g., 1,2,3)

Relevant Domain IDs:"""
            
            response = await self.llm_service.generate_response(prompt)
            
            # Parse domain IDs from response
            domain_ids = self._parse_domain_ids_from_response(response)
            
            if not domain_ids:
                # Fallback to keyword matching
                domain_ids = self._identify_domains_by_keywords(user_query)
            
            return domain_ids[:4]  # Limit to 4 domains max
            
        except Exception as e:
            logger.error(f"Error identifying domains: {e}")
            return list(self.business_domains.keys())[:3]  # Fallback to first 3 domains
    
    async def _retrieve_core_views(self, user_query: str, domain_ids: List[int]) -> List[Dict[str, Any]]:
        """Step 2: Retrieve core views for identified domains."""
        try:
            # Get candidate views from domains
            candidate_views = self._get_candidate_views_for_domains(domain_ids, core_only=True)
            
            if not candidate_views:
                # Fallback to all core views if domain filtering fails
                candidate_views = [v for v in self.view_metadata.values() 
                                 if v.get('view_type') == 'CORE']
            
            # Rank by semantic similarity
            ranked_views = await self._rank_views_by_similarity(user_query, candidate_views)
            
            return ranked_views[:self.config.max_core_views]
            
        except Exception as e:
            logger.error(f"Error retrieving core views: {e}")
            return []
    
    async def _retrieve_supporting_views(self, user_query: str, core_views: List[Dict[str, Any]], domain_ids: List[int]) -> List[Dict[str, Any]]:
        """Step 3: Retrieve supporting views based on core views and domains."""
        try:
            # Get supporting view candidates from domains
            candidate_views = self._get_candidate_views_for_domains(domain_ids, core_only=False)
            
            # Filter out core views we already have
            core_view_names = {v.get('view_name') for v in core_views}
            candidate_views = [v for v in candidate_views 
                             if v.get('view_name') not in core_view_names]
            
            # Filter to supporting views
            candidate_views = [v for v in candidate_views 
                             if v.get('view_type') == 'SUPPORTING']
            
            # Rank by semantic similarity
            ranked_views = await self._rank_views_by_similarity(user_query, candidate_views)
            
            return ranked_views[:self.config.max_supporting_views]
            
        except Exception as e:
            logger.error(f"Error retrieving supporting views: {e}")
            return []
    
    async def _retrieve_report_examples(self, user_query: str, all_views: List[Dict[str, Any]], domain_ids: List[int]) -> List[Dict[str, Any]]:
        """Step 4: Retrieve relevant report examples."""
        try:
            # Get view names for filtering
            view_names = {v.get('view_name') for v in all_views}
            
            # Find reports that relate to our views and domains
            candidate_reports = []
            
            for report in self.report_metadata.values():
                # Check domain overlap
                report_domains = set(report.get('business_domains', []))
                if report_domains.intersection(set(domain_ids)):
                    candidate_reports.append(report)
                    continue
                
                # Check view relationships
                report_view = report.get('view_name')
                related_views = set(report.get('related_views', []))
                
                if report_view in view_names or related_views.intersection(view_names):
                    candidate_reports.append(report)
            
            # Rank by relevance to query
            ranked_reports = await self._rank_reports_by_similarity(user_query, candidate_reports)
            
            return ranked_reports[:self.config.max_reports]
            
        except Exception as e:
            logger.error(f"Error retrieving reports: {e}")
            return []
    
    async def _retrieve_lookup_values(self, all_views: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 5: Retrieve lookup values for columns with lookup_id."""
        try:
            # Extract all lookup IDs from view columns
            lookup_ids = set()
            
            for view in all_views:
                columns = view.get('columns', [])
                for column in columns:
                    lookup_id = column.get('lookup_id')
                    if lookup_id:
                        lookup_ids.add(lookup_id)
            
            # Get lookup metadata for these IDs
            lookups = []
            for lookup_id in lookup_ids:
                lookup_data = self.lookup_metadata.get(lookup_id)
                if lookup_data:
                    lookups.append(lookup_data)
            
            return lookups[:self.config.max_lookups]
            
        except Exception as e:
            logger.error(f"Error retrieving lookups: {e}")
            return []
    
    # Helper methods continue in next part...
    
    def _create_domain_context_for_llm(self) -> str:
        """Create formatted domain context for LLM."""
        context_parts = []
        
        for domain_id, domain in self.business_domains.items():
            context_parts.append(
                f"ID: {domain_id}\n"
                f"Name: {domain['domain_name']}\n"
                f"Description: {domain['description']}\n"
                f"Keywords: {', '.join(domain['keywords'])}\n"
            )
        
        return "\n".join(context_parts)
    
    def _parse_domain_ids_from_response(self, response: str) -> List[int]:
        """Parse domain IDs from LLM response."""
        try:
            import re
            numbers = re.findall(r'\b\d+\b', response)
            
            domain_ids = []
            for num_str in numbers:
                domain_id = int(num_str)
                if domain_id in self.business_domains:
                    domain_ids.append(domain_id)
            
            return domain_ids
            
        except Exception:
            return []
    
    def _identify_domains_by_keywords(self, user_query: str) -> List[int]:
        """Fallback keyword-based domain identification."""
        query_lower = user_query.lower()
        matched_domains = []
        
        for domain_id, domain in self.business_domains.items():
            keywords = domain.get('keywords', [])
            if any(keyword.lower() in query_lower for keyword in keywords):
                matched_domains.append(domain_id)
        
        return matched_domains
    
    def _get_candidate_views_for_domains(self, domain_ids: List[int], core_only: bool = False) -> List[Dict[str, Any]]:
        """Get candidate views for specified domains."""
        candidate_views = []
        
        for view_name, view_data in self.view_metadata.items():
            view_domains = set(view_data.get('business_domains', []))
            
            # Check if view belongs to any of our domains
            if view_domains.intersection(set(domain_ids)):
                if not core_only or view_data.get('view_type') == 'CORE':
                    candidate_views.append(view_data)
        
        return candidate_views
    
    async def _rank_views_by_similarity(self, query: str, views: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank views by semantic similarity to query."""
        if not views:
            return []
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.get_embedding(query)
            
            view_scores = []
            
            for view in views:
                # Create view text for similarity
                view_text = self._create_view_text_for_similarity(view)
                view_embedding = await self.embedding_service.get_embedding(view_text)
                
                # Calculate cosine similarity
                similarity = self._calculate_cosine_similarity(query_embedding, view_embedding)
                view_scores.append((view, similarity))
            
            # Sort by similarity descending
            view_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [view for view, score in view_scores if score >= self.config.semantic_search_threshold]
            
        except Exception as e:
            logger.error(f"Error ranking views by similarity: {e}")
            return views  # Return unranked if similarity ranking fails
    
    async def _rank_reports_by_similarity(self, query: str, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank reports by semantic similarity to query."""
        if not reports:
            return []
        
        try:
            query_embedding = await self.embedding_service.get_embedding(query)
            
            report_scores = []
            
            for report in reports:
                report_text = self._create_report_text_for_similarity(report)
                report_embedding = await self.embedding_service.get_embedding(report_text)
                
                similarity = self._calculate_cosine_similarity(query_embedding, report_embedding)
                report_scores.append((report, similarity))
            
            report_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [report for report, score in report_scores if score >= self.config.semantic_search_threshold]
            
        except Exception as e:
            logger.error(f"Error ranking reports by similarity: {e}")
            return reports
    
    def _create_view_text_for_similarity(self, view: Dict[str, Any]) -> str:
        """Create text representation of view for similarity calculation."""
        parts = [
            view.get('view_name', ''),
            view.get('description', ''),
            view.get('use_cases', '')
        ]
        
        # Add column information
        columns = view.get('columns', [])
        column_names = [col.get('name', '') for col in columns]
        if column_names:
            parts.append(' '.join(column_names))
        
        return ' '.join(filter(None, parts))
    
    def _create_report_text_for_similarity(self, report: Dict[str, Any]) -> str:
        """Create text representation of report for similarity calculation."""
        parts = [
            report.get('report_name', ''),
            report.get('report_description', ''),
            report.get('use_cases', ''),
            report.get('data_returned', '')
        ]
        
        return ' '.join(filter(None, parts))
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import math
            
            if len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude_a = math.sqrt(sum(a * a for a in vec1))
            magnitude_b = math.sqrt(sum(b * b for b in vec2))
            
            if magnitude_a == 0 or magnitude_b == 0:
                return 0.0
            
            return dot_product / (magnitude_a * magnitude_b)
            
        except Exception:
            return 0.0
    
    def _calculate_retrieval_confidence(self, domains: List[int], core_views: List[Dict[str, Any]], supporting_views: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in retrieval results."""
        # Base confidence from domain identification
        domain_confidence = min(len(domains) / 2.0, 1.0)  # Normalize to 0-1
        
        # View retrieval confidence
        view_confidence = min((len(core_views) + len(supporting_views)) / 5.0, 1.0)
        
        # Combined confidence
        overall_confidence = (domain_confidence * 0.4 + view_confidence * 0.6)
        
        return max(overall_confidence, 0.3)  # Minimum confidence of 0.3
    
    def _create_fallback_context(self, user_query: str, retrieval_steps: Dict[str, Any]) -> HierarchicalContext:
        """Create fallback context when retrieval fails."""
        # Return basic context with available metadata
        fallback_views = list(self.view_metadata.values())[:3]
        
        return HierarchicalContext(
            identified_domains=list(self.business_domains.keys())[:2],
            core_views=fallback_views,
            supporting_views=[],
            reports=list(self.report_metadata.values())[:1],
            lookups=list(self.lookup_metadata.values())[:3],
            retrieval_steps=retrieval_steps,
            confidence=0.3
        )