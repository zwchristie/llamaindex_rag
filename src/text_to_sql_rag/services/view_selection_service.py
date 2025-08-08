"""
Domain-Filtered View Selection Service for Stage 2 of Cascading RAG.
Selects relevant core and supporting views based on identified business domains.
"""

import structlog
from typing import Dict, Any, List, Optional
import json

from .vector_service import LlamaIndexVectorService
from .llm_service import LLMService
from .view_metadata_service import ViewMetadataService
from ..models.simple_models import DocumentType, ViewContext, DomainContext

logger = structlog.get_logger(__name__)


class ViewSelectionService:
    """Service for domain-filtered view selection with dependency resolution."""
    
    def __init__(
        self,
        vector_service: LlamaIndexVectorService,
        llm_service: LLMService,
        view_metadata_service: Optional[ViewMetadataService] = None
    ):
        """
        Initialize view selection service.
        
        Args:
            vector_service: Vector search service for view retrieval
            llm_service: LLM service for intelligent view selection
            view_metadata_service: Service for loading view metadata from MongoDB
        """
        self.vector_service = vector_service
        self.llm_service = llm_service
        self.view_metadata_service = view_metadata_service or ViewMetadataService()
        
        # Load view domain mappings from MongoDB (with fallback to hardcoded)
        self.view_domain_mappings = self._load_view_domain_mappings()
        
        # Load view priorities for ranking
        self.view_priorities = self._load_view_priorities()
        
        # Load view dependencies from MongoDB (with fallback to hardcoded)
        self.view_dependencies = self._load_view_dependencies()
        
        logger.info("ViewSelectionService initialized", 
                   mappings_count=len(self.view_domain_mappings),
                   dependencies_count=len(self.view_dependencies))
    
    def select_domain_views(
        self, 
        domain_context: DomainContext, 
        max_core_views: int = 5,
        max_supporting_views: int = 8,
        debug: bool = False
    ) -> ViewContext:
        """
        Stage 2: Select relevant views based on identified business domains.
        
        Args:
            domain_context: Result from business domain identification
            max_core_views: Maximum number of core views to select
            max_supporting_views: Maximum number of supporting views
            debug: Enable debug logging
            
        Returns:
            ViewContext with selected core and supporting views
        """
        logger.info(
            "Starting domain-filtered view selection",
            domains=domain_context.identified_domains,
            enhanced_query=domain_context.enhanced_query
        )
        
        # Step 1: Filter views by business domains
        candidate_views = self._filter_views_by_domains(domain_context.identified_domains)
        
        # Step 2: Search for relevant views using enhanced query
        core_views = self._search_core_views(
            domain_context.enhanced_query, 
            candidate_views["core"],
            max_core_views
        )
        
        # Step 3: Resolve view dependencies for supporting views
        supporting_views = self._resolve_view_dependencies(
            core_views,
            candidate_views["supporting"],
            max_supporting_views
        )
        
        # Step 4: Use LLM for intelligent view refinement
        refined_views = self._llm_refine_view_selection(
            domain_context.enhanced_query,
            core_views,
            supporting_views,
            domain_context.business_context
        )
        
        # Step 5: Create dependency mapping
        view_dependencies = self._create_dependency_mapping(
            refined_views["core"],
            refined_views["supporting"]
        )
        
        # Step 6: Generate selection reasoning
        selection_reasoning = self._generate_selection_reasoning(
            domain_context,
            refined_views,
            view_dependencies
        )
        
        context = ViewContext(
            core_views=refined_views["core"],
            supporting_views=refined_views["supporting"],
            view_dependencies=view_dependencies,
            business_domains=domain_context.identified_domains,
            selection_reasoning=selection_reasoning
        )
        
        if debug:
            logger.info(
                "View selection complete",
                core_views=len(refined_views["core"]),
                supporting_views=len(refined_views["supporting"]),
                dependencies=len(view_dependencies)
            )
        
        return context
    
    def _filter_views_by_domains(self, domains: List[str]) -> Dict[str, List[str]]:
        """Filter views by business domains to reduce search space."""
        filtered_views = {"core": [], "supporting": []}
        
        for view_name, view_domains in self.view_domain_mappings.items():
            # Check if view belongs to any of the identified domains
            if any(domain in view_domains for domain in domains):
                # Classify as core or supporting based on naming convention
                if self._is_core_view(view_name):
                    filtered_views["core"].append(view_name)
                else:
                    filtered_views["supporting"].append(view_name)
        
        logger.info(
            "Domain filtering complete",
            core_candidates=len(filtered_views["core"]),
            supporting_candidates=len(filtered_views["supporting"])
        )
        
        return filtered_views
    
    def _is_core_view(self, view_name: str) -> bool:
        """Determine if a view is a core view based on naming patterns."""
        # Core view patterns (main business objects)
        core_patterns = [
            "SUMMARY", "METRICS", "ALLOCATION", "EXECUTION", 
            "PORTFOLIO", "PARTICIPATION", "PRICING"
        ]
        
        # Supporting view patterns (detailed/supporting data)  
        supporting_patterns = [
            "DETAILS", "INSTRUMENTS", "SETTLEMENT", "BREAKDOWN"
        ]
        
        view_upper = view_name.upper()
        
        # Check for core patterns first
        if any(pattern in view_upper for pattern in core_patterns):
            return True
        
        # Check for supporting patterns
        if any(pattern in view_upper for pattern in supporting_patterns):
            return False
        
        # Default to core view if unclear
        return True
    
    def _search_core_views(
        self, 
        enhanced_query: str, 
        candidate_views: List[str], 
        max_views: int
    ) -> List[str]:
        """Search for relevant core views using enhanced query."""
        try:
            # Search core view documents
            results = self.vector_service.search_similar(
                query=enhanced_query,
                retriever_type="hybrid",
                similarity_top_k=max_views * 2,  # Get more results for filtering
                document_type=DocumentType.CORE_VIEW.value
            )
            
            # Extract view names from results
            found_views = []
            for result in results:
                view_name = self._extract_view_name_from_result(result)
                if view_name and view_name in candidate_views:
                    if view_name not in found_views:
                        found_views.append(view_name)
                        if len(found_views) >= max_views:
                            break
            
            # If not enough found, add remaining candidates
            for view in candidate_views:
                if view not in found_views and len(found_views) < max_views:
                    found_views.append(view)
            
            return found_views
            
        except Exception as e:
            logger.warning("Core view search failed, using candidates", error=str(e))
            return candidate_views[:max_views]
    
    def _resolve_view_dependencies(
        self, 
        core_views: List[str], 
        candidate_supporting: List[str],
        max_supporting: int
    ) -> List[str]:
        """Resolve view dependencies to include relevant supporting views."""
        supporting_views = []
        
        # Add supporting views that are dependencies of selected core views
        for core_view in core_views:
            if core_view in self.view_dependencies:
                dependencies = self.view_dependencies[core_view]
                for dep_view in dependencies:
                    if (dep_view in candidate_supporting and 
                        dep_view not in supporting_views and
                        len(supporting_views) < max_supporting):
                        supporting_views.append(dep_view)
        
        # Add remaining candidate supporting views if space allows
        for view in candidate_supporting:
            if (view not in supporting_views and 
                len(supporting_views) < max_supporting):
                supporting_views.append(view)
        
        return supporting_views
    
    def _llm_refine_view_selection(
        self,
        enhanced_query: str,
        core_views: List[str],
        supporting_views: List[str],
        business_context: str
    ) -> Dict[str, List[str]]:
        """Use LLM to refine view selection based on query intent."""
        
        all_views = core_views + supporting_views
        if len(all_views) <= 8:  # If selection is reasonable, keep as is
            return {"core": core_views, "supporting": supporting_views}
        
        # Generate dynamic view descriptions from MongoDB metadata
        view_descriptions = self._generate_dynamic_view_descriptions(core_views + supporting_views)
        
        prompt = f"""Analyze the user query and refine the view selection to focus on the most relevant views.

Enhanced Query: {enhanced_query}
Business Context: {business_context}

Available Views:
Core Views: {', '.join(core_views)}
Supporting Views: {', '.join(supporting_views)}

View Descriptions:
{view_descriptions}

Instructions:
1. Select 3-5 most relevant CORE views that directly answer the query
2. Select 2-5 SUPPORTING views that provide necessary detail
3. Prioritize views based on their priority scores and domain relevance
4. Consider view descriptions to understand each view's purpose
5. Avoid views that don't contribute to answering the query

Return as JSON:
{{"core": ["VIEW1", "VIEW2"], "supporting": ["VIEW3", "VIEW4"]}}

Refined Selection:"""

        try:
            response = self.llm_service.generate_response(prompt)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                selection = json.loads(json_match.group())
                
                # Validate selections exist in original lists
                refined_core = [v for v in selection.get("core", []) if v in core_views]
                refined_supporting = [v for v in selection.get("supporting", []) if v in supporting_views]
                
                if refined_core:  # Only use LLM result if it found core views
                    return {"core": refined_core, "supporting": refined_supporting}
                
        except Exception as e:
            logger.warning("LLM view refinement failed", error=str(e))
        
        # Fallback: use original selection with size limits
        return {
            "core": core_views[:5], 
            "supporting": supporting_views[:5]
        }
    
    def _create_dependency_mapping(
        self, 
        core_views: List[str], 
        supporting_views: List[str]
    ) -> Dict[str, List[str]]:
        """Create mapping of which supporting views support which core views."""
        dependencies = {}
        
        for core_view in core_views:
            core_deps = []
            if core_view in self.view_dependencies:
                # Add known dependencies that are in supporting views
                known_deps = self.view_dependencies[core_view]
                core_deps.extend([dep for dep in known_deps if dep in supporting_views])
            
            if core_deps:
                dependencies[core_view] = core_deps
        
        return dependencies
    
    def _generate_selection_reasoning(
        self,
        domain_context: DomainContext,
        selected_views: Dict[str, List[str]],
        dependencies: Dict[str, List[str]]
    ) -> str:
        """Generate human-readable reasoning for view selection."""
        reasoning_parts = []
        
        # Domain-based reasoning
        domains_str = ", ".join(domain_context.identified_domains)
        reasoning_parts.append(f"Selected views based on business domains: {domains_str}")
        
        # Core view reasoning
        if selected_views["core"]:
            core_str = ", ".join(selected_views["core"])
            reasoning_parts.append(f"Core views ({len(selected_views['core'])}): {core_str}")
        
        # Supporting view reasoning
        if selected_views["supporting"]:
            supporting_str = ", ".join(selected_views["supporting"])
            reasoning_parts.append(f"Supporting views ({len(selected_views['supporting'])}): {supporting_str}")
        
        # Dependency reasoning
        if dependencies:
            dep_count = sum(len(deps) for deps in dependencies.values())
            reasoning_parts.append(f"Resolved {dep_count} view dependencies for enhanced context")
        
        return ". ".join(reasoning_parts)
    
    def _extract_view_name_from_result(self, result: Any) -> Optional[str]:
        """Extract view name from search result."""
        try:
            if isinstance(result, dict):
                metadata = result.get("metadata", {})
                return (metadata.get("view_name") or 
                       metadata.get("title") or
                       metadata.get("file_name", "").replace(".json", "").upper())
            else:
                # Handle LlamaIndex node format
                metadata = getattr(result, 'metadata', {})
                return (metadata.get("view_name") or
                       metadata.get("title") or 
                       metadata.get("file_name", "").replace(".json", "").upper())
        except Exception:
            return None
    
    def update_view_mappings(self, view_mappings: Dict[str, List[str]]) -> None:
        """Update view-to-domain mappings with new data."""
        self.view_domain_mappings.update(view_mappings)
        logger.info("Updated view domain mappings", count=len(view_mappings))
    
    def update_view_dependencies(self, dependencies: Dict[str, List[str]]) -> None:
        """Update view dependency mappings with new data.""" 
        self.view_dependencies.update(dependencies)
        logger.info("Updated view dependencies", count=len(dependencies))
    
    def get_view_mappings(self) -> Dict[str, List[str]]:
        """Get current view-to-domain mappings."""
        return self.view_domain_mappings
    
    def get_view_dependencies(self) -> Dict[str, List[str]]:
        """Get current view dependency mappings."""
        return self.view_dependencies
    
    def _load_view_domain_mappings(self) -> Dict[str, List[str]]:
        """Load view domain mappings from MongoDB with fallback."""
        try:
            mappings = self.view_metadata_service.get_view_domain_mappings()
            if mappings:
                logger.info("Loaded view domain mappings from MongoDB", count=len(mappings))
                return mappings
            else:
                logger.error("No mappings found in MongoDB! Please run the discovery script.")
                return {}
        except Exception as e:
            logger.error("Failed to load view domain mappings", error=str(e))
            logger.error("MongoDB connection required for view metadata. Please check connection.")
            return {}
    
    def _load_view_dependencies(self) -> Dict[str, List[str]]:
        """Load view dependencies from MongoDB with fallback."""
        try:
            dependencies = self.view_metadata_service.get_view_dependencies()
            if dependencies:
                logger.info("Loaded view dependencies from MongoDB", count=len(dependencies))
                return dependencies
            else:
                logger.info("No dependencies found in MongoDB - this is normal for new installations")
                return {}
        except Exception as e:
            logger.error("Failed to load view dependencies", error=str(e))
            logger.warning("View dependencies unavailable, continuing without dependencies")
            return {}
    
    def _load_view_priorities(self) -> Dict[str, int]:
        """Load view priorities from MongoDB."""
        try:
            priorities = self.view_metadata_service.get_view_priorities()
            logger.info("Loaded view priorities from MongoDB", count=len(priorities))
            return priorities
        except Exception as e:
            logger.error("Failed to load view priorities", error=str(e))
            return {}
    
    def _get_fallback_mappings(self) -> Dict[str, List[str]]:
        """Fallback mappings removed - everything comes from MongoDB."""
        logger.error("Fallback mappings called but removed. All metadata must come from MongoDB.")
        return {}
    
    def _get_fallback_dependencies(self) -> Dict[str, List[str]]:
        """Fallback dependencies removed - everything comes from MongoDB."""
        logger.error("Fallback dependencies called but removed. All metadata must come from MongoDB.")
        return {}
    
    def refresh_metadata(self):
        """Refresh view metadata from MongoDB."""
        logger.info("Refreshing view metadata from MongoDB")
        self.view_metadata_service.refresh_cache()
        self.view_domain_mappings = self._load_view_domain_mappings()
        self.view_dependencies = self._load_view_dependencies()
        self.view_priorities = self._load_view_priorities()
        logger.info("View metadata refreshed", 
                   mappings=len(self.view_domain_mappings),
                   dependencies=len(self.view_dependencies))
    
    def update_view_usage_stats(self, view_name: str, success: bool, confidence: float = None,
                               response_time_ms: float = None, query: str = None):
        """Update usage statistics for a view."""
        self.view_metadata_service.update_usage_stats(
            view_name=view_name,
            success=success, 
            confidence=confidence,
            response_time_ms=response_time_ms,
            query=query
        )
    
    def _generate_dynamic_view_descriptions(self, view_names: List[str]) -> str:
        """Generate dynamic view descriptions from MongoDB metadata."""
        try:
            descriptions = []
            
            # Get view metadata from MongoDB
            for view_name in view_names[:20]:  # Limit to prevent huge prompts
                try:
                    view_doc = self.view_metadata_service.view_mappings_collection.find_one(
                        {"view_name": view_name}
                    )
                    
                    if view_doc:
                        description = view_doc.get("description", "No description available")
                        view_type = view_doc.get("view_type", "unknown")
                        priority = view_doc.get("priority_score", 5)
                        domains = view_doc.get("business_domains", [])
                        
                        desc_line = f"- {view_name} ({view_type}, priority {priority}): {description}"
                        if domains:
                            desc_line += f" [Domains: {', '.join(domains)}]"
                        
                        descriptions.append(desc_line)
                    else:
                        descriptions.append(f"- {view_name}: No metadata available")
                        
                except Exception as e:
                    logger.warning(f"Error getting metadata for {view_name}: {e}")
                    descriptions.append(f"- {view_name}: Metadata unavailable")
            
            return "\n".join(descriptions)
            
        except Exception as e:
            logger.error(f"Error generating dynamic view descriptions: {e}")
            return "View descriptions unavailable - using view names only"