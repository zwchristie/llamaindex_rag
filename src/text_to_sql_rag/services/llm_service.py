"""
LLM Service for text processing and reasoning tasks.
Provides a simple interface for LLM operations used by hierarchical context service.
"""

import structlog
from typing import Optional, Dict, Any

from .llm_provider_factory import llm_factory

logger = structlog.get_logger(__name__)


class LLMService:
    """Service for LLM-based text processing and reasoning."""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4"):
        """
        Initialize LLM service.
        
        Args:
            provider: LLM provider name
            model: Model name to use
        """
        self.provider = provider
        self.model = model
        self._llm = None
    
    def _get_llm(self):
        """Get or create LLM instance."""
        if self._llm is None:
            self._llm = llm_factory.get_llm_client(
                provider=self.provider,
                model=self.model
            )
        return self._llm
    
    def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
        **kwargs
    ) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Generated response text
        """
        try:
            llm = self._get_llm()
            
            # Call the LLM - this will depend on your LLM factory implementation
            response = llm.complete(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            elif isinstance(response, str):
                return response
            else:
                return str(response)
                
        except Exception as e:
            logger.error("LLM generation failed", error=str(e), prompt_length=len(prompt))
            # Return a fallback response
            return "Error: Unable to generate response"
    
    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze query complexity for context planning.
        
        Args:
            query: User query to analyze
            
        Returns:
            Dictionary with complexity analysis
        """
        complexity_prompt = f"""Analyze this user query for complexity:

Query: {query}

Classify the complexity and identify key characteristics:
1. Simple: Basic single-table queries
2. Medium: Multi-table joins or aggregations
3. Complex: Multiple joins, subqueries, or advanced functions

Return JSON format:
{{
    "complexity": "simple|medium|complex",
    "requires_joins": true/false,
    "requires_aggregation": true/false,
    "requires_date_filtering": true/false,
    "requires_lookups": true/false,
    "estimated_tables": 1-5
}}"""

        try:
            response = self.generate_response(complexity_prompt, temperature=0.0)
            
            # Try to parse JSON response
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                return analysis
            
        except Exception as e:
            logger.warning("Query complexity analysis failed", error=str(e))
        
        # Fallback analysis
        return {
            "complexity": "medium",
            "requires_joins": True,
            "requires_aggregation": False,
            "requires_date_filtering": "date" in query.lower(),
            "requires_lookups": any(word in query.lower() for word in ["status", "type", "category"]),
            "estimated_tables": 3
        }