"""Custom internal LLM API integration service."""

import json
import uuid
from typing import Dict, Any, Optional, List
import requests
import structlog

from ..config.settings import settings

logger = structlog.get_logger(__name__)


class CustomLLMService:
    """Service for text generation using custom internal LLM API."""
    
    def __init__(self):
        if not settings.custom_llm:
            raise ValueError("Custom LLM settings not configured")
        
        self.base_url = settings.custom_llm.base_url
        self.deployment_id = settings.custom_llm.deployment_id
        self.model_name = settings.custom_llm.model_name
        self.timeout = settings.custom_llm.timeout
        self.max_retries = settings.custom_llm.max_retries
        
        # HTTP client for API calls
        self.client = requests.Session()
        self.client.timeout = self.timeout
        
        logger.info("Custom LLM service initialized", base_url=self.base_url, deployment_id=self.deployment_id)
    
    def generate_text(
        self, 
        prompt: str,
        conversation_id: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using the custom LLM API."""
        try:
            # Use provided model name or fall back to configured default
            model = model_name or self.model_name
            
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            # Build the API URL
            url = f"{self.base_url}/invoke/{self.deployment_id}"
            
            # Prepare request payload
            payload = {
                "message": prompt,
                "conversation_id": conversation_id
            }
            
            if model:
                payload["model_name"] = model
            
            logger.debug("Sending request to custom LLM API", url=url, conversation_id=conversation_id)
            
            # Make the API request
            response = self.client.post(url, json=payload)
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            if "message" not in response_data:
                raise ValueError("Invalid response format: missing 'message' field")
            
            generated_text = response_data["message"]
            
            logger.info(
                "Generated text using custom LLM",
                conversation_id=conversation_id,
                response_length=len(generated_text),
                model=model
            )
            
            return generated_text
            
        except requests.HTTPError as e:
            logger.error("HTTP error from custom LLM API", error=str(e))
            raise
        except Exception as e:
            logger.error("Failed to generate text with custom LLM", error=str(e))
            raise
    
    def continue_conversation(
        self,
        message: str,
        conversation_id: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Continue an existing conversation using the follow-up endpoint."""
        try:
            model = model_name or self.model_name
            
            # Build the follow-up API URL
            url = f"{self.base_url}/invoke/followup/{self.deployment_id}"
            
            # Prepare request payload
            payload = {
                "message": message,
                "conversation_id": conversation_id
            }
            
            if model:
                payload["model_name"] = model
            
            logger.debug("Continuing conversation with custom LLM API", url=url, conversation_id=conversation_id)
            
            # Make the API request
            response = self.client.post(url, json=payload)
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            if "message" not in response_data:
                raise ValueError("Invalid response format: missing 'message' field")
            
            generated_text = response_data["message"]
            
            logger.info(
                "Continued conversation with custom LLM",
                conversation_id=conversation_id,
                response_length=len(generated_text),
                model=model
            )
            
            return generated_text
            
        except requests.HTTPError as e:
            logger.error("HTTP error from custom LLM follow-up API", error=str(e))
            raise
        except Exception as e:
            logger.error("Failed to continue conversation with custom LLM", error=str(e))
            raise
    
    def get_conversation_messages(
        self,
        conversation_id: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get current messages from a conversation."""
        try:
            # Build the conversation API URL
            url = f"{self.base_url}/conversation/{self.deployment_id}"
            
            # Add conversation_id as query parameter
            params = {"conversation_id": conversation_id}
            
            logger.debug("Getting conversation messages", url=url, conversation_id=conversation_id)
            
            # Make the API request
            response = self.client.get(url, params=params)
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            # Assume the API returns messages in some format
            # You may need to adjust this based on your actual API response format
            messages = response_data.get("messages", [])
            
            logger.info(
                "Retrieved conversation messages",
                conversation_id=conversation_id,
                message_count=len(messages)
            )
            
            return messages
            
        except requests.HTTPError as e:
            logger.error("HTTP error from conversation API", error=str(e))
            raise
        except Exception as e:
            logger.error("Failed to get conversation messages", error=str(e))
            raise
    
    def generate_sql_query(
        self, 
        natural_language_query: str,
        schema_context: str,
        example_queries: Optional[List[str]] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate SQL query from natural language with context."""
        
        # Build prompt with context
        prompt_parts = [
            "You are an expert SQL query generator. Given a natural language question and database schema context, generate an accurate SQL query.",
            "",
            "Database Schema Context:",
            schema_context,
            ""
        ]
        
        if example_queries:
            prompt_parts.extend([
                "Example Queries:",
                *[f"- {query}" for query in example_queries],
                ""
            ])
        
        prompt_parts.extend([
            f"Natural Language Query: {natural_language_query}",
            "",
            "Generate a SQL query that answers the question. Respond with JSON in this format:",
            '{"sql_query": "SELECT ...", "explanation": "Brief explanation of the query", "confidence": 0.95}'
        ])
        
        prompt = "\n".join(prompt_parts)
        
        try:
            response_text = self.generate_text(
                prompt=prompt,
                conversation_id=conversation_id
            )
            
            # Try to parse JSON response
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError:
                # Fallback: extract SQL from plain text response
                return {
                    "sql_query": self._extract_sql_from_text(response_text),
                    "explanation": "Generated from text response",
                    "confidence": 0.7
                }
                
        except Exception as e:
            logger.error("Failed to generate SQL query", error=str(e))
            return {
                "sql_query": "",
                "explanation": f"Error generating query: {str(e)}",
                "confidence": 0.0
            }
    
    def _extract_sql_from_text(self, text: str) -> str:
        """Extract SQL query from plain text response."""
        import re
        
        # Look for SQL in code blocks first
        sql_block_pattern = re.compile(r'```sql\s*(.*?)\s*```', re.DOTALL | re.IGNORECASE)
        match = sql_block_pattern.search(text)
        if match:
            return match.group(1).strip()
        
        # Look for SQL keywords
        sql_pattern = re.compile(
            r'(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP).*?(?=\n\n|\Z)',
            re.IGNORECASE | re.DOTALL
        )
        
        matches = sql_pattern.findall(text)
        if matches:
            return matches[0].strip()
        
        return text.strip()
    
    def health_check(self) -> bool:
        """Check if the custom LLM service is accessible."""
        try:
            # Try a simple request to test connectivity
            # You might want to implement a specific health check endpoint
            url = f"{self.base_url}/invoke/{self.deployment_id}"
            
            # Simple test message
            test_payload = {
                "message": "Health check",
                "conversation_id": "health-check"
            }
            
            response = self.client.post(url, json=test_payload, timeout=5)
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            logger.error("Custom LLM health check failed", error=str(e))
            return False
    
    def __del__(self):
        """Cleanup HTTP client on destruction."""
        if hasattr(self, 'client'):
            self.client.close()