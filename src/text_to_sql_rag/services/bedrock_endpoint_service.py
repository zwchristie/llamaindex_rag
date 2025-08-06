"""Bedrock integration using HTTP endpoint instead of direct AWS SDK."""

import json
import requests
from typing import List, Dict, Any, Optional
import structlog

from ..config.settings import settings

logger = structlog.get_logger(__name__)


class BedrockEndpointService:
    """Service for interacting with Bedrock through HTTP endpoint."""
    
    def __init__(self, endpoint_base_url: str):
        """
        Initialize Bedrock endpoint service.
        
        Args:
            endpoint_base_url: Base URL for the Bedrock endpoint (e.g., "https://api.company.com")
        """
        self.endpoint_base_url = endpoint_base_url.rstrip('/')
        self.invoke_url = f"{self.endpoint_base_url}/invokeBedrock/"
        self.session = requests.Session()
        
        # Configure SSL verification based on settings
        self.verify_ssl = settings.bedrock_endpoint_verify_ssl
        if not self.verify_ssl:
            # Disable SSL warnings when verification is disabled
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("SSL verification disabled for Bedrock endpoint", endpoint=self.invoke_url)
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        logger.info("Initialized Bedrock endpoint service", 
                   endpoint=self.invoke_url, 
                   verify_ssl=self.verify_ssl)
    
    def invoke_model(
        self,
        model_id: str,
        query: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        top_k: int = 250,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Invoke a Bedrock model through the HTTP endpoint.
        
        Args:
            model_id: Bedrock model ID (e.g., "us.anthropic.claude-3-haiku-20240307-v1:0")
            query: Input text/prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            stop_sequences: List of stop sequences
            
        Returns:
            Dictionary with model response
        """
        if stop_sequences is None:
            stop_sequences = ["\n\nHuman"]
        
        payload = {
            "model_id": model_id,
            "query": query,
            "model_kwargs": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "stop_sequences": stop_sequences
            },
            "invoke_type": "llm"
        }
        
        try:
            logger.debug("Invoking Bedrock model via endpoint", 
                        model_id=model_id,
                        query_length=len(query),
                        endpoint=self.invoke_url)
            
            response = self.session.post(
                self.invoke_url,
                json=payload,
                timeout=60,  # 60 second timeout
                verify=self.verify_ssl  # Use SSL verification setting
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.debug("Bedrock endpoint response received",
                        model_id=model_id,
                        status_code=response.status_code,
                        response_keys=list(result.keys()) if isinstance(result, dict) else "non-dict")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error("Bedrock endpoint request failed",
                        error=str(e),
                        model_id=model_id,
                        endpoint=self.invoke_url)
            raise
        except json.JSONDecodeError as e:
            logger.error("Failed to decode Bedrock endpoint response",
                        error=str(e),
                        response_text=response.text[:500] if 'response' in locals() else "N/A")
            raise
    
    def invoke_embedding(
        self,
        model_id: str,
        query: str
    ) -> Dict[str, Any]:
        """
        Invoke a Bedrock embedding model through the HTTP endpoint.
        
        Args:
            model_id: Bedrock embedding model ID (e.g., "amazon.titan-embed-text-v2:0")
            query: Input text to embed
            
        Returns:
            Dictionary with embedding response
        """
        payload = {
            "model_id": model_id,
            "query": query,
            "invoke_type": "embedding"
        }
        
        try:
            logger.debug("Invoking Bedrock embedding via endpoint", 
                        model_id=model_id,
                        query_length=len(query),
                        endpoint=self.invoke_url)
            
            response = self.session.post(
                self.invoke_url,
                json=payload,
                timeout=60,  # 60 second timeout
                verify=self.verify_ssl  # Use SSL verification setting
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.debug("Bedrock embedding endpoint response received",
                        model_id=model_id,
                        status_code=response.status_code,
                        response_keys=list(result.keys()) if isinstance(result, dict) else "non-dict")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error("Bedrock embedding endpoint request failed",
                        error=str(e),
                        model_id=model_id,
                        endpoint=self.invoke_url)
            raise
        except json.JSONDecodeError as e:
            logger.error("Failed to decode Bedrock embedding endpoint response",
                        error=str(e),
                        response_text=response.text[:500] if 'response' in locals() else "N/A")
            raise
    
    def generate_text(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1
    ) -> str:
        """
        Generate text using Claude model via endpoint.
        
        Args:
            prompt: Input prompt
            model_id: Model ID (defaults to Claude 3 Haiku)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text string
        """
        if model_id is None:
            model_id = "us.anthropic.claude-3-haiku-20240307-v1:0"
        
        response = self.invoke_model(
            model_id=model_id,
            query=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Extract text from response - adjust based on actual response format
        if isinstance(response, dict):
            # Common response formats to try (including 'result' from your endpoint)
            for key in ['result', 'text', 'content', 'response', 'output', 'completion']:
                if key in response:
                    return str(response[key])
            
            # If none of the common keys found, log and return string representation
            logger.warning("Unexpected response format from Bedrock endpoint",
                          response_keys=list(response.keys()),
                          response_sample=str(response)[:200])
            return str(response)
        
        return str(response)


class BedrockEndpointEmbeddingService:
    """Service for generating embeddings through Bedrock endpoint."""
    
    def __init__(self, endpoint_service: BedrockEndpointService):
        """Initialize embedding service with endpoint service."""
        self.bedrock_service = endpoint_service
        self.embedding_model = settings.aws.embedding_model
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            try:
                # Use endpoint for embeddings with dedicated embedding method
                response = self.bedrock_service.invoke_embedding(
                    model_id=self.embedding_model,
                    query=text
                )
                
                # Extract embedding from response - adjust based on actual format
                if isinstance(response, dict):
                    # Try common embedding response formats (including 'result' from your endpoint)
                    for key in ['result', 'embedding', 'embeddings', 'vector']:
                        if key in response:
                            embedding_data = response[key]
                            if isinstance(embedding_data, list) and len(embedding_data) > 0:
                                embeddings.append(embedding_data)
                                break
                    else:
                        logger.error("Unexpected embedding response format from endpoint", 
                                    response_keys=list(response.keys()) if isinstance(response, dict) else "non-dict",
                                    response_sample=str(response)[:200])
                        # Return zero vector as fallback
                        embeddings.append([0.0] * settings.opensearch.vector_size)
                else:
                    logger.error("Non-dict embedding response from endpoint", 
                                response_type=type(response).__name__,
                                response_sample=str(response)[:200])
                    # Return zero vector as fallback
                    embeddings.append([0.0] * settings.opensearch.vector_size)
                    
            except Exception as e:
                logger.error("Endpoint embedding failed", error=str(e), text_length=len(text))
                # Return zero vector as fallback
                embeddings.append([0.0] * settings.opensearch.vector_size)
        
        return embeddings
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.get_embeddings([text])[0]


class BedrockEndpointLLMWrapper:
    """LLM wrapper compatible with existing LLM interfaces."""
    
    def __init__(self, endpoint_service: BedrockEndpointService, model_id: Optional[str] = None):
        """Initialize LLM wrapper."""
        self.bedrock_service = endpoint_service
        self.model_id = model_id or settings.aws.llm_model
    
    def complete(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
        **kwargs
    ) -> 'CompletionResponse':
        """Complete text using the endpoint (compatible with LlamaIndex interface)."""
        max_tokens = max_tokens or 2048
        
        text = self.bedrock_service.generate_text(
            prompt=prompt,
            model_id=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return CompletionResponse(text=text)
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Simple text generation method."""
        return self.bedrock_service.generate_text(prompt=prompt, model_id=self.model_id, **kwargs)


class CompletionResponse:
    """Simple response object compatible with LlamaIndex."""
    
    def __init__(self, text: str):
        self.text = text
        self.raw = {"text": text}
    
    def __str__(self):
        return self.text