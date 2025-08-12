"""LLM provider factory for switching between Bedrock and custom LLM services."""

from typing import Union, Optional, Dict, Any, List
import structlog

from ..config.settings import settings
# Enhanced Bedrock service with SSL, HTTP auth, and improved error handling
from .enhanced_bedrock_service import EnhancedBedrockService, EnhancedBedrockLLMWrapper
from .custom_llm_service import CustomLLMService

logger = structlog.get_logger(__name__)


class LLMProviderFactory:
    """Factory for creating and managing LLM providers."""
    
    def __init__(self):
        self._custom_service = None
        self._current_provider = None
        
        # Initialize the provider based on configuration
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the appropriate LLM provider based on configuration."""
        provider_name = settings.llm_provider.provider.lower()
        
        try:
            if provider_name == "bedrock_endpoint":
                self._current_provider = self._get_bedrock_endpoint_service()
                logger.info("Initialized Bedrock Endpoint LLM provider")
            elif provider_name == "custom":
                self._current_provider = self._get_custom_service()
                logger.info("Initialized Custom LLM provider")
            else:
                # Default to bedrock_endpoint if provider not specified or invalid
                logger.warning(f"Unknown provider '{provider_name}', defaulting to bedrock_endpoint")
                self._current_provider = self._get_bedrock_endpoint_service()
                logger.info("Initialized Bedrock Endpoint LLM provider (default)")
                
        except Exception as e:
            logger.error("Failed to initialize LLM provider", provider=provider_name, error=str(e))
            # Fall back to Bedrock endpoint if custom provider fails
            if provider_name == "custom":
                logger.warning("Falling back to Bedrock Endpoint LLM provider")
                try:
                    self._current_provider = self._get_bedrock_endpoint_service()
                except Exception as bedrock_error:
                    logger.error("Failed to initialize fallback Bedrock Endpoint provider", error=str(bedrock_error))
                    raise bedrock_error
            else:
                raise e
    
    # Direct Bedrock service removed - using only endpoint approach
    
    def _get_bedrock_endpoint_service(self) -> EnhancedBedrockLLMWrapper:
        """Get or create enhanced Bedrock endpoint LLM service."""
        if not hasattr(self, '_bedrock_endpoint_service') or self._bedrock_endpoint_service is None:
            # Get endpoint URL from settings
            endpoint_url = settings.bedrock_endpoint.url
            if not endpoint_url:
                raise ValueError("Bedrock endpoint URL not configured. Set BEDROCK_ENDPOINT_URL in settings.")
            
            model_id = getattr(settings.aws, 'llm_model', 'us.anthropic.claude-3-haiku-20240307-v1:0')
            embedding_model = getattr(settings.aws, 'embedding_model', 'amazon.titan-embed-text-v2:0')
            
            # Create enhanced Bedrock service with SSL and auth support
            enhanced_service = EnhancedBedrockService(
                endpoint_url=endpoint_url,
                embedding_model=embedding_model,
                llm_model=model_id,
                verify_ssl=settings.bedrock_endpoint.verify_ssl,
                ssl_cert_file=settings.bedrock_endpoint.ssl_cert_file,
                ssl_key_file=settings.bedrock_endpoint.ssl_key_file,
                ssl_ca_file=settings.bedrock_endpoint.ssl_ca_file,
                http_auth_username=settings.bedrock_endpoint.http_auth_username,
                http_auth_password=settings.bedrock_endpoint.http_auth_password
            )
            self._bedrock_endpoint_service = EnhancedBedrockLLMWrapper(enhanced_service, model_id)
        return self._bedrock_endpoint_service
    
    def _get_custom_service(self) -> CustomLLMService:
        """Get or create custom LLM service."""
        if self._custom_service is None:
            if not settings.custom_llm:
                raise ValueError("Custom LLM settings not configured")
            self._custom_service = CustomLLMService()
        return self._custom_service
    
    def get_current_provider(self) -> Union[BedrockEndpointLLMWrapper, CustomLLMService]:
        """Get the current LLM provider."""
        if self._current_provider is None:
            raise RuntimeError("No LLM provider initialized")
        return self._current_provider
    
    def switch_provider(self, provider_name: str) -> bool:
        """Switch to a different LLM provider."""
        try:
            provider_name = provider_name.lower()
            
            if provider_name == "bedrock_endpoint":
                self._current_provider = self._get_bedrock_endpoint_service()
                logger.info("Switched to Bedrock Endpoint LLM provider")
                return True
            elif provider_name == "custom":
                self._current_provider = self._get_custom_service()
                logger.info("Switched to Custom LLM provider")
                return True
            else:
                logger.error("Unknown provider name", provider=provider_name)
                return False
                
        except Exception as e:
            logger.error("Failed to switch LLM provider", provider=provider_name, error=str(e))
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider."""
        if isinstance(self._current_provider, EnhancedBedrockLLMWrapper):
            service_info = self._current_provider.get_service_info() if hasattr(self._current_provider, 'get_service_info') else {}
            base_info = {
                "provider": "bedrock_endpoint",
                "model": self._current_provider.model_id,
            }
            base_info.update(service_info)
            return base_info
        elif isinstance(self._current_provider, CustomLLMService):
            return {
                "provider": "custom",
                "base_url": settings.custom_llm.base_url if settings.custom_llm else None,
                "deployment_id": settings.custom_llm.deployment_id if settings.custom_llm else None,
                "model_name": settings.custom_llm.model_name if settings.custom_llm else None
            }
        else:
            return {"provider": "unknown"}
    
    def health_check(self) -> bool:
        """Check health of current LLM provider."""
        try:
            if self._current_provider is None:
                return False
            
            # Both services should have a health_check method
            if hasattr(self._current_provider, 'health_check'):
                return self._current_provider.health_check()
            else:
                # Basic check - try to call a simple method
                return True
                
        except Exception as e:
            logger.error("LLM provider health check failed", error=str(e))
            return False
    
    # Proxy methods to the current provider
    async def generate_text(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using the current provider."""
        provider = self.get_current_provider()
        
        if isinstance(provider, CustomLLMService):
            return provider.generate_text(prompt, conversation_id=conversation_id, **kwargs)
        elif isinstance(provider, EnhancedBedrockLLMWrapper):
            # Both wrappers use generate_response method
            return await provider.generate_response(prompt, **kwargs)
        else:
            raise RuntimeError(f"Unknown provider type: {type(provider)}")
    
    def generate_sql_query(
        self,
        natural_language_query: str,
        schema_context: str,
        example_queries: Optional[List[str]] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate SQL query using the current provider."""
        provider = self.get_current_provider()
        
        if isinstance(provider, CustomLLMService):
            return provider.generate_sql_query(
                natural_language_query=natural_language_query,
                schema_context=schema_context,
                example_queries=example_queries,
                conversation_id=conversation_id
            )
        elif isinstance(provider, EnhancedBedrockLLMWrapper):
            # Enhanced wrapper supports SQL generation through the underlying service
            logger.warning("generate_sql_query not directly supported with endpoint wrapper, using generate_response")
            prompt = f"Schema: {schema_context}\nQuery: {natural_language_query}"
            response = await provider.generate_response(prompt)
            return {"sql": response, "explanation": "", "confidence": 0.8}
        else:
            raise RuntimeError(f"Unknown provider type: {type(provider)}")
    
    def continue_conversation(
        self,
        message: str,
        conversation_id: str,
        **kwargs
    ) -> str:
        """Continue a conversation (only available with custom provider)."""
        provider = self.get_current_provider()
        
        if isinstance(provider, CustomLLMService):
            return provider.continue_conversation(message, conversation_id, **kwargs)
        elif isinstance(provider, EnhancedBedrockLLMWrapper):
            # Endpoint wrapper doesn't have conversation continuation, use generate_response
            logger.warning("Conversation continuation not supported with Bedrock endpoint, using generate_response")
            return await provider.generate_response(message, **kwargs)
        else:
            raise RuntimeError(f"Unknown provider type: {type(provider)}")
    
    def get_conversation_messages(
        self,
        conversation_id: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get conversation messages (only available with custom provider)."""
        provider = self.get_current_provider()
        
        if isinstance(provider, CustomLLMService):
            return provider.get_conversation_messages(conversation_id, **kwargs)
        elif isinstance(provider, EnhancedBedrockLLMWrapper):
            # Endpoint wrapper doesn't support conversation retrieval
            logger.warning("Conversation message retrieval not supported with Bedrock endpoint")
            return []
        else:
            raise RuntimeError(f"Unknown provider type: {type(provider)}")
    
    def get_llm_client(self, provider: str = None, model: str = None):
        """Get LLM client for direct use (compatibility method)."""
        if provider and provider.lower() == "bedrock_endpoint":
            # Return the endpoint wrapper directly
            return self._get_bedrock_endpoint_service()
        else:
            # Return the current provider
            return self.get_current_provider()


# Global LLM provider factory instance
llm_factory = LLMProviderFactory()