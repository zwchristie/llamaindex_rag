"""LLM provider factory for switching between Bedrock and custom LLM services."""

from typing import Union, Optional, Dict, Any, List
import structlog

from ..config.settings import settings
from .bedrock_service import BedrockLLMService
from .custom_llm_service import CustomLLMService

logger = structlog.get_logger(__name__)


class LLMProviderFactory:
    """Factory for creating and managing LLM providers."""
    
    def __init__(self):
        self._bedrock_service = None
        self._custom_service = None
        self._current_provider = None
        
        # Initialize the provider based on configuration
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the appropriate LLM provider based on configuration."""
        provider_name = settings.llm_provider.provider.lower()
        
        try:
            if provider_name == "bedrock":
                self._current_provider = self._get_bedrock_service()
                logger.info("Initialized Bedrock LLM provider")
            elif provider_name == "custom":
                self._current_provider = self._get_custom_service()
                logger.info("Initialized Custom LLM provider")
            else:
                raise ValueError(f"Unknown LLM provider: {provider_name}")
                
        except Exception as e:
            logger.error("Failed to initialize LLM provider", provider=provider_name, error=str(e))
            # Fall back to Bedrock if custom provider fails
            if provider_name == "custom":
                logger.warning("Falling back to Bedrock LLM provider")
                try:
                    self._current_provider = self._get_bedrock_service()
                except Exception as bedrock_error:
                    logger.error("Failed to initialize fallback Bedrock provider", error=str(bedrock_error))
                    raise bedrock_error
            else:
                raise e
    
    def _get_bedrock_service(self) -> BedrockLLMService:
        """Get or create Bedrock LLM service."""
        if self._bedrock_service is None:
            self._bedrock_service = BedrockLLMService()
        return self._bedrock_service
    
    def _get_custom_service(self) -> CustomLLMService:
        """Get or create custom LLM service."""
        if self._custom_service is None:
            if not settings.custom_llm:
                raise ValueError("Custom LLM settings not configured")
            self._custom_service = CustomLLMService()
        return self._custom_service
    
    def get_current_provider(self) -> Union[BedrockLLMService, CustomLLMService]:
        """Get the current LLM provider."""
        if self._current_provider is None:
            raise RuntimeError("No LLM provider initialized")
        return self._current_provider
    
    def switch_provider(self, provider_name: str) -> bool:
        """Switch to a different LLM provider."""
        try:
            provider_name = provider_name.lower()
            
            if provider_name == "bedrock":
                self._current_provider = self._get_bedrock_service()
                logger.info("Switched to Bedrock LLM provider")
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
        if isinstance(self._current_provider, BedrockLLMService):
            return {
                "provider": "bedrock",
                "model": settings.aws.llm_model,
                "region": settings.aws.region,
                "using_profile": settings.aws.use_profile,
                "profile_name": settings.aws.profile_name if settings.aws.use_profile else None
            }
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
    def generate_text(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using the current provider."""
        provider = self.get_current_provider()
        
        if isinstance(provider, CustomLLMService):
            return provider.generate_text(prompt, conversation_id=conversation_id, **kwargs)
        elif isinstance(provider, BedrockLLMService):
            # BedrockLLMService has different parameter names
            return provider.generate_text(prompt, **kwargs)
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
        elif isinstance(provider, BedrockLLMService):
            return provider.generate_sql_query(
                natural_language_query=natural_language_query,
                schema_context=schema_context,
                example_queries=example_queries
            )
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
        elif isinstance(provider, BedrockLLMService):
            # Bedrock doesn't have conversation continuation, so just generate new text
            logger.warning("Conversation continuation not supported with Bedrock, using generate_text")
            return provider.generate_text(message, **kwargs)
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
        elif isinstance(provider, BedrockLLMService):
            # Bedrock doesn't support conversation retrieval
            logger.warning("Conversation message retrieval not supported with Bedrock")
            return []
        else:
            raise RuntimeError(f"Unknown provider type: {type(provider)}")


# Global LLM provider factory instance
llm_factory = LLMProviderFactory()