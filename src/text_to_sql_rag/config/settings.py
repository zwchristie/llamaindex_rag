"""Application configuration management."""

from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    url: str = Field(default="sqlite:///./text_to_sql_rag.db", env="DATABASE_URL")
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    class Config:
        env_prefix = "DATABASE_"


class OpenSearchSettings(BaseSettings):
    """OpenSearch vector store configuration."""
    
    host: str = Field(default="localhost", env="OPENSEARCH_HOST")
    port: int = Field(default=9200, env="OPENSEARCH_PORT")
    username: Optional[str] = Field(default=None, env="OPENSEARCH_USERNAME")
    password: Optional[str] = Field(default=None, env="OPENSEARCH_PASSWORD")
    use_ssl: bool = Field(default=False, env="OPENSEARCH_USE_SSL")
    verify_certs: bool = Field(default=False, env="OPENSEARCH_VERIFY_CERTS")
    index_name: str = Field(default="documents", env="OPENSEARCH_INDEX_NAME")
    vector_field: str = Field(default="vector", env="OPENSEARCH_VECTOR_FIELD")
    vector_size: int = Field(default=1536, env="OPENSEARCH_VECTOR_SIZE")
    
    class Config:
        env_prefix = "OPENSEARCH_"


class AWSSettings(BaseSettings):
    """AWS Bedrock configuration."""
    
    region: str = Field(default="us-east-1", env="AWS_REGION")
    profile_name: Optional[str] = Field(default=None, env="AWS_PROFILE")
    access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    session_token: Optional[str] = Field(default=None, env="AWS_SESSION_TOKEN")
    use_profile: bool = Field(default=False, env="AWS_USE_PROFILE")
    
    # Model configurations
    embedding_model: str = Field(default="amazon.titan-embed-text-v1", env="AWS_EMBEDDING_MODEL")
    llm_model: str = Field(default="anthropic.claude-3-sonnet-20240229-v1:0", env="AWS_LLM_MODEL")
    
    class Config:
        env_prefix = "AWS_"


class RedisSettings(BaseSettings):
    """Redis configuration for session management."""
    
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")
    
    class Config:
        env_prefix = "REDIS_"


class SecuritySettings(BaseSettings):
    """Security configuration."""
    
    secret_key: str = Field(env="SECRET_KEY")  # No default - must be set via environment
    algorithm: str = Field(default="HS256", env="SECURITY_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="SECURITY_ACCESS_TOKEN_EXPIRE_MINUTES")
    
    class Config:
        env_prefix = ""  # Don't use prefix since we're being explicit


class CustomLLMSettings(BaseSettings):
    """Custom internal LLM API configuration."""
    
    base_url: str = Field(env="CUSTOM_LLM_BASE_URL")
    deployment_id: str = Field(env="CUSTOM_LLM_DEPLOYMENT_ID")
    model_name: Optional[str] = Field(default=None, env="CUSTOM_LLM_MODEL_NAME")
    timeout: int = Field(default=30, env="CUSTOM_LLM_TIMEOUT")
    max_retries: int = Field(default=3, env="CUSTOM_LLM_MAX_RETRIES")
    
    class Config:
        env_prefix = "CUSTOM_LLM_"


class LLMProviderSettings(BaseSettings):
    """LLM Provider configuration."""
    
    provider: str = Field(default="bedrock", env="LLM_PROVIDER")  # "bedrock" or "custom"
    
    class Config:
        env_prefix = "LLM_"


class AppSettings(BaseSettings):
    """Main application configuration."""
    
    title: str = Field(default="Text-to-SQL RAG API", env="APP_TITLE")
    description: str = Field(
        default="Agentic text-to-SQL RAG solution with LlamaIndex and OpenSearch",
        env="APP_DESCRIPTION"
    )
    version: str = Field(default="0.1.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="APP_DEBUG")
    
    # File upload settings
    max_upload_size: int = Field(default=10 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 10MB
    allowed_file_types: List[str] = Field(
        default=["txt", "md", "json", "sql"],
        env="ALLOWED_FILE_TYPES"
    )
    
    # RAG settings
    chunk_size: int = Field(default=1024, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    similarity_top_k: int = Field(default=5, env="SIMILARITY_TOP_K")
    confidence_threshold: float = Field(default=0.7, env="CONFIDENCE_THRESHOLD")
    
    # External API settings
    execution_api_url: str = Field(default="http://localhost:8001", env="EXECUTION_API_URL")
    
    # Meta documents path
    meta_documents_path: str = Field(default="meta_documents", env="META_DOCUMENTS_PATH")
    
    class Config:
        env_prefix = "APP_"


class MongoDBSettings(BaseSettings):
    """MongoDB configuration."""
    
    url: str = Field(default="mongodb://localhost:27017", env="MONGODB_URL")
    database: str = Field(default="text_to_sql_rag", env="MONGODB_DATABASE")
    
    class Config:
        env_prefix = "MONGODB_"


class Settings:
    """Global application settings."""
    
    def __init__(self):
        self.app = AppSettings()
        self.database = DatabaseSettings()
        self.opensearch = OpenSearchSettings()
        self.aws = AWSSettings()
        self.redis = RedisSettings()
        self.security = SecuritySettings()
        self.mongodb = MongoDBSettings()
        self.llm_provider = LLMProviderSettings()
        
        # Only load custom LLM settings if needed
        if self._should_load_custom_llm():
            try:
                self.custom_llm = CustomLLMSettings()
            except Exception:
                self.custom_llm = None
        else:
            self.custom_llm = None
    
    def _should_load_custom_llm(self) -> bool:
        """Check if custom LLM settings should be loaded."""
        return (
            os.getenv("LLM_PROVIDER", "bedrock") == "custom" or 
            os.getenv("CUSTOM_LLM_BASE_URL") is not None
        )
    
    def is_using_custom_llm(self) -> bool:
        """Check if using custom LLM provider."""
        return self.llm_provider.provider == "custom" and self.custom_llm is not None
    
    def is_using_bedrock(self) -> bool:
        """Check if using AWS Bedrock provider."""
        return self.llm_provider.provider == "bedrock"


# Global settings instance
settings = Settings()