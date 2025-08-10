"""
Settings configuration for the simplified text-to-SQL system.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application settings
    app_debug: bool = Field(default=True, env="APP_DEBUG")
    app_title: str = Field(default="Text-to-SQL RAG API", env="APP_TITLE")
    app_description: str = Field(
        default="Simplified text-to-SQL system with HITL approval", 
        env="APP_DESCRIPTION"
    )
    app_version: str = Field(default="2.0.0", env="APP_VERSION")
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    
    # MongoDB settings
    mongodb_url: str = Field(
        default="mongodb://admin:password@localhost:27017",
        env="MONGODB_URL"
    )
    mongodb_database: str = Field(
        default="text_to_sql_rag", 
        env="MONGODB_DATABASE"
    )
    
    # OpenSearch settings
    opensearch_host: str = Field(default="localhost", env="OPENSEARCH_HOST")
    opensearch_port: int = Field(default=9200, env="OPENSEARCH_PORT")
    opensearch_use_ssl: bool = Field(default=False, env="OPENSEARCH_USE_SSL")
    opensearch_verify_certs: bool = Field(default=False, env="OPENSEARCH_VERIFY_CERTS")
    opensearch_index_name: str = Field(
        default="view_metadata", 
        env="OPENSEARCH_INDEX_NAME"
    )
    opensearch_vector_field: str = Field(
        default="embedding", 
        env="OPENSEARCH_VECTOR_FIELD"
    )
    opensearch_vector_size: int = Field(
        default=1536, 
        env="OPENSEARCH_VECTOR_SIZE"
    )
    
    # Bedrock API Gateway settings
    bedrock_endpoint_url: str = Field(
        default="https://8v1n9dbomk.execute-api.us-east-1.amazonaws.com/testaccess",
        env="BEDROCK_ENDPOINT_URL"
    )
    bedrock_llm_model: str = Field(
        default="anthropic.claude-3-haiku-20240307-v1:0",
        env="BEDROCK_LLM_MODEL"
    )
    bedrock_embedding_model: str = Field(
        default="amazon.titan-embed-text-v2:0",
        env="BEDROCK_EMBEDDING_MODEL"
    )
    use_mock_embeddings: bool = Field(
        default=False,
        env="USE_MOCK_EMBEDDINGS"
    )
    
    # Redis settings (optional)
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # HITL settings
    hitl_timeout_minutes: int = Field(default=30, env="HITL_TIMEOUT_MINUTES")
    max_pending_requests: int = Field(default=100, env="MAX_PENDING_REQUESTS")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_docs: bool = Field(default=True, env="ENABLE_DOCS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings