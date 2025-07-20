"""Application configuration management."""

from typing import Optional, List
from pydantic import BaseSettings, Field
import os


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    url: str = Field(default="sqlite:///./text_to_sql_rag.db", env="DATABASE_URL")
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    class Config:
        env_prefix = "DATABASE_"


class QdrantSettings(BaseSettings):
    """Qdrant vector store configuration."""
    
    host: str = Field(default="localhost", env="QDRANT_HOST")
    port: int = Field(default=6333, env="QDRANT_PORT")
    api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    collection_name: str = Field(default="documents", env="QDRANT_COLLECTION_NAME")
    vector_size: int = Field(default=1536, env="QDRANT_VECTOR_SIZE")
    
    class Config:
        env_prefix = "QDRANT_"


class AWSSettings(BaseSettings):
    """AWS Bedrock configuration."""
    
    region: str = Field(default="us-east-1", env="AWS_REGION")
    access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    session_token: Optional[str] = Field(default=None, env="AWS_SESSION_TOKEN")
    
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
    
    secret_key: str = Field(default="your-secret-key-change-this", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    class Config:
        env_prefix = "SECURITY_"


class AppSettings(BaseSettings):
    """Main application configuration."""
    
    title: str = Field(default="Text-to-SQL RAG API", env="APP_TITLE")
    description: str = Field(
        default="Agentic text-to-SQL RAG solution with LlamaIndex and Qdrant",
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
    
    # External API settings
    execution_api_url: str = Field(default="http://localhost:8001", env="EXECUTION_API_URL")
    
    class Config:
        env_prefix = "APP_"


class Settings(BaseSettings):
    """Global application settings."""
    
    app: AppSettings = AppSettings()
    database: DatabaseSettings = DatabaseSettings()
    qdrant: QdrantSettings = QdrantSettings()
    aws: AWSSettings = AWSSettings()
    redis: RedisSettings = RedisSettings()
    security: SecuritySettings = SecuritySettings()
    
    class Config:
        case_sensitive = False


# Global settings instance
settings = Settings()