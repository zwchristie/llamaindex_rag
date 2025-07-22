"""AWS Bedrock integration for embeddings and LLM services."""

import boto3
import json
from typing import List, Dict, Any, Optional
from botocore.exceptions import ClientError
import structlog

from ..config.settings import settings

logger = structlog.get_logger(__name__)


class BedrockEmbeddingService:
    """Service for generating embeddings using AWS Bedrock."""
    
    def __init__(self):
        self.client = self._create_bedrock_client()
        self.model_id = settings.aws.embedding_model
        
    def _create_bedrock_client(self):
        """Create AWS Bedrock client with proper configuration."""
        try:
            if settings.aws.use_profile and settings.aws.profile_name:
                # Use AWS profile (for local development)
                logger.info("Using AWS profile for Bedrock client", profile=settings.aws.profile_name)
                session = boto3.Session(
                    profile_name=settings.aws.profile_name,
                    region_name=settings.aws.region
                )
            elif settings.aws.access_key_id and settings.aws.secret_access_key:
                # Use explicit credentials (for production)
                logger.info("Using explicit AWS credentials for Bedrock client")
                session = boto3.Session(
                    aws_access_key_id=settings.aws.access_key_id,
                    aws_secret_access_key=settings.aws.secret_access_key,
                    aws_session_token=settings.aws.session_token,
                    region_name=settings.aws.region
                )
            else:
                # Use default credential chain (IAM roles, etc.)
                logger.info("Using default AWS credential chain for Bedrock client")
                session = boto3.Session(region_name=settings.aws.region)
            
            return session.client('bedrock-runtime')
        except Exception as e:
            logger.error("Failed to create Bedrock client", error=str(e))
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.get_embeddings([text])[0]
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        
        for text in texts:
            try:
                embedding = self._call_embedding_model(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error("Failed to get embedding", text_length=len(text), error=str(e))
                # Return zero vector as fallback
                embeddings.append([0.0] * settings.opensearch.vector_size)
        
        return embeddings
    
    def _call_embedding_model(self, text: str) -> List[float]:
        """Call the embedding model API."""
        if self.model_id.startswith("amazon.titan-embed"):
            return self._call_titan_embedding(text)
        elif self.model_id.startswith("cohere.embed"):
            return self._call_cohere_embedding(text)
        else:
            raise ValueError(f"Unsupported embedding model: {self.model_id}")
    
    def _call_titan_embedding(self, text: str) -> List[float]:
        """Call Amazon Titan embedding model."""
        body = json.dumps({
            "inputText": text
        })
        
        try:
            response = self.client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            return response_body.get('embedding', [])
            
        except ClientError as e:
            logger.error("Bedrock API error", model_id=self.model_id, error=str(e))
            raise
    
    def _call_cohere_embedding(self, text: str) -> List[float]:
        """Call Cohere embedding model."""
        body = json.dumps({
            "texts": [text],
            "input_type": "search_document"
        })
        
        try:
            response = self.client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            embeddings = response_body.get('embeddings', [])
            return embeddings[0] if embeddings else []
            
        except ClientError as e:
            logger.error("Bedrock API error", model_id=self.model_id, error=str(e))
            raise


class BedrockLLMService:
    """Service for text generation using AWS Bedrock LLMs."""
    
    def __init__(self):
        self.client = self._create_bedrock_client()
        self.model_id = settings.aws.llm_model
        
    def _create_bedrock_client(self):
        """Create AWS Bedrock client with proper configuration."""
        try:
            if settings.aws.use_profile and settings.aws.profile_name:
                # Use AWS profile (for local development)
                logger.info("Using AWS profile for Bedrock client", profile=settings.aws.profile_name)
                session = boto3.Session(
                    profile_name=settings.aws.profile_name,
                    region_name=settings.aws.region
                )
            elif settings.aws.access_key_id and settings.aws.secret_access_key:
                # Use explicit credentials (for production)
                logger.info("Using explicit AWS credentials for Bedrock client")
                session = boto3.Session(
                    aws_access_key_id=settings.aws.access_key_id,
                    aws_secret_access_key=settings.aws.secret_access_key,
                    aws_session_token=settings.aws.session_token,
                    region_name=settings.aws.region
                )
            else:
                # Use default credential chain (IAM roles, etc.)
                logger.info("Using default AWS credential chain for Bedrock client")
                session = boto3.Session(region_name=settings.aws.region)
            
            return session.client('bedrock-runtime')
        except Exception as e:
            logger.error("Failed to create Bedrock client", error=str(e))
            raise
    
    def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate text using the configured LLM."""
        try:
            if self.model_id.startswith("anthropic.claude"):
                return self._call_claude_model(prompt, max_tokens, temperature, top_p, stop_sequences)
            elif self.model_id.startswith("amazon.titan"):
                return self._call_titan_model(prompt, max_tokens, temperature, top_p, stop_sequences)
            elif self.model_id.startswith("meta.llama"):
                return self._call_llama_model(prompt, max_tokens, temperature, top_p, stop_sequences)
            else:
                raise ValueError(f"Unsupported LLM model: {self.model_id}")
        except Exception as e:
            logger.error("Failed to generate text", model_id=self.model_id, error=str(e))
            raise
    
    def _call_claude_model(
        self, 
        prompt: str, 
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[List[str]]
    ) -> str:
        """Call Anthropic Claude model."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        if stop_sequences:
            body["stop_sequences"] = stop_sequences
        
        try:
            response = self.client.invoke_model(
                body=json.dumps(body),
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            content = response_body.get('content', [])
            
            if content and len(content) > 0:
                return content[0].get('text', '')
            
            return ''
            
        except ClientError as e:
            logger.error("Claude API error", model_id=self.model_id, error=str(e))
            raise
    
    def _call_titan_model(
        self, 
        prompt: str, 
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[List[str]]
    ) -> str:
        """Call Amazon Titan model."""
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": top_p
            }
        }
        
        if stop_sequences:
            body["textGenerationConfig"]["stopSequences"] = stop_sequences
        
        try:
            response = self.client.invoke_model(
                body=json.dumps(body),
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            results = response_body.get('results', [])
            
            if results and len(results) > 0:
                return results[0].get('outputText', '')
            
            return ''
            
        except ClientError as e:
            logger.error("Titan API error", model_id=self.model_id, error=str(e))
            raise
    
    def _call_llama_model(
        self, 
        prompt: str, 
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[List[str]]
    ) -> str:
        """Call Meta Llama model."""
        body = {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        try:
            response = self.client.invoke_model(
                body=json.dumps(body),
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            return response_body.get('generation', '')
            
        except ClientError as e:
            logger.error("Llama API error", model_id=self.model_id, error=str(e))
            raise
    
    def generate_sql_query(
        self, 
        natural_language_query: str,
        schema_context: str,
        example_queries: Optional[List[str]] = None
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
                temperature=0.1,
                stop_sequences=["\n\n"]
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
        
        # Look for SQL keywords
        sql_pattern = re.compile(
            r'(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP).*?(?=\n\n|\Z)',
            re.IGNORECASE | re.DOTALL
        )
        
        matches = sql_pattern.findall(text)
        if matches:
            return matches[0].strip()
        
        return text.strip()