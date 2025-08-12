"""Enhanced Bedrock integration with SSL, HTTP auth, and consolidated functionality."""

import json
import httpx
import ssl
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)


class EnhancedBedrockService:
    """Consolidated Bedrock service with enhanced features for HTTP endpoint connectivity."""
    
    def __init__(
        self, 
        endpoint_url: str, 
        embedding_model: str, 
        llm_model: str,
        verify_ssl: bool = True,
        ssl_cert_file: Optional[str] = None,
        ssl_key_file: Optional[str] = None,
        ssl_ca_file: Optional[str] = None,
        http_auth_username: Optional[str] = None,
        http_auth_password: Optional[str] = None
    ):
        """
        Initialize enhanced Bedrock service.
        
        Args:
            endpoint_url: Full URL for the Bedrock endpoint
            embedding_model: Model ID for embeddings
            llm_model: Model ID for LLM
            verify_ssl: Whether to verify SSL certificates
            ssl_cert_file: Path to SSL certificate file
            ssl_key_file: Path to SSL key file
            ssl_ca_file: Path to SSL CA file
            http_auth_username: HTTP basic auth username
            http_auth_password: HTTP basic auth password
        """
        self.endpoint_url = endpoint_url.rstrip('/')
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.verify_ssl = verify_ssl
        self.ssl_cert_file = ssl_cert_file
        self.ssl_key_file = ssl_key_file
        self.ssl_ca_file = ssl_ca_file
        self.http_auth_username = http_auth_username
        self.http_auth_password = http_auth_password
        self.embedding_dimension = None  # Will be detected dynamically
        
        # Prepare SSL context if certificates are provided
        self.ssl_context = self._create_ssl_context()
        
        logger.info(f"Initialized Enhanced Bedrock service at {self.endpoint_url}")
        logger.info(f"SSL verification: {verify_ssl}")
        if http_auth_username:
            logger.info(f"HTTP Auth enabled for user: {http_auth_username}")
    
    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context with optional certificates."""
        if not self.verify_ssl:
            # Create context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            logger.info("SSL verification disabled")
            return ssl_context
        
        # Default context for SSL verification
        ssl_context = ssl.create_default_context()
        
        # Load custom CA file if provided
        if self.ssl_ca_file and Path(self.ssl_ca_file).exists():
            ssl_context.load_verify_locations(cafile=self.ssl_ca_file)
            logger.info(f"Loaded SSL CA file: {self.ssl_ca_file}")
        
        # Load client certificate if provided
        if self.ssl_cert_file and self.ssl_key_file:
            if Path(self.ssl_cert_file).exists() and Path(self.ssl_key_file).exists():
                ssl_context.load_cert_chain(self.ssl_cert_file, self.ssl_key_file)
                logger.info(f"Loaded SSL client certificate: {self.ssl_cert_file}")
            else:
                logger.warning(f"SSL certificate files not found: {self.ssl_cert_file}, {self.ssl_key_file}")
        
        return ssl_context
    
    def _get_http_auth(self) -> Optional[httpx.BasicAuth]:
        """Get HTTP basic auth if configured."""
        if self.http_auth_username and self.http_auth_password:
            return httpx.BasicAuth(self.http_auth_username, self.http_auth_password)
        return None
    
    def _create_http_client(self, timeout: float = 30.0) -> httpx.AsyncClient:
        """Create configured HTTP client with SSL and auth settings."""
        client_kwargs = {
            "timeout": timeout,
            "verify": self.ssl_context if self.ssl_context else self.verify_ssl,
            "follow_redirects": True
        }
        
        # Add HTTP auth if configured
        auth = self._get_http_auth()
        if auth:
            client_kwargs["auth"] = auth
        
        return httpx.AsyncClient(**client_kwargs)
    
    # Embedding functionality
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        try:
            payload = {
                "model_id": self.embedding_model,
                "invoke_type": "embedding", 
                "query": text
            }
            
            async with self._create_http_client(timeout=30.0) as client:
                response = await client.post(
                    self.endpoint_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Extract embedding from response
                if "embedding" in result:
                    embedding = result["embedding"]
                elif "body" in result:
                    body_data = json.loads(result["body"]) if isinstance(result["body"], str) else result["body"]
                    embedding = body_data.get("embedding", [])
                else:
                    logger.error(f"Unexpected embedding response format: {result}")
                    raise ValueError("Could not extract embedding from response")
                
                # Detect embedding dimension on first call
                if self.embedding_dimension is None:
                    self.embedding_dimension = len(embedding)
                    logger.info(f"Detected embedding dimension: {self.embedding_dimension}")
                
                return embedding
                    
        except httpx.HTTPError as e:
            logger.error(f"HTTP error getting embedding: {e}")
            # Add more specific error handling for common issues
            if e.response.status_code == 404:
                raise ValueError(f"Bedrock endpoint not found: {self.endpoint_url}")
            elif e.response.status_code == 401:
                raise ValueError("Unauthorized: Check your authentication credentials")
            elif e.response.status_code == 403:
                raise ValueError("Forbidden: Check your permissions for the Bedrock model")
            raise
        except ssl.SSLError as e:
            logger.error(f"SSL error: {e}")
            raise ValueError(f"SSL connection failed: {e}. Try setting verify_ssl=False for testing.")
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    async def get_embeddings_batch(self, texts: List[str], batch_size: int = 5) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            # Process batch concurrently
            tasks = [self.get_embedding(text) for text in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error embedding text {i+j}: {result}")
                    # Use zero vector as fallback
                    batch_embeddings.append([0.0] * (self.embedding_dimension or 1536))
                else:
                    batch_embeddings.append(result)
            
            embeddings.extend(batch_embeddings)
            
            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return embeddings
    
    def get_embedding_dimension(self) -> Optional[int]:
        """Get the embedding dimension (detected after first embedding call)."""
        return self.embedding_dimension
    
    # LLM functionality
    async def generate_sql(self, user_query: str, context: str) -> Dict[str, str]:
        """Generate Oracle SQL using LLM with hierarchical context and Oracle dialect."""
        prompt = f"""You are an Oracle SQL expert. Generate an Oracle SQL query based on the user's question and the provided hierarchical database context.

User Question: {user_query}

Hierarchical Database Context:
{context}

ORACLE SQL REQUIREMENTS:
1. Generate a valid Oracle SQL query that answers the user's question
2. Use only the tables/views and columns mentioned in the context
3. Schema-qualify all table references (e.g., SCHEMA.TABLE_NAME)
4. Use Oracle-specific syntax and functions:
   - Oracle date functions: SYSDATE, TO_DATE, TO_CHAR
   - Use ROWNUM for row limiting (not LIMIT)
   - Use Oracle outer join syntax: (+) or ANSI JOIN
   - Use UPPER() for case-insensitive string comparisons
5. For lookup columns with LOOKUP_ID references:
   - Use the numeric ID values in WHERE clauses, not the text names
   - Reference the lookup values provided in the context
6. Include appropriate WHERE clauses, JOINs, and filters
7. Use Oracle data types (VARCHAR2, NUMBER, DATE, etc.)
8. Follow the SQL examples from reports when applicable

Response format:
SQL: [your Oracle SQL query here]
EXPLANATION: [brief explanation of what the query does and why specific Oracle features were used]"""

        try:
            payload = {
                "model_id": self.llm_model,
                "invoke_type": "llm",
                "query": prompt
            }
            
            async with self._create_http_client(timeout=60.0) as client:
                response = await client.post(
                    self.endpoint_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Extract text from response
                if "result" in result:
                    llm_response = result["result"]
                elif "text" in result:
                    llm_response = result["text"]
                elif "body" in result:
                    body_data = json.loads(result["body"]) if isinstance(result["body"], str) else result["body"]
                    llm_response = body_data.get("text", str(result))
                else:
                    logger.warning(f"Unexpected LLM response format: {result}")
                    llm_response = str(result)
                
                # Parse SQL and explanation from response
                return self._parse_sql_response(llm_response)
                    
        except httpx.HTTPError as e:
            logger.error(f"HTTP error generating SQL: {e}")
            # Add more specific error handling
            if e.response.status_code == 404:
                raise ValueError(f"Bedrock endpoint not found: {self.endpoint_url}")
            elif e.response.status_code == 401:
                raise ValueError("Unauthorized: Check your authentication credentials")
            elif e.response.status_code == 403:
                raise ValueError("Forbidden: Check your permissions for the Bedrock model")
            raise
        except ssl.SSLError as e:
            logger.error(f"SSL error: {e}")
            raise ValueError(f"SSL connection failed: {e}. Try setting verify_ssl=False for testing.")
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise
    
    async def generate_response(self, prompt: str) -> str:
        """Generate general text response using LLM (alias for generate_text)."""
        return await self.generate_text(prompt)
    
    async def generate_text(self, prompt: str) -> str:
        """Generate text response using LLM."""
        try:
            payload = {
                "model_id": self.llm_model,
                "invoke_type": "llm",
                "query": prompt
            }
            
            async with self._create_http_client(timeout=60.0) as client:
                response = await client.post(
                    self.endpoint_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Extract text from response
                if "result" in result:
                    return result["result"]
                elif "text" in result:
                    return result["text"]
                elif "body" in result:
                    body_data = json.loads(result["body"]) if isinstance(result["body"], str) else result["body"]
                    return body_data.get("text", str(result))
                else:
                    logger.warning(f"Unexpected LLM response format: {result}")
                    return str(result)
                    
        except httpx.HTTPError as e:
            logger.error(f"HTTP error generating text: {e}")
            # Add more specific error handling
            if e.response.status_code == 404:
                raise ValueError(f"Bedrock endpoint not found: {self.endpoint_url}")
            elif e.response.status_code == 401:
                raise ValueError("Unauthorized: Check your authentication credentials")
            elif e.response.status_code == 403:
                raise ValueError("Forbidden: Check your permissions for the Bedrock model")
            raise
        except ssl.SSLError as e:
            logger.error(f"SSL error: {e}")
            raise ValueError(f"SSL connection failed: {e}. Try setting verify_ssl=False for testing.")
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def _parse_sql_response(self, response: str) -> Dict[str, str]:
        """Parse SQL and explanation from LLM response."""
        sql = ""
        explanation = ""
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('SQL:'):
                current_section = 'sql'
                sql_part = line.replace('SQL:', '').strip()
                if sql_part:
                    sql = sql_part
            elif line.startswith('EXPLANATION:'):
                current_section = 'explanation'
                exp_part = line.replace('EXPLANATION:', '').strip()
                if exp_part:
                    explanation = exp_part
            elif line.startswith('```sql'):
                current_section = 'sql_block'
            elif line.startswith('```') and current_section == 'sql_block':
                current_section = None
            elif current_section == 'sql_block':
                if line:
                    sql += line + '\n'
            elif current_section == 'sql' and line:
                sql += ' ' + line if sql else line
            elif current_section == 'explanation' and line:
                explanation += ' ' + line if explanation else line
        
        # If no structured format found, try to extract SQL from code blocks
        if not sql:
            import re
            sql_matches = re.findall(r'```sql\n(.*?)\n```', response, re.DOTALL)
            if sql_matches:
                sql = sql_matches[0].strip()
        
        # If still no SQL, use the whole response as explanation
        if not sql and not explanation:
            explanation = response.strip()
            sql = "-- Unable to generate SQL from the given context"
        
        return {
            "sql": sql.strip(),
            "explanation": explanation.strip() if explanation else "Generated SQL query for the given requirements."
        }
    
    # Health check and utility methods
    async def health_check(self) -> bool:
        """Check if the service is healthy."""
        try:
            # Simple test to check if service is responsive
            result = await self.generate_text("Test")
            return bool(result)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service configuration information."""
        return {
            "endpoint_url": self.endpoint_url,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "embedding_dimension": self.embedding_dimension,
            "ssl_verification": self.verify_ssl,
            "has_ssl_certs": bool(self.ssl_cert_file and self.ssl_key_file),
            "has_http_auth": bool(self.http_auth_username),
        }


class EnhancedBedrockLLMWrapper:
    """Wrapper class for LLM provider factory compatibility."""
    
    def __init__(self, bedrock_service: EnhancedBedrockService, model_id: str = None):
        self.bedrock_service = bedrock_service
        self.model_id = model_id or bedrock_service.llm_model
        self.endpoint_service = bedrock_service  # For backward compatibility
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate text response using the enhanced Bedrock service."""
        return await self.bedrock_service.generate_text(prompt)
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Complete method for LLM compatibility."""
        return await self.bedrock_service.generate_text(prompt)
    
    def health_check(self) -> bool:
        """Check if the service is healthy."""
        try:
            import asyncio
            return asyncio.run(self.bedrock_service.health_check())
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information."""
        return self.bedrock_service.get_service_info()