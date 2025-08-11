"""Bedrock integration using HTTP endpoint."""

import json
import httpx
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BedrockEndpointService:
    """Service for interacting with Bedrock through HTTP endpoint."""
    
    def __init__(self, endpoint_url: str, embedding_model: str, llm_model: str):
        """
        Initialize Bedrock endpoint service.
        
        Args:
            endpoint_url: Full URL for the Bedrock endpoint
            embedding_model: Model ID for embeddings
            llm_model: Model ID for LLM
        """
        self.endpoint_url = endpoint_url.rstrip('/')
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        logger.info(f"Initialized Bedrock endpoint service at {self.endpoint_url}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        try:
            payload = {
                "model_id": self.embedding_model,
                "invoke_type": "embedding", 
                "query": text
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.endpoint_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Extract embedding from response
                if "embedding" in result:
                    return result["embedding"]
                elif "body" in result:
                    body_data = json.loads(result["body"]) if isinstance(result["body"], str) else result["body"]
                    return body_data.get("embedding", [])
                else:
                    logger.error(f"Unexpected embedding response format: {result}")
                    raise ValueError("Could not extract embedding from response")
                    
        except httpx.HTTPError as e:
            logger.error(f"HTTP error getting embedding: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
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
            
            async with httpx.AsyncClient(timeout=60.0) as client:
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
            raise
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
            
            async with httpx.AsyncClient(timeout=60.0) as client:
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
            raise
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


class BedrockEndpointLLMWrapper:
    """Wrapper class for LLM provider factory compatibility."""
    
    def __init__(self, endpoint_service: BedrockEndpointService, model_id: str = None):
        self.endpoint_service = endpoint_service
        self.model_id = model_id or endpoint_service.llm_model
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate text response using the endpoint service."""
        return await self.endpoint_service.generate_text(prompt)
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Complete method for LLM compatibility."""
        return await self.endpoint_service.generate_text(prompt)
    
    def health_check(self) -> bool:
        """Check if the service is healthy."""
        try:
            # Simple test to check if service is responsive
            import asyncio
            async def test_health():
                try:
                    result = await self.endpoint_service.generate_text("Test")
                    return bool(result)
                except:
                    return False
            
            return asyncio.run(test_health())
        except:
            return False