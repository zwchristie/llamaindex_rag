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
        """Generate SQL using LLM with structured prompt."""
        prompt = f"""You are a SQL expert. Generate a SQL query based on the user's question and the provided database context.

User Question: {user_query}

Database Context:
{context}

Requirements:
1. Generate a valid SQL query that answers the user's question
2. Use only the tables and columns mentioned in the context
3. Include appropriate WHERE clauses, JOINs, and filters
4. Provide a brief explanation of what the query does

Response format:
SQL: [your SQL query here]
EXPLANATION: [brief explanation of what the query does]"""

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