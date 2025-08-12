"""Service for executing SQL queries via external API."""

import requests
from typing import Dict, Any, Optional
import structlog
from datetime import datetime

from ..config.settings import settings

logger = structlog.get_logger(__name__)


class QueryExecutionService:
    """Service for executing SQL queries through external database API."""
    
    def __init__(self, execution_api_url: Optional[str] = None):
        self.execution_api_url = execution_api_url or settings.app.execution_api_url
        self.timeout = 30.0
        
    def execute_query(
        self,
        sql_query: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute SQL query via external API."""
        start_time = datetime.utcnow()
        
        request_payload = {
            "query": sql_query,
            "session_id": session_id,
            "metadata": metadata or {}
        }
        
        try:
            response = requests.post(
                f"{self.execution_api_url}/execute/query",
                json=request_payload,
                timeout=self.timeout
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                result = response.json()
                result["execution_time_ms"] = execution_time
                result["query_executed"] = sql_query
                
                logger.info(
                    "Query executed successfully",
                    execution_time_ms=execution_time,
                    rows_returned=len(result.get("data", [])),
                    session_id=session_id
                )
                
                return result
                
            else:
                error_detail = "Unknown error"
                try:
                    error_response = response.json()
                    error_detail = error_response.get("detail", error_detail)
                except:
                    error_detail = response.text
                
                logger.error(
                    "Query execution failed",
                    status_code=response.status_code,
                    error=error_detail,
                    sql_query=sql_query
                )
                
                return {
                    "success": False,
                    "error": error_detail,
                    "error_code": response.status_code,
                    "query_executed": sql_query,
                    "execution_time_ms": execution_time
                }
                    
        except requests.Timeout:
            logger.error("Query execution timeout", sql_query=sql_query, timeout=self.timeout)
            return {
                "success": False,
                "error": f"Query execution timeout after {self.timeout}s",
                "error_code": "TIMEOUT",
                "query_executed": sql_query
            }
            
        except Exception as e:
            logger.error("Query execution error", error=str(e), sql_query=sql_query)
            return {
                "success": False,
                "error": f"Execution service error: {str(e)}",
                "error_code": "SERVICE_ERROR",
                "query_executed": sql_query
            }
    
    def validate_query(self, sql_query: str) -> Dict[str, Any]:
        """Validate SQL query without executing it."""
        request_payload = {
            "query": sql_query,
            "validate_only": True
        }
        
        try:
            response = requests.post(
                f"{self.execution_api_url}/validate/query",
                json=request_payload,
                timeout=10.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_response = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"error": response.text}
                return {
                    "valid": False,
                    "error": error_response.get("detail", "Validation failed"),
                    "error_code": response.status_code
                }
                    
        except Exception as e:
            logger.error("Query validation error", error=str(e))
            return {
                "valid": False,
                "error": f"Validation service error: {str(e)}",
                "error_code": "SERVICE_ERROR"
            }
    
    def get_schema_info(self, table_names: Optional[list] = None) -> Dict[str, Any]:
        """Get schema information from the database."""
        request_payload = {}
        if table_names:
            request_payload["tables"] = table_names
            
        try:
            response = requests.post(
                f"{self.execution_api_url}/schema/info",
                json=request_payload,
                timeout=15.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": "Failed to retrieve schema information"
                }
                    
        except Exception as e:
            logger.error("Schema info retrieval error", error=str(e))
            return {
                "success": False,
                "error": f"Schema service error: {str(e)}"
            }
    
    def health_check(self) -> bool:
        """Check if the execution service is available."""
        try:
            response = requests.get(f"{self.execution_api_url}/health", timeout=5.0)
            return response.status_code == 200
        except:
            return False