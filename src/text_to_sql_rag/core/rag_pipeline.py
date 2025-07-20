"""RAG pipeline for text-to-SQL query generation."""

from typing import Dict, Any, List, Optional, Tuple
import structlog
from datetime import datetime

from ..services.vector_service import LlamaIndexVectorService
from ..services.document_service import DocumentService
from ..services.query_execution_service import QueryExecutionService
from ..models.document import DocumentType, DocumentSearchRequest
from ..config.settings import settings

logger = structlog.get_logger(__name__)


class TextToSQLRAGPipeline:
    """Main RAG pipeline for text-to-SQL generation."""
    
    def __init__(self, vector_service: LlamaIndexVectorService, document_service: DocumentService):
        self.vector_service = vector_service
        self.document_service = document_service
        self.query_execution_service = QueryExecutionService()
        
    def generate_sql_query(
        self,
        natural_language_query: str,
        session_id: Optional[str] = None,
        use_hybrid_retrieval: bool = True
    ) -> Dict[str, Any]:
        """Generate SQL query from natural language using RAG."""
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Retrieve relevant schema documents
            schema_context = self._retrieve_schema_context(
                natural_language_query,
                use_hybrid_retrieval
            )
            
            # Step 2: Retrieve relevant example queries (report documents)
            example_queries = self._retrieve_example_queries(
                natural_language_query,
                use_hybrid_retrieval
            )
            
            # Step 3: Build comprehensive context
            context = self._build_query_context(
                natural_language_query,
                schema_context,
                example_queries
            )
            
            # Step 4: Generate SQL using LLM with context
            sql_result = self._generate_sql_with_context(
                natural_language_query,
                context
            )
            
            # Step 5: Post-process and validate
            result = self._post_process_result(
                sql_result,
                context,
                natural_language_query
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            result["processing_time_ms"] = processing_time
            result["session_id"] = session_id
            
            logger.info(
                "Generated SQL query",
                query_length=len(natural_language_query),
                processing_time_ms=processing_time,
                confidence=result.get("confidence", 0.0)
            )
            
            return result
            
        except Exception as e:
            logger.error("Failed to generate SQL query", error=str(e))
            return {
                "sql_query": "",
                "explanation": f"Error generating query: {str(e)}",
                "confidence": 0.0,
                "error": str(e),
                "session_id": session_id
            }
    
    def _retrieve_schema_context(
        self,
        query: str,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant schema information."""
        try:
            retriever_type = "hybrid" if use_hybrid else "vector"
            
            schema_results = self.vector_service.search_similar(
                query=query,
                retriever_type=retriever_type,
                similarity_top_k=5,
                document_type=DocumentType.SCHEMA.value
            )
            
            return schema_results
            
        except Exception as e:
            logger.error("Failed to retrieve schema context", error=str(e))
            return []
    
    def _retrieve_example_queries(
        self,
        query: str,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant example queries from report documents."""
        try:
            retriever_type = "hybrid" if use_hybrid else "vector"
            
            example_results = self.vector_service.search_similar(
                query=query,
                retriever_type=retriever_type,
                similarity_top_k=3,
                document_type=DocumentType.REPORT.value
            )
            
            return example_results
            
        except Exception as e:
            logger.error("Failed to retrieve example queries", error=str(e))
            return []
    
    def _build_query_context(
        self,
        natural_query: str,
        schema_context: List[Dict[str, Any]],
        example_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build comprehensive context for SQL generation."""
        
        # Process schema information
        schema_info = []
        table_names = set()
        
        for result in schema_context:
            metadata = result.get("metadata", {})
            content = result.get("content", "")
            
            schema_info.append({
                "content": content,
                "table_name": metadata.get("table_name", ""),
                "relevance_score": result.get("score", 0.0)
            })
            
            if metadata.get("table_name"):
                table_names.add(metadata.get("table_name"))
        
        # Process example queries
        examples = []
        for result in example_queries:
            metadata = result.get("metadata", {})
            content = result.get("content", "")
            
            examples.append({
                "description": metadata.get("title", ""),
                "sql_query": metadata.get("sql_query", ""),
                "content": content,
                "relevance_score": result.get("score", 0.0)
            })
        
        return {
            "natural_query": natural_query,
            "schema_context": schema_info,
            "available_tables": list(table_names),
            "example_queries": examples,
            "num_schema_sources": len(schema_context),
            "num_example_sources": len(example_queries)
        }
    
    def _generate_sql_with_context(
        self,
        natural_query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate SQL query using LLM with full context."""
        
        # Build comprehensive prompt
        prompt_parts = [
            "You are an expert SQL query generator. Given a natural language question and database context, generate an accurate SQL query.",
            "",
            "=== DATABASE SCHEMA CONTEXT ===",
        ]
        
        # Add schema information
        for schema in context["schema_context"]:
            if schema["table_name"]:
                prompt_parts.append(f"Table: {schema['table_name']}")
            prompt_parts.append(schema["content"])
            prompt_parts.append("")
        
        # Add available tables summary
        if context["available_tables"]:
            prompt_parts.extend([
                "Available Tables: " + ", ".join(context["available_tables"]),
                ""
            ])
        
        # Add example queries
        if context["example_queries"]:
            prompt_parts.extend([
                "=== EXAMPLE QUERIES ===",
                "Here are similar queries for reference:",
                ""
            ])
            
            for example in context["example_queries"]:
                if example["sql_query"]:
                    prompt_parts.extend([
                        f"Description: {example['description']}",
                        f"SQL: {example['sql_query']}",
                        ""
                    ])
        
        # Add the actual query
        prompt_parts.extend([
            "=== QUERY TO PROCESS ===",
            f"Natural Language Query: {natural_query}",
            "",
            "Generate a SQL query that answers the question. Respond with JSON in this exact format:",
            '{"sql_query": "SELECT ...", "explanation": "Brief explanation", "confidence": 0.95, "tables_used": ["table1", "table2"]}'
        ])
        
        prompt = "\n".join(prompt_parts)
        
        # Use the vector service's query engine for generation
        try:
            response = self.vector_service.query_with_context(
                query=f"Generate SQL for: {natural_query}",
                retriever_type="hybrid"
            )
            
            # Try to parse structured response
            import json
            try:
                # Look for JSON in the response
                response_text = response.get("response", "")
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    result = json.loads(json_text)
                    
                    # Add source information
                    result["sources"] = response.get("sources", [])
                    return result
                
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Fallback: extract SQL from text response
            return {
                "sql_query": self._extract_sql_from_response(response.get("response", "")),
                "explanation": "Generated from text response",
                "confidence": 0.7,
                "tables_used": list(context["available_tables"]),
                "sources": response.get("sources", [])
            }
            
        except Exception as e:
            logger.error("Failed to generate SQL with LLM", error=str(e))
            return {
                "sql_query": "",
                "explanation": f"Error in SQL generation: {str(e)}",
                "confidence": 0.0,
                "tables_used": [],
                "sources": []
            }
    
    def _extract_sql_from_response(self, response_text: str) -> str:
        """Extract SQL query from plain text response."""
        import re
        
        # Look for SQL keywords and patterns
        sql_patterns = [
            r'(?i)(SELECT.*?(?:;|\n\n|\Z))',
            r'(?i)(INSERT.*?(?:;|\n\n|\Z))',
            r'(?i)(UPDATE.*?(?:;|\n\n|\Z))',
            r'(?i)(DELETE.*?(?:;|\n\n|\Z))',
            r'(?i)(WITH.*?SELECT.*?(?:;|\n\n|\Z))'
        ]
        
        for pattern in sql_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            if matches:
                return matches[0].strip().rstrip(';') + ';'
        
        return ""
    
    def _post_process_result(
        self,
        sql_result: Dict[str, Any],
        context: Dict[str, Any],
        natural_query: str
    ) -> Dict[str, Any]:
        """Post-process and validate the generated result."""
        
        # Basic validation
        sql_query = sql_result.get("sql_query", "").strip()
        
        # Calculate confidence adjustments
        confidence = sql_result.get("confidence", 0.0)
        
        # Reduce confidence if no schema context was found
        if context["num_schema_sources"] == 0:
            confidence *= 0.7
            
        # Reduce confidence if SQL looks invalid
        if not sql_query or not self._basic_sql_validation(sql_query):
            confidence *= 0.5
        
        # Add metadata
        result = {
            "natural_query": natural_query,
            "sql_query": sql_query,
            "explanation": sql_result.get("explanation", ""),
            "confidence": min(confidence, 1.0),
            "tables_used": sql_result.get("tables_used", []),
            "sources": sql_result.get("sources", []),
            "context_quality": {
                "schema_sources": context["num_schema_sources"],
                "example_sources": context["num_example_sources"],
                "available_tables": len(context["available_tables"])
            }
        }
        
        return result
    
    def _basic_sql_validation(self, sql: str) -> bool:
        """Basic SQL syntax validation."""
        if not sql.strip():
            return False
            
        sql_upper = sql.strip().upper()
        
        # Check for basic SQL keywords
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']
        if not any(sql_upper.startswith(keyword) for keyword in sql_keywords):
            return False
        
        # Basic bracket matching
        if sql.count('(') != sql.count(')'):
            return False
        
        # Check for FROM clause in SELECT statements
        if sql_upper.startswith('SELECT') and ' FROM ' not in sql_upper:
            return False
        
        return True
    
    def explain_query(self, sql_query: str) -> Dict[str, Any]:
        """Explain what a SQL query does."""
        try:
            response = self.vector_service.query_with_context(
                query=f"Explain this SQL query: {sql_query}",
                retriever_type="hybrid"
            )
            
            return {
                "sql_query": sql_query,
                "explanation": response.get("response", ""),
                "sources": response.get("sources", [])
            }
            
        except Exception as e:
            logger.error("Failed to explain query", error=str(e))
            return {
                "sql_query": sql_query,
                "explanation": f"Error explaining query: {str(e)}",
                "sources": []
            }
    
    async def generate_and_execute_query(
        self,
        natural_language_query: str,
        session_id: Optional[str] = None,
        use_hybrid_retrieval: bool = True,
        auto_execute: bool = False
    ) -> Dict[str, Any]:
        """Generate SQL query and optionally execute it."""
        
        # Step 1: Generate SQL query
        generation_result = self.generate_sql_query(
            natural_language_query=natural_language_query,
            session_id=session_id,
            use_hybrid_retrieval=use_hybrid_retrieval
        )
        
        if not generation_result.get("sql_query") or generation_result.get("error"):
            return {
                **generation_result,
                "execution_result": None,
                "auto_executed": False
            }
        
        sql_query = generation_result["sql_query"]
        
        # Step 2: Validate query if possible
        try:
            validation_result = await self.query_execution_service.validate_query(sql_query)
            generation_result["validation"] = validation_result
            
            if not validation_result.get("valid", True):
                return {
                    **generation_result,
                    "execution_result": None,
                    "auto_executed": False,
                    "validation_error": validation_result.get("error")
                }
        except Exception as e:
            logger.warning("Query validation failed", error=str(e))
            generation_result["validation"] = {"valid": True, "warning": "Validation service unavailable"}
        
        # Step 3: Execute query if requested and valid
        execution_result = None
        if auto_execute and generation_result.get("confidence", 0) > 0.7:
            try:
                execution_result = await self.query_execution_service.execute_query(
                    sql_query=sql_query,
                    session_id=session_id,
                    metadata={
                        "natural_query": natural_language_query,
                        "confidence": generation_result.get("confidence"),
                        "sources": generation_result.get("sources", [])
                    }
                )
                
                # Process execution results
                if execution_result.get("success", False):
                    execution_result = self._process_execution_results(
                        execution_result,
                        natural_language_query,
                        sql_query
                    )
                
            except Exception as e:
                logger.error("Query execution failed", error=str(e))
                execution_result = {
                    "success": False,
                    "error": f"Execution failed: {str(e)}",
                    "query_executed": sql_query
                }
        
        return {
            **generation_result,
            "execution_result": execution_result,
            "auto_executed": auto_execute and execution_result is not None
        }
    
    def _process_execution_results(
        self,
        execution_result: Dict[str, Any],
        natural_query: str,
        sql_query: str
    ) -> Dict[str, Any]:
        """Process and enhance execution results."""
        
        data = execution_result.get("data", [])
        
        # Add result analysis
        analysis = {
            "total_rows": len(data),
            "columns": list(data[0].keys()) if data else [],
            "has_data": len(data) > 0
        }
        
        # Generate natural language summary of results
        try:
            if data:
                summary_prompt = f"""
                Natural query: {natural_query}
                SQL query: {sql_query}
                Results: {len(data)} rows returned
                Sample data: {data[:3] if len(data) > 3 else data}
                
                Provide a brief natural language summary of these query results.
                """
                
                summary_response = self.vector_service.query_with_context(
                    query=summary_prompt,
                    retriever_type="vector"
                )
                
                analysis["summary"] = summary_response.get("response", "Query executed successfully")
            else:
                analysis["summary"] = "Query executed successfully but returned no results"
                
        except Exception as e:
            logger.error("Failed to generate result summary", error=str(e))
            analysis["summary"] = "Query executed successfully"
        
        return {
            **execution_result,
            "analysis": analysis
        }
    
    async def validate_and_suggest_fixes(self, sql_query: str) -> Dict[str, Any]:
        """Validate SQL query and suggest fixes if invalid."""
        try:
            validation_result = await self.query_execution_service.validate_query(sql_query)
            
            if validation_result.get("valid", True):
                return {
                    "valid": True,
                    "sql_query": sql_query,
                    "message": "Query is valid"
                }
            
            # Generate suggestions for fixing the query
            error_message = validation_result.get("error", "Unknown validation error")
            
            suggestion_response = self.vector_service.query_with_context(
                query=f"Fix this SQL query error: {sql_query}\nError: {error_message}",
                retriever_type="hybrid"
            )
            
            return {
                "valid": False,
                "sql_query": sql_query,
                "error": error_message,
                "suggestions": suggestion_response.get("response", "Unable to generate suggestions"),
                "sources": suggestion_response.get("sources", [])
            }
            
        except Exception as e:
            logger.error("Failed to validate and suggest fixes", error=str(e))
            return {
                "valid": False,
                "sql_query": sql_query,
                "error": f"Validation service error: {str(e)}",
                "suggestions": "Unable to validate query at this time"
            }