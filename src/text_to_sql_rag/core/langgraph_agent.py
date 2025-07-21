"""
LangGraph-based text-to-SQL agent with human-in-the-loop and confidence assessment.
"""
import json
import logging
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END

from ..models.simple_models import (
    DocumentType,
    SQLGenerationResponse
)
from ..models.conversation import (
    ConversationState,
    RequestType,
    ConversationStatus,
    WorkflowStep,
    AgentResponse,
    ClarificationRequest,
    SQLArtifact
)
from ..services.vector_service import LlamaIndexVectorService
from ..services.query_execution_service import QueryExecutionService

logger = logging.getLogger(__name__)


class TextToSQLAgent:
    """LangGraph-based agent for text-to-SQL generation with HITL and confidence assessment."""
    
    def __init__(
        self,
        vector_service: LlamaIndexVectorService,
        query_execution_service: Optional[QueryExecutionService] = None,
        max_retries: int = 3,
        confidence_threshold: float = 0.7
    ):
        self.vector_service = vector_service
        self.query_execution_service = query_execution_service
        self.max_retries = max_retries
        self.confidence_threshold = confidence_threshold
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the enhanced LangGraph workflow with HITL and confidence assessment."""
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("classify_request", self._classify_request_node)
        workflow.add_node("get_metadata", self._get_metadata_node)
        workflow.add_node("assess_confidence", self._assess_confidence_node)
        workflow.add_node("request_clarification", self._request_clarification_node)
        workflow.add_node("generate_sql", self._generate_sql_node)
        workflow.add_node("execute_sql", self._execute_sql_node)
        workflow.add_node("describe_sql", self._describe_sql_node)
        workflow.add_node("fix_sql", self._fix_sql_node)
        workflow.add_node("return_results", self._return_results_node)
        
        # Define the flow
        workflow.set_entry_point("classify_request")
        
        # From classify_request, route based on request type
        workflow.add_conditional_edges(
            "classify_request",
            self._route_by_request_type,
            {
                "get_metadata": "get_metadata",
                "describe_sql": "describe_sql",
                "execute_sql": "execute_sql"
            }
        )
        
        # From get_metadata, assess confidence
        workflow.add_edge("get_metadata", "assess_confidence")
        
        # From assess_confidence, either clarify or generate SQL
        workflow.add_conditional_edges(
            "assess_confidence",
            self._should_request_clarification,
            {
                "clarify": "request_clarification",
                "proceed": "generate_sql"
            }
        )
        
        # From request_clarification, wait for human input (terminal for now)
        workflow.add_edge("request_clarification", END)
        
        # From generate_sql, decide next step
        workflow.add_conditional_edges(
            "generate_sql",
            self._should_execute_sql,
            {
                "execute": "execute_sql",
                "skip": "return_results"
            }
        )
        
        # From execute_sql, check if we need to retry
        workflow.add_conditional_edges(
            "execute_sql",
            self._should_retry_sql,
            {
                "retry": "fix_sql",
                "success": "return_results",
                "max_retries": "return_results"
            }
        )
        
        # From fix_sql, go back to generate_sql
        workflow.add_edge("fix_sql", "generate_sql")
        
        # From describe_sql, go to return_results
        workflow.add_edge("describe_sql", "return_results")
        
        # return_results is terminal
        workflow.add_edge("return_results", END)
        
        return workflow.compile()
    
    def _classify_request_node(self, state: ConversationState) -> ConversationState:
        """Classify the type of user request to determine workflow routing."""
        logger.info(f"Classifying request: {state.current_request}")
        
        state.workflow_step = WorkflowStep.CLASSIFY_REQUEST
        state.add_message("system", f"Processing request: {state.current_request}", "classification")
        
        try:
            # Simple keyword-based classification for now
            request_lower = state.current_request.lower()
            
            # Check for SQL execution requests
            if any(keyword in request_lower for keyword in ["execute", "run", "execute this sql", "run this query"]):
                state.request_type = RequestType.EXECUTE_SQL
                
            # Check for SQL description requests
            elif any(keyword in request_lower for keyword in ["explain", "describe", "what does", "understand"]):
                state.request_type = RequestType.DESCRIBE_SQL
                
            # Check for editing previous SQL
            elif any(keyword in request_lower for keyword in ["edit", "modify", "change", "update", "fix"]):
                if state.sql_history:
                    state.request_type = RequestType.EDIT_PREVIOUS
                else:
                    state.request_type = RequestType.GENERATE_NEW
                    
            # Check for follow-up questions
            elif any(keyword in request_lower for keyword in ["also", "additionally", "furthermore", "and"]):
                state.request_type = RequestType.FOLLOW_UP
                
            # Check for clarification responses
            elif state.status == ConversationStatus.WAITING_FOR_CLARIFICATION:
                state.request_type = RequestType.CLARIFICATION
                
            # Default to new SQL generation
            else:
                state.request_type = RequestType.GENERATE_NEW
            
            logger.info(f"Classified request as: {state.request_type}")
            
        except Exception as e:
            logger.error(f"Error classifying request: {e}")
            state.request_type = RequestType.GENERATE_NEW
        
        return state
    
    def _get_metadata_node(self, state: ConversationState) -> ConversationState:
        """Retrieve relevant schema and example metadata."""
        logger.info(f"Getting metadata for query: {state.current_request}")
        
        state.workflow_step = WorkflowStep.GET_METADATA
        
        try:
            # Get schema context
            schema_results = self.vector_service.search_similar(
                query=state.current_request,
                retriever_type="hybrid",
                similarity_top_k=5,
                document_type=DocumentType.SCHEMA.value
            )
            
            # Get example context
            example_results = self.vector_service.search_similar(
                query=state.current_request,
                retriever_type="hybrid", 
                similarity_top_k=3,
                document_type=DocumentType.REPORT.value
            )
            
            # Extract context information
            schema_context = []
            for result in schema_results:
                schema_context.append({
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("score", 0.0)
                })
            
            example_context = []
            for result in example_results:
                example_context.append({
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("score", 0.0)
                })
            
            state.schema_context = schema_context
            state.example_context = example_context
            
            # Extract sources
            sources = []
            for result in schema_results + example_results:
                metadata = result.get("metadata", {})
                if "source" in metadata:
                    sources.append(metadata["source"])
            state.sources = list(set(sources))
            
            logger.info(f"Retrieved {len(schema_context)} schema docs and {len(example_context)} example docs")
            
        except Exception as e:
            logger.error(f"Error getting metadata: {e}")
            state.schema_context = []
            state.example_context = []
            state.sources = []
        
        return state
    
    def _assess_confidence_node(self, state: ConversationState) -> ConversationState:
        """Assess confidence in available metadata to answer the user's question."""
        logger.info("Assessing confidence in available metadata")
        
        state.workflow_step = WorkflowStep.ASSESS_CONFIDENCE
        
        try:
            # Build confidence assessment prompt
            assessment_prompt = f"""
You are a SQL expert analyzing whether you have sufficient database metadata to confidently answer a user's question.

User's request: "{state.current_request}"

Available schema context ({len(state.schema_context)} documents):
{self._format_context_for_assessment(state.schema_context[:3])}

Available example queries ({len(state.example_context)} documents):
{self._format_context_for_assessment(state.example_context[:2])}

Analyze the following:
1. Do you have clear information about the relevant tables and columns needed?
2. Are there any ambiguous terms in the request that need clarification?
3. Are there multiple possible interpretations of the request?
4. Do you have sufficient examples to understand the data structure?

Respond in this format:

## Confidence Assessment
Confidence: [0.0-1.0]

## Analysis
[Brief analysis of what you know and what might be missing]

## Missing Information
[List specific information needed for clarification, or "None" if confident]

## Suggested Questions
[Specific questions to ask the user, or "None" if confident]

Only request clarification if confidence is below 0.7 or if there are genuine ambiguities.
"""
            
            # Get confidence assessment from LLM
            response = self.vector_service.query_engine.query(assessment_prompt)
            assessment_result = self._parse_confidence_assessment(str(response))
            
            state.confidence_score = assessment_result["confidence"]
            
            # Determine if clarification is needed
            if (assessment_result["confidence"] < self.confidence_threshold or 
                assessment_result["missing_info"]):
                
                state.needs_clarification = True
                state.clarification_request = ClarificationRequest(
                    question=assessment_result["clarification_question"],
                    context=assessment_result["analysis"],
                    missing_info=assessment_result["missing_info"],
                    suggested_tables=self._extract_table_suggestions(state.schema_context),
                    confidence_issues=assessment_result["confidence_issues"]
                )
            else:
                state.needs_clarification = False
            
            logger.info(f"Confidence assessment: {state.confidence_score}, needs clarification: {state.needs_clarification}")
            
        except Exception as e:
            logger.error(f"Error assessing confidence: {e}")
            # Default to proceeding if assessment fails
            state.confidence_score = 0.6
            state.needs_clarification = False
        
        return state
    
    def _request_clarification_node(self, state: ConversationState) -> ConversationState:
        """Request clarification from the user."""
        logger.info("Requesting clarification from user")
        
        state.workflow_step = WorkflowStep.REQUEST_CLARIFICATION
        state.status = ConversationStatus.WAITING_FOR_CLARIFICATION
        state.requires_human_input = True
        
        # Add clarification message to conversation
        clarification_msg = self._format_clarification_message(state.clarification_request)
        state.add_message("assistant", clarification_msg, "clarification_request")
        
        # Create final result for clarification request
        state.final_result = {
            "response_type": "clarification_request",
            "status": "waiting_for_clarification",
            "clarification_request": state.clarification_request.dict() if state.clarification_request else None,
            "confidence_score": state.confidence_score,
            "conversation_id": state.conversation_id
        }
        
        logger.info("Clarification request prepared")
        return state
    
    def _generate_sql_node(self, state: ConversationState) -> ConversationState:
        """Generate SQL using LLM with schema and example context."""
        logger.info("Generating SQL")
        
        state.workflow_step = WorkflowStep.GENERATE_SQL
        
        try:
            # Build context prompt
            context_parts = []
            
            # Add conversation context if this is a follow-up
            if state.request_type in [RequestType.FOLLOW_UP, RequestType.EDIT_PREVIOUS]:
                context_parts.append("=== CONVERSATION HISTORY ===")
                context_parts.append(state.get_conversation_summary())
                context_parts.append("")
            
            # Add schema context
            if state.schema_context:
                context_parts.append("=== DATABASE SCHEMA ===")
                for i, schema in enumerate(state.schema_context[:3]):  # Limit to top 3
                    context_parts.append(f"Schema {i+1}:")
                    context_parts.append(schema["content"])
                    context_parts.append("")
            
            # Add example context
            if state.example_context:
                context_parts.append("=== EXAMPLE QUERIES ===")
                for i, example in enumerate(state.example_context):
                    context_parts.append(f"Example {i+1}:")
                    context_parts.append(example["content"])
                    context_parts.append("")
            
            # Add previous SQL if editing
            if state.request_type == RequestType.EDIT_PREVIOUS and state.sql_history:
                recent_sql = state.get_recent_sql()
                if recent_sql:
                    context_parts.append("=== PREVIOUS SQL ===")
                    context_parts.append(f"```sql\n{recent_sql.sql}\n```")
                    context_parts.append(f"Explanation: {recent_sql.explanation}")
                    context_parts.append("")
            
            # Add error context for retries
            error_context = ""
            if state.error_message and state.retry_count > 0:
                error_context = f"""
=== PREVIOUS ERROR ===
The previous SQL query failed with this error:
{state.error_message}

Please fix the SQL query to address this error.
"""
            
            # Build the full prompt
            prompt = f"""
{'\n'.join(context_parts)}

{error_context}

Based on the database schema and example queries above, generate a SQL query for the following request:

"{state.current_request}"

Please respond in markdown format with the following structure:

## SQL Query
```sql
-- Your SQL query here
SELECT ...
```

## Explanation
Brief explanation of what the query does and how it answers the request.

## Confidence
Confidence score: 0.85 (between 0.0 and 1.0)

Requirements:
- Only use tables and columns that exist in the provided schema
- Follow the patterns shown in the example queries
- Make sure the SQL is syntactically correct
- Include comments in the SQL where helpful
"""
            
            # Generate SQL using the vector service's query engine
            response = self.vector_service.query_engine.query(prompt)
            
            # Parse response
            sql_result = self._parse_sql_response(str(response))
            
            state.current_sql = sql_result["sql"]
            
            # Add SQL artifact to history
            if sql_result["sql"]:
                tables_used = self._extract_tables_from_sql(sql_result["sql"])
                state.add_sql_artifact(
                    sql=sql_result["sql"],
                    explanation=sql_result["explanation"],
                    confidence=sql_result["confidence"],
                    tables_used=tables_used
                )
            
            logger.info(f"Generated SQL with confidence {sql_result['confidence']}")
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            state.current_sql = None
        
        return state
    
    def _describe_sql_node(self, state: ConversationState) -> ConversationState:
        """Describe and explain an SQL query."""
        logger.info("Describing SQL query")
        
        state.workflow_step = WorkflowStep.DESCRIBE_SQL
        
        try:
            # Extract SQL from the request if provided
            sql_to_describe = self._extract_sql_from_request(state.current_request)
            
            if not sql_to_describe and state.sql_history:
                # Use the most recent SQL if no SQL provided in request
                recent_sql = state.get_recent_sql()
                sql_to_describe = recent_sql.sql if recent_sql else None
            
            if sql_to_describe:
                description_prompt = f"""
Please explain the following SQL query in detail:

```sql
{sql_to_describe}
```

Provide a comprehensive explanation including:
1. What data the query retrieves
2. Which tables and columns are used
3. Any joins, filters, or aggregations applied
4. The business purpose or use case
5. Any potential performance considerations

Format your response clearly and make it understandable for both technical and non-technical users.
"""
                
                response = self.vector_service.query_engine.query(description_prompt)
                
                state.final_result = {
                    "response_type": "sql_description",
                    "sql": sql_to_describe,
                    "description": str(response),
                    "conversation_id": state.conversation_id
                }
            else:
                state.final_result = {
                    "response_type": "error",
                    "error": "No SQL query found to describe",
                    "conversation_id": state.conversation_id
                }
            
        except Exception as e:
            logger.error(f"Error describing SQL: {e}")
            state.final_result = {
                "response_type": "error",
                "error": str(e),
                "conversation_id": state.conversation_id
            }
        
        return state
    
    def _execute_sql_node(self, state: ConversationState) -> ConversationState:
        """Execute the generated SQL query."""
        if not state.current_sql or not self.query_execution_service:
            return state
        
        logger.info("Executing SQL query")
        state.workflow_step = WorkflowStep.EXECUTE_SQL
        
        try:
            result = self.query_execution_service.execute_query(state.current_sql)
            
            if result["success"]:
                state.execution_result = result
                state.error_message = None
                
                # Update the latest SQL artifact with execution result
                recent_sql = state.get_recent_sql()
                if recent_sql:
                    recent_sql.execution_result = result
                
                logger.info("SQL executed successfully")
            else:
                state.error_message = result.get("error", "Unknown execution error")
                
                # Update the latest SQL artifact with error
                recent_sql = state.get_recent_sql()
                if recent_sql:
                    recent_sql.error_message = state.error_message
                
                logger.warning(f"SQL execution failed: {state.error_message}")
                
        except Exception as e:
            state.error_message = str(e)
            logger.error(f"Error executing SQL: {e}")
        
        return state
    
    def _fix_sql_node(self, state: ConversationState) -> ConversationState:
        """Prepare for SQL retry by incrementing counter."""
        state.retry_count += 1
        state.workflow_step = WorkflowStep.FIX_SQL
        
        # Update the retry count on the most recent SQL artifact
        recent_sql = state.get_recent_sql()
        if recent_sql:
            recent_sql.retry_count = state.retry_count
        
        logger.info(f"Preparing SQL retry {state.retry_count}/{state.max_retries}")
        return state
    
    def _return_results_node(self, state: ConversationState) -> ConversationState:
        """Format and return final results."""
        logger.info("Formatting final results")
        
        state.workflow_step = WorkflowStep.RETURN_RESULTS
        state.status = ConversationStatus.COMPLETED
        
        # Return early if we already have a final result (e.g., from clarification)
        if state.final_result:
            return state
        
        # Determine success status
        success = state.current_sql is not None
        if self.query_execution_service:
            success = success and state.execution_result and state.execution_result.get("success", False)
        
        # Get the most recent SQL artifact
        recent_sql = state.get_recent_sql()
        
        # Build response based on request type
        if state.request_type == RequestType.DESCRIBE_SQL:
            # This should have been handled in describe_sql_node
            pass
        else:
            # Build SQL generation response
            response_data = {
                "response_type": "sql_result",
                "conversation_id": state.conversation_id,
                "status": "completed",
                "sql": state.current_sql or "",
                "explanation": recent_sql.explanation if recent_sql else f"Generated SQL query for: {state.current_request}",
                "confidence": recent_sql.confidence if recent_sql else 0.0,
                "tables_used": recent_sql.tables_used if recent_sql else [],
                "sources": [{"source": src} for src in state.sources],
                "processing_info": {
                    "retry_count": state.retry_count,
                    "max_retries": state.max_retries,
                    "schema_docs_count": len(state.schema_context),
                    "example_docs_count": len(state.example_context),
                    "request_type": state.request_type.value if state.request_type else None
                }
            }
            
            # Add execution results if available
            if state.execution_result:
                response_data["execution_result"] = state.execution_result
                response_data["auto_executed"] = True
            
            # Add error information if final attempt failed
            if state.error_message and state.retry_count >= state.max_retries:
                response_data["processing_info"]["final_error"] = state.error_message
                response_data["status"] = "error"
            
            state.final_result = response_data
        
        return state
    
    def _route_by_request_type(self, state: ConversationState) -> str:
        """Route workflow based on request type."""
        if state.request_type == RequestType.DESCRIBE_SQL:
            return "describe_sql"
        elif state.request_type == RequestType.EXECUTE_SQL:
            return "execute_sql"
        else:
            # For GENERATE_NEW, EDIT_PREVIOUS, FOLLOW_UP, CLARIFICATION
            return "get_metadata"
    
    def _should_request_clarification(self, state: ConversationState) -> str:
        """Determine if we should request clarification."""
        if state.needs_clarification:
            return "clarify"
        return "proceed"
    
    def _should_execute_sql(self, state: ConversationState) -> str:
        """Determine if we should execute the SQL."""
        if self.query_execution_service and state.current_sql and state.should_execute:
            return "execute"
        return "skip"
    
    def _should_retry_sql(self, state: ConversationState) -> str:
        """Determine if we should retry SQL generation."""
        # If execution was successful, we're done
        if state.execution_result and state.execution_result.get("success", False):
            return "success"
        
        # If we have an error and haven't exceeded max retries, retry
        if (state.error_message and state.retry_count < state.max_retries):
            return "retry"
        
        # Otherwise, we've hit max retries or no error to retry
        return "max_retries"
    
    def _format_context_for_assessment(self, context_list: List[Dict[str, Any]]) -> str:
        """Format context for confidence assessment."""
        formatted_parts = []
        for i, ctx in enumerate(context_list, 1):
            content_preview = ctx.get("content", "")[:200] + "..." if len(ctx.get("content", "")) > 200 else ctx.get("content", "")
            formatted_parts.append(f"Document {i}: {content_preview}")
        return "\n".join(formatted_parts)
    
    def _parse_confidence_assessment(self, response_text: str) -> Dict[str, Any]:
        """Parse confidence assessment response."""
        import re
        
        result = {
            "confidence": 0.5,
            "analysis": "",
            "missing_info": [],
            "clarification_question": "",
            "confidence_issues": []
        }
        
        try:
            # Extract confidence score
            confidence_pattern = re.compile(r'Confidence:\s*([0-9.]+)', re.IGNORECASE)
            confidence_match = confidence_pattern.search(response_text)
            if confidence_match:
                result["confidence"] = float(confidence_match.group(1))
            
            # Extract analysis
            analysis_pattern = re.compile(r'## Analysis\s*([^#]*)', re.IGNORECASE | re.DOTALL)
            analysis_match = analysis_pattern.search(response_text)
            if analysis_match:
                result["analysis"] = analysis_match.group(1).strip()
            
            # Extract missing information
            missing_pattern = re.compile(r'## Missing Information\s*([^#]*)', re.IGNORECASE | re.DOTALL)
            missing_match = missing_pattern.search(response_text)
            if missing_match:
                missing_text = missing_match.group(1).strip()
                if missing_text.lower() != "none":
                    result["missing_info"] = [item.strip("- ") for item in missing_text.split("\n") if item.strip()]
            
            # Extract suggested questions
            questions_pattern = re.compile(r'## Suggested Questions\s*([^#]*)', re.IGNORECASE | re.DOTALL)
            questions_match = questions_pattern.search(response_text)
            if questions_match:
                questions_text = questions_match.group(1).strip()
                if questions_text.lower() != "none":
                    result["clarification_question"] = questions_text
            
        except Exception as e:
            logger.warning(f"Error parsing confidence assessment: {e}")
        
        return result
    
    def _extract_table_suggestions(self, schema_context: List[Dict[str, Any]]) -> List[str]:
        """Extract table names from schema context."""
        tables = set()
        for ctx in schema_context:
            content = ctx.get("content", "")
            # Simple regex to find table names
            import re
            table_matches = re.findall(r'(?:table|TABLE)\s+([A-Za-z_][A-Za-z0-9_]*)', content)
            tables.update(table_matches)
        return list(tables)[:5]  # Limit to 5 suggestions
    
    def _format_clarification_message(self, clarification_request: ClarificationRequest) -> str:
        """Format clarification request as user-friendly message."""
        message_parts = []
        
        message_parts.append("I need some clarification to provide you with the most accurate SQL query.")
        message_parts.append("")
        message_parts.append(clarification_request.question)
        
        if clarification_request.missing_info:
            message_parts.append("")
            message_parts.append("Specifically, I need to understand:")
            for info in clarification_request.missing_info:
                message_parts.append(f"• {info}")
        
        if clarification_request.suggested_tables:
            message_parts.append("")
            message_parts.append(f"Available tables that might be relevant: {', '.join(clarification_request.suggested_tables)}")
        
        return "\n".join(message_parts)
    
    def _extract_sql_from_request(self, request: str) -> Optional[str]:
        """Extract SQL query from user request."""
        import re
        
        # Look for SQL in code blocks
        sql_pattern = re.compile(r'```sql\s*(.*?)\s*```', re.DOTALL | re.IGNORECASE)
        match = sql_pattern.search(request)
        if match:
            return match.group(1).strip()
        
        # Look for SQL keywords
        sql_keywords_pattern = re.compile(
            r'(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s+.*?(?=\n\n|\Z|;)',
            re.IGNORECASE | re.DOTALL
        )
        match = sql_keywords_pattern.search(request)
        if match:
            return match.group(0).strip()
        
        return None
    
    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL query."""
        import re
        
        tables = set()
        
        # Extract from FROM clauses
        from_pattern = re.compile(r'\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)', re.IGNORECASE)
        tables.update(from_pattern.findall(sql))
        
        # Extract from JOIN clauses
        join_pattern = re.compile(r'\bJOIN\s+([A-Za-z_][A-Za-z0-9_]*)', re.IGNORECASE)
        tables.update(join_pattern.findall(sql))
        
        # Extract from INSERT INTO
        insert_pattern = re.compile(r'\bINSERT\s+INTO\s+([A-Za-z_][A-Za-z0-9_]*)', re.IGNORECASE)
        tables.update(insert_pattern.findall(sql))
        
        # Extract from UPDATE
        update_pattern = re.compile(r'\bUPDATE\s+([A-Za-z_][A-Za-z0-9_]*)', re.IGNORECASE)
        tables.update(update_pattern.findall(sql))
        
        return list(tables)
    
    def _parse_sql_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response in markdown format to extract SQL and metadata."""
        import re
        
        result = {
            "sql": "",
            "explanation": "",
            "confidence": 0.7
        }
        
        try:
            # Extract SQL from code block
            sql_pattern = re.compile(r'```sql\s*(.*?)\s*```', re.DOTALL | re.IGNORECASE)
            sql_match = sql_pattern.search(response_text)
            
            if sql_match:
                sql_content = sql_match.group(1).strip()
                # Remove comments and clean up
                sql_lines = []
                for line in sql_content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('--'):
                        sql_lines.append(line)
                result["sql"] = ' '.join(sql_lines)
            
            # Extract explanation
            explanation_pattern = re.compile(r'## Explanation\s*(.*?)(?=##|$)', re.DOTALL | re.IGNORECASE)
            explanation_match = explanation_pattern.search(response_text)
            
            if explanation_match:
                result["explanation"] = explanation_match.group(1).strip()
            
            # Extract confidence score
            confidence_pattern = re.compile(r'Confidence score:\s*([0-9.]+)', re.IGNORECASE)
            confidence_match = confidence_pattern.search(response_text)
            
            if confidence_match:
                result["confidence"] = float(confidence_match.group(1))
            
            # If no SQL found in code block, try fallback extraction
            if not result["sql"]:
                result = self._fallback_sql_extraction(response_text)
                
        except Exception as e:
            logger.warning(f"Error parsing SQL response: {e}")
            result = self._fallback_sql_extraction(response_text)
        
        return result
    
    def _fallback_sql_extraction(self, response_text: str) -> Dict[str, Any]:
        """Fallback method to extract SQL from unstructured text."""
        import re
        
        # Look for SQL keywords and patterns
        sql_pattern = re.compile(
            r'(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s+.*?(?=\n\n|\Z)',
            re.IGNORECASE | re.DOTALL
        )
        
        matches = sql_pattern.findall(response_text)
        sql = ""
        
        if matches:
            # Take the longest match (likely the main query)
            sql = max(matches, key=len).strip()
            # Clean up SQL
            sql = re.sub(r'\s+', ' ', sql)  # Normalize whitespace
            sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)  # Remove comments
        
        return {
            "sql": sql,
            "explanation": "Extracted using fallback method",
            "confidence": 0.5
        }
    
    async def generate_sql(self, query: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Main entry point for SQL generation with enhanced conversation support."""
        logger.info(f"Starting SQL generation for query: {query}")
        
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Initialize conversation state
        initial_state = ConversationState(
            conversation_id=conversation_id,
            current_request=query,
            max_retries=self.max_retries
        )
        
        # Add initial user message
        initial_state.add_message("user", query, "request")
        
        # Run the workflow
        final_state = self.graph.invoke(initial_state)
        
        # Return the final result
        return final_state.final_result
    
    async def continue_conversation(
        self, 
        conversation_id: str, 
        message: str, 
        conversation_state: ConversationState
    ) -> Dict[str, Any]:
        """Continue an existing conversation with clarification or follow-up."""
        logger.info(f"Continuing conversation {conversation_id} with message: {message}")
        
        # Update conversation state
        conversation_state.current_request = message
        conversation_state.status = ConversationStatus.ACTIVE
        conversation_state.requires_human_input = False
        conversation_state.needs_clarification = False
        conversation_state.updated_at = datetime.utcnow()
        
        # Add user message
        conversation_state.add_message("user", message, "clarification")
        
        # If this was a clarification response, we can proceed to generate SQL
        if conversation_state.clarification_request:
            conversation_state.request_type = RequestType.CLARIFICATION
            # Clear clarification request
            conversation_state.clarification_request = None
        
        # Run the workflow from appropriate point
        final_state = self.graph.invoke(conversation_state)
        
        return final_state.final_result
    
    async def generate_sql_legacy(self, query: str) -> SQLGenerationResponse:
        """Legacy entry point for SQL generation (backwards compatibility)."""
        logger.info(f"Starting legacy SQL generation for query: {query}")
        
        # Use the new method and convert response
        result = await self.generate_sql(query)
        
        # Convert to SQLGenerationResponse for backwards compatibility
        return SQLGenerationResponse(
            natural_query=query,
            sql_query=result.get("sql", ""),
            explanation=result.get("explanation", ""),
            confidence=result.get("confidence", 0.0),
            tables_used=result.get("tables_used", []),
            sources=result.get("sources", []),
            context_quality=result.get("processing_info", {}),
            execution_result=result.get("execution_result"),
            auto_executed=result.get("auto_executed", False)
        )