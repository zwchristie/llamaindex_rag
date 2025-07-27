"""
LangGraph-based text-to-SQL agent with human-in-the-loop and confidence assessment.
"""
import json
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime
import structlog

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
from ..services.llm_provider_factory import llm_factory

logger = structlog.get_logger(__name__)


class TextToSQLAgent:
    """LangGraph-based agent for text-to-SQL generation with enhanced HITL and confidence assessment.
    
    This agent orchestrates intelligent text-to-SQL workflows using LangGraph with advanced
    human-in-the-loop capabilities. It features complete workflow state persistence,
    enabling seamless resumption of complex SQL generation workflows after clarification.
    
    Key Features:
    - LLM-powered request classification with reasoning and context analysis
    - Advanced confidence assessment using structured LLM evaluation
    - Checkpoint-based workflow state persistence for HITL scenarios
    - Dual ID system supporting conversation threads and individual request fulfillment
    - Intelligent workflow resumption with preserved context (schema, confidence, metadata)
    - Sophisticated retry mechanisms with context injection for SQL generation
    
    Architecture:
    - Uses LangGraph state machine for workflow orchestration
    - Implements checkpoint storage for human-in-the-loop interruption points
    - Separates conversation-level state from request-level workflow state
    - Provides fallback mechanisms for LLM classification and confidence assessment
    
    Workflow Steps:
    1. classify_request: LLM-powered request type detection with reasoning
    2. get_metadata: Schema and example retrieval from vector store
    3. assess_confidence: LLM analysis of metadata completeness and query clarity
    4. request_clarification: Save checkpoint and request human input (if needed)
    5. generate_sql: Context-aware SQL generation with retry logic
    6. execute_sql: Optional query execution with result processing
    7. return_results: Final response formatting and delivery
    
    HITL Process:
    - When confidence is low, complete workflow state is serialized as checkpoint
    - Checkpoint includes schema context, confidence scores, retrieved documents
    - User provides clarification through continue_from_checkpoint()
    - Workflow resumes with exact preserved context and continues seamlessly
    """
    
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
        
        # Checkpoint storage for HITL state persistence
        # In production, this should be Redis or a database
        self._checkpoint_storage = {}
    
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
        """Classify the type of user request using LLM to determine workflow routing.
        
        Uses advanced LLM-powered classification to analyze user requests and determine
        the appropriate workflow routing. This includes analyzing conversation context,
        SQL history, and request intent to provide accurate classification with reasoning.
        
        The classification supports these request types:
        - GENERATE_NEW: New SQL query generation from natural language
        - EXECUTE_SQL: Execute specific SQL code provided by user
        - DESCRIBE_SQL: Explain what an SQL query does
        - EDIT_PREVIOUS: Modify previously generated SQL
        - FOLLOW_UP: Continue previous conversation with additional requirements
        - CLARIFICATION: Response to previous clarification request
        
        Includes fallback to keyword-based classification if LLM fails.
        """
        print("DEBUG: About to log CLASSIFY REQUEST")
        logger.info("=== WORKFLOW NODE: CLASSIFY REQUEST ===")
        logger.info("Starting request classification", 
                   request=state.current_request,
                   conversation_id=state.conversation_id,
                   request_id=state.request_id)
        print("DEBUG: Finished logging CLASSIFY REQUEST")
        
        state.workflow_step = WorkflowStep.CLASSIFY_REQUEST
        state.add_message("system", f"Processing request: {state.current_request}", "classification")
        
        try:
            # Check for clarification responses first (no LLM needed)
            if state.status == ConversationStatus.WAITING_FOR_CLARIFICATION:
                state.request_type = RequestType.CLARIFICATION
                logger.info(f"Classified request as: {state.request_type}")
                return state
            
            # Build context for LLM classification
            context_parts = []
            
            # Add conversation history if available
            if state.message_history:
                context_parts.append("=== CONVERSATION CONTEXT ===")
                recent_messages = state.message_history[-3:]  # Last 3 messages for context
                for msg in recent_messages:
                    context_parts.append(f"{msg.role}: {msg.content}")
                context_parts.append("")
            
            # Add SQL history if available
            if state.sql_history:
                context_parts.append("=== PREVIOUS SQL QUERIES ===")
                recent_sql = state.get_recent_sql()
                if recent_sql:
                    context_parts.append(f"Most recent SQL: {recent_sql.sql}")
                    context_parts.append(f"Explanation: {recent_sql.explanation}")
                context_parts.append("")
            
            # Build classification prompt
            classification_prompt = f"""
{chr(10).join(context_parts)}

You are a SQL assistant analyzing user requests to determine the appropriate action. 

Current user request: "{state.current_request}"

Classify this request into ONE of the following categories:

1. **GENERATE_NEW** - User wants a new SQL query generated from natural language
   Examples: "Show me all users", "Find top selling products", "Get monthly sales data"

2. **EXECUTE_SQL** - User wants to execute/run a specific SQL query they provided
   Examples: "Execute this SQL", "Run this query", "Execute: SELECT * FROM users"

3. **DESCRIBE_SQL** - User wants an explanation of what an SQL query does
   Examples: "Explain this SQL", "What does this query do?", "Describe: SELECT * FROM users"

4. **EDIT_PREVIOUS** - User wants to modify/fix the previously generated SQL
   Examples: "Add a WHERE clause", "Change the ORDER BY", "Fix the previous query", "Make it sort by date"

5. **FOLLOW_UP** - User is asking a follow-up question related to previous results
   Examples: "Show me more details", "What about last month?", "Also include the names"

**Classification Rules:**
- If the request contains explicit SQL code and asks to run/execute it → EXECUTE_SQL
- If the request contains explicit SQL code and asks to explain it → DESCRIBE_SQL  
- If there's previous SQL history and the request asks to modify/change it → EDIT_PREVIOUS
- If there's previous context and the request builds upon it → FOLLOW_UP
- If the request is a new natural language question → GENERATE_NEW

**Response Format:**
Classification: [GENERATE_NEW|EXECUTE_SQL|DESCRIBE_SQL|EDIT_PREVIOUS|FOLLOW_UP]
Reasoning: [Brief explanation of why this classification was chosen]
"""
            
            # Get classification from LLM
            response_text = llm_factory.generate_text(classification_prompt)
            
            # Parse the LLM response
            classification_result = self._parse_classification_response(response_text)
            
            # Map classification to RequestType
            classification_mapping = {
                "GENERATE_NEW": RequestType.GENERATE_NEW,
                "EXECUTE_SQL": RequestType.EXECUTE_SQL,
                "DESCRIBE_SQL": RequestType.DESCRIBE_SQL,
                "EDIT_PREVIOUS": RequestType.EDIT_PREVIOUS,
                "FOLLOW_UP": RequestType.FOLLOW_UP
            }
            
            state.request_type = classification_mapping.get(
                classification_result["classification"], 
                RequestType.GENERATE_NEW
            )
            
            # Store classification reasoning for debugging
            if classification_result["reasoning"]:
                state.add_message("system", f"Classification reasoning: {classification_result['reasoning']}", "classification_reasoning")
            
            logger.info(f"LLM classified request as: {state.request_type} - {classification_result['reasoning']}")
            
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            # Fallback to simple keyword-based classification
            state.request_type = self._fallback_classify_request(state.current_request, state)
            logger.info(f"Fallback classified request as: {state.request_type}")
        
        logger.info("=== CLASSIFY REQUEST COMPLETE ===", 
                   final_request_type=state.request_type.value if state.request_type else None,
                   next_step="get_metadata")
        
        return state
    
    def _get_metadata_node(self, state: ConversationState) -> ConversationState:
        """Retrieve relevant schema and example metadata."""
        logger.info("=== WORKFLOW NODE: GET METADATA ===")
        logger.info("Starting metadata retrieval", 
                   request=state.current_request,
                   request_type=state.request_type.value if state.request_type else None)
        
        state.workflow_step = WorkflowStep.GET_METADATA
        
        try:
            # Get schema context
            logger.info("Searching for schema documents", 
                       query=state.current_request, 
                       document_type=DocumentType.SCHEMA.value,
                       similarity_top_k=5)
            schema_results = self.vector_service.search_similar(
                query=state.current_request,
                retriever_type="hybrid",
                similarity_top_k=5,
                document_type=DocumentType.SCHEMA.value
            )
            logger.info("Schema search results", 
                       num_results=len(schema_results),
                       result_scores=[r.get("score", 0.0) for r in schema_results],
                       result_doc_ids=[r.get("metadata", {}).get("document_id") for r in schema_results])
            
            # Get example context
            logger.info("Searching for example/report documents", 
                       query=state.current_request, 
                       document_type=DocumentType.REPORT.value,
                       similarity_top_k=3)
            example_results = self.vector_service.search_similar(
                query=state.current_request,
                retriever_type="hybrid", 
                similarity_top_k=3,
                document_type=DocumentType.REPORT.value
            )
            logger.info("Example search results", 
                       num_results=len(example_results),
                       result_scores=[r.get("score", 0.0) for r in example_results],
                       result_doc_ids=[r.get("metadata", {}).get("document_id") for r in example_results])
            
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
            
            # Log the actual content being retrieved for debugging
            logger.info("Retrieved schema context details")
            for i, ctx in enumerate(schema_context):
                logger.info(f"Schema result {i+1}", 
                           document_id=ctx["metadata"].get("document_id"),
                           content_length=len(ctx["content"]),
                           content_preview=ctx["content"][:200] + "..." if len(ctx["content"]) > 200 else ctx["content"],
                           score=ctx["score"])
            
            logger.info("Retrieved example context details")
            for i, ctx in enumerate(example_context):
                logger.info(f"Example result {i+1}", 
                           document_id=ctx["metadata"].get("document_id"),
                           content_length=len(ctx["content"]),
                           content_preview=ctx["content"][:200] + "..." if len(ctx["content"]) > 200 else ctx["content"],
                           score=ctx["score"])
            
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
        
        logger.info("=== GET METADATA COMPLETE ===", 
                   schema_chunks=len(state.schema_context),
                   example_chunks=len(state.example_context),
                   sources=len(state.sources),
                   next_step="assess_confidence")
        
        return state
    
    def _assess_confidence_node(self, state: ConversationState) -> ConversationState:
        """Assess confidence in available metadata to answer the user's question."""
        logger.info("=== WORKFLOW NODE: ASSESS CONFIDENCE ===")
        logger.info("Starting confidence assessment", 
                   schema_chunks=len(state.schema_context),
                   example_chunks=len(state.example_context),
                   confidence_threshold=self.confidence_threshold)
        
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
            
            # Log confidence assessment prompt for debugging
            logger.info("Sending confidence assessment prompt to LLM",
                       prompt_length=len(assessment_prompt),
                       schema_docs=len(state.schema_context),
                       example_docs=len(state.example_context))
            logger.debug("Confidence assessment prompt", prompt=assessment_prompt)
            
            # Get confidence assessment from LLM
            response_text = llm_factory.generate_text(assessment_prompt)
            
            # Log LLM assessment response
            logger.info("Received confidence assessment response",
                       response_length=len(response_text),
                       response_preview=response_text[:400] + "..." if len(response_text) > 400 else response_text)
            
            assessment_result = self._parse_confidence_assessment(response_text)
            
            state.confidence_score = assessment_result["confidence"]
            
            # Log confidence assessment result
            logger.info("Confidence assessment result",
                       confidence_score=assessment_result["confidence"],
                       confidence_threshold=self.confidence_threshold,
                       missing_info=assessment_result.get("missing_info", []),
                       analysis=assessment_result.get("analysis", ""),
                       will_request_clarification=(assessment_result["confidence"] < self.confidence_threshold or assessment_result.get("missing_info")))
            
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
        
        next_step = "request_clarification" if state.needs_clarification else "generate_sql"
        logger.info("=== ASSESS CONFIDENCE COMPLETE ===", 
                   confidence_score=state.confidence_score,
                   needs_clarification=state.needs_clarification,
                   next_step=next_step)
        
        return state
    
    def _request_clarification_node(self, state: ConversationState) -> ConversationState:
        """Request clarification from the user."""
        logger.info("=== WORKFLOW NODE: REQUEST CLARIFICATION ===")
        logger.info("Starting clarification request", 
                   confidence_score=state.confidence_score,
                   clarification_question=state.clarification_request.question if state.clarification_request else None)
        
        state.workflow_step = WorkflowStep.REQUEST_CLARIFICATION
        state.status = ConversationStatus.WAITING_FOR_CLARIFICATION
        state.requires_human_input = True
        
        # Add clarification message to conversation
        clarification_msg = self._format_clarification_message(state.clarification_request)
        state.add_message("assistant", clarification_msg, "clarification_request")
        
        # Save workflow checkpoint for HITL resumption
        checkpoint_data = state.save_workflow_checkpoint()
        
        # Store checkpoint in conversations for retrieval
        if hasattr(self, '_checkpoint_storage'):
            self._checkpoint_storage[state.request_id] = checkpoint_data
        
        # Create final result for clarification request with checkpoint info
        state.final_result = {
            "response_type": "clarification_request",
            "status": "waiting_for_clarification",
            "clarification": {
                "message": state.clarification_request.question if state.clarification_request else "Please provide more information.",
                "suggestions": [] # Can be enhanced based on clarification_request
            },
            "confidence_score": state.confidence_score,
            "conversation_id": state.conversation_id,
            "request_id": state.request_id,
            "checkpoint_saved": True
        }
        
        logger.info("=== REQUEST CLARIFICATION COMPLETE ===", 
                   request_id=state.request_id,
                   clarification_saved=True,
                   workflow_paused=True)
        return state
    
    def _generate_sql_node(self, state: ConversationState) -> ConversationState:
        """Generate SQL using LLM with schema and example context."""
        logger.info("=== WORKFLOW NODE: GENERATE SQL ===")
        logger.info("Starting SQL generation", 
                   request=state.current_request,
                   schema_chunks=len(state.schema_context),
                   example_chunks=len(state.example_context),
                   retry_count=state.retry_count)
        
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
            
            # Log the full prompt being sent to LLM for debugging
            logger.info("Sending prompt to LLM", 
                       prompt_length=len(prompt),
                       schema_chunks=len(state.schema_context),
                       example_chunks=len(state.example_context))
            logger.debug("Full LLM prompt", prompt=prompt)
            
            # Generate SQL using the vector service's query engine
            response_text = llm_factory.generate_text(prompt)
            
            # Log LLM response
            logger.info("Received LLM response", 
                       response_length=len(response_text),
                       response_preview=response_text[:300] + "..." if len(response_text) > 300 else response_text)
            
            # Parse response
            sql_result = self._parse_sql_response(response_text)
            
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
        
        next_step = "execute_sql" if self.query_execution_service and state.current_sql else "return_results"
        logger.info("=== GENERATE SQL COMPLETE ===", 
                   sql_generated=bool(state.current_sql),
                   sql_preview=state.current_sql[:100] + "..." if state.current_sql and len(state.current_sql) > 100 else state.current_sql,
                   confidence=sql_result.get('confidence', 0.0) if 'sql_result' in locals() else 0.0,
                   next_step=next_step)
        
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
                
                response_text = llm_factory.generate_text(description_prompt)
                
                state.final_result = {
                    "response_type": "sql_description",
                    "sql": sql_to_describe,
                    "description": response_text,
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
        logger.info("=== WORKFLOW NODE: RETURN RESULTS ===")
        logger.info("Formatting final results", 
                   has_sql=bool(state.current_sql),
                   has_execution_result=bool(state.execution_result),
                   request_type=state.request_type.value if state.request_type else None)
        
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
        
        logger.info("=== RETURN RESULTS COMPLETE ===", 
                   response_type=state.final_result.get("response_type") if state.final_result else None,
                   status=state.final_result.get("status") if state.final_result else None,
                   workflow_complete=True)
        
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
    
    def _parse_classification_response(self, response_text: str) -> Dict[str, str]:
        """Parse the LLM classification response."""
        import re
        
        result = {
            "classification": "GENERATE_NEW",
            "reasoning": ""
        }
        
        try:
            # Extract classification
            classification_pattern = re.compile(r'Classification:\s*([A-Z_]+)', re.IGNORECASE)
            classification_match = classification_pattern.search(response_text)
            if classification_match:
                result["classification"] = classification_match.group(1).upper()
            
            # Extract reasoning
            reasoning_pattern = re.compile(r'Reasoning:\s*(.*?)(?=\n\n|\Z)', re.IGNORECASE | re.DOTALL)
            reasoning_match = reasoning_pattern.search(response_text)
            if reasoning_match:
                result["reasoning"] = reasoning_match.group(1).strip()
            
        except Exception as e:
            logger.warning(f"Error parsing classification response: {e}")
        
        return result
    
    def _fallback_classify_request(self, request: str, state: ConversationState) -> RequestType:
        """Fallback keyword-based classification when LLM fails."""
        request_lower = request.lower()
        
        # Check for SQL execution requests
        if any(keyword in request_lower for keyword in ["execute", "run", "execute this sql", "run this query"]):
            return RequestType.EXECUTE_SQL
            
        # Check for SQL description requests
        elif any(keyword in request_lower for keyword in ["explain", "describe", "what does", "understand"]):
            return RequestType.DESCRIBE_SQL
            
        # Check for editing previous SQL
        elif any(keyword in request_lower for keyword in ["edit", "modify", "change", "update", "fix"]):
            if state.sql_history:
                return RequestType.EDIT_PREVIOUS
            else:
                return RequestType.GENERATE_NEW
                
        # Check for follow-up questions
        elif any(keyword in request_lower for keyword in ["also", "additionally", "furthermore", "and"]):
            return RequestType.FOLLOW_UP
            
        # Default to new SQL generation
        else:
            return RequestType.GENERATE_NEW
    
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
        logger.info("===============================================")
        logger.info("=== STARTING SQL GENERATION WORKFLOW ===")
        logger.info("===============================================")
        logger.info("Workflow entry point", 
                   query=query,
                   query_length=len(query),
                   conversation_id=conversation_id)
        
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            logger.info("Generated new conversation ID", conversation_id=conversation_id)
        
        # Generate unique request ID for this specific request fulfillment
        request_id = str(uuid.uuid4())
        logger.info("Generated request ID", request_id=request_id)
        
        # Initialize conversation state
        initial_state = ConversationState(
            conversation_id=conversation_id,
            request_id=request_id,
            current_request=query,
            max_retries=self.max_retries
        )
        
        # Add initial user message
        initial_state.add_message("user", query, "request")
        
        # Run the workflow
        logger.info("Invoking LangGraph workflow", initial_state_ready=True)
        final_state = self.graph.invoke(initial_state)
        
        # Return the final result - access from the state dict
        result = final_state.get("final_result")
        logger.info("===============================================")
        logger.info("=== WORKFLOW EXECUTION COMPLETE ===")
        logger.info("===============================================")
        logger.info("Final workflow result", 
                   result_type=result.get("response_type") if result else None,
                   status=result.get("status") if result else None,
                   has_sql=bool(result.get("sql")) if result else False,
                   confidence=result.get("confidence_score") if result else None)
        
        return result
    
    async def continue_from_checkpoint(
        self, 
        request_id: str, 
        clarification_response: str
    ) -> Dict[str, Any]:
        """Continue workflow execution from a saved checkpoint after HITL clarification.
        
        This method enables seamless resumption of workflow execution after human
        clarification by restoring the complete workflow state from a checkpoint.
        All intermediate data including schema context, confidence scores, and
        retrieved documents are preserved and restored.
        
        Args:
            request_id: The unique request ID that was saved in the checkpoint
            clarification_response: User's clarification or additional information
            
        Returns:
            Final workflow result with preserved context and continued execution
            
        Raises:
            ValueError: If no checkpoint exists for the given request_id
        """
        logger.info(f"Resuming workflow from checkpoint for request {request_id}")
        
        # Retrieve checkpoint data
        if request_id not in self._checkpoint_storage:
            raise ValueError(f"No checkpoint found for request {request_id}")
        
        checkpoint_data = self._checkpoint_storage[request_id]
        
        # Restore conversation state from checkpoint
        restored_state = ConversationState.restore_from_checkpoint(
            checkpoint_data, 
            clarification_response
        )
        
        # Add the clarification response as a user message
        restored_state.add_message("user", clarification_response, "clarification_response")
        
        # Set request type to clarification to route correctly
        restored_state.request_type = RequestType.CLARIFICATION
        
        # Clear the clarification request since we now have a response
        restored_state.clarification_request = None
        
        logger.info(f"Restored state: conversation_id={restored_state.conversation_id}, workflow_step={restored_state.workflow_step}")
        
        # Resume workflow execution - it will continue from get_metadata or generate_sql
        # depending on the workflow routing logic
        final_state = self.graph.invoke(restored_state)
        
        # Clean up checkpoint after successful continuation
        if request_id in self._checkpoint_storage:
            del self._checkpoint_storage[request_id]
        
        return final_state.final_result

    async def continue_conversation(
        self, 
        conversation_id: str, 
        message: str, 
        conversation_state: ConversationState
    ) -> Dict[str, Any]:
        """Continue an existing conversation with clarification or follow-up (legacy method)."""
        logger.info(f"Continuing conversation {conversation_id} with message: {message}")
        
        # For HITL scenarios, try to find a checkpoint first
        if hasattr(conversation_state, 'request_id') and conversation_state.request_id in self._checkpoint_storage:
            return await self.continue_from_checkpoint(conversation_state.request_id, message)
        
        # Generate new request ID for this continuation
        conversation_state.request_id = str(uuid.uuid4())
        
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