"""
Simplified Text-to-SQL Agent without domain concepts using LangGraph.
"""

import json
import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from ..models.view_models import ViewMetadata, SessionState, HITLRequest
from ..services.view_service import ViewService
from ..services.embedding_service import EmbeddingService, VectorService
from ..services.hitl_service import HITLService

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the text-to-SQL agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    session_id: str
    
    # Context retrieval
    retrieved_views: List[ViewMetadata]
    selected_views: List[str]
    
    # SQL generation
    generated_sql: Optional[str]
    sql_explanation: Optional[str]
    
    # HITL workflow
    hitl_request_id: Optional[str]
    hitl_status: Optional[str]
    hitl_result: Optional[Dict[str, Any]]
    
    # Final result
    query_result: Optional[Any]
    formatted_response: Optional[str]
    
    # Error handling
    error: Optional[str]


class TextToSQLAgent:
    """Simplified text-to-SQL agent without domain concepts."""
    
    def __init__(
        self,
        view_service: ViewService,
        embedding_service: EmbeddingService,
        vector_service: VectorService,
        hitl_service: HITLService,
        llm_service,
        session_service
    ):
        self.view_service = view_service
        self.embedding_service = embedding_service
        self.vector_service = vector_service
        self.hitl_service = hitl_service
        self.llm_service = llm_service
        self.session_service = session_service
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve_views", self.retrieve_views)
        workflow.add_node("generate_sql", self.generate_sql)
        workflow.add_node("request_approval", self.request_approval)
        workflow.add_node("wait_for_approval", self.wait_for_approval)
        workflow.add_node("execute_sql", self.execute_sql)
        workflow.add_node("format_response", self.format_response)
        workflow.add_node("handle_error", self.handle_error)
        
        # Define the flow
        workflow.set_entry_point("retrieve_views")
        
        workflow.add_edge("retrieve_views", "generate_sql")
        workflow.add_edge("generate_sql", "request_approval")
        workflow.add_edge("request_approval", "wait_for_approval")
        
        # Conditional routing after approval
        workflow.add_conditional_edges(
            "wait_for_approval",
            self._approval_router,
            {
                "approved": "execute_sql",
                "rejected": "generate_sql",  # Regenerate SQL
                "expired": "handle_error",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("execute_sql", "format_response")
        workflow.add_edge("format_response", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    async def retrieve_views(self, state: AgentState) -> AgentState:
        """Retrieve relevant views using vector similarity."""
        try:
            user_query = state["user_query"]
            logger.info(f"Retrieving views for query: {user_query}")
            
            # Generate embedding for the query
            query_embedding = await self.embedding_service.get_embedding(user_query)
            
            # Search for similar views
            results = await self.vector_service.search_similar_views(query_embedding, k=5)
            
            retrieved_views = [view for view, score in results]
            selected_view_names = [view.view_name for view in retrieved_views]
            
            logger.info(f"Retrieved {len(retrieved_views)} relevant views: {selected_view_names}")
            
            state["retrieved_views"] = retrieved_views
            state["selected_views"] = selected_view_names
            
            # Add system message with retrieved context
            context_message = self._build_context_message(retrieved_views)
            state["messages"].append(context_message)
            
            return state
            
        except Exception as e:
            logger.error(f"Error retrieving views: {e}")
            state["error"] = f"Failed to retrieve views: {str(e)}"
            return state
    
    async def generate_sql(self, state: AgentState) -> AgentState:
        """Generate SQL query using retrieved views."""
        try:
            user_query = state["user_query"]
            retrieved_views = state["retrieved_views"]
            
            if not retrieved_views:
                raise ValueError("No views available for SQL generation")
            
            # Build prompt for SQL generation
            prompt = self._build_sql_prompt(user_query, retrieved_views)
            
            # Generate SQL using LLM
            response = await self.llm_service.generate_sql(prompt)
            
            # Parse the response
            sql_data = self._parse_sql_response(response)
            
            state["generated_sql"] = sql_data["sql"]
            state["sql_explanation"] = sql_data["explanation"]
            
            # Add AI message with generated SQL
            ai_message = AIMessage(
                content=f"Generated SQL:\n```sql\n{sql_data['sql']}\n```\n\nExplanation: {sql_data['explanation']}"
            )
            state["messages"].append(ai_message)
            
            logger.info("SQL generated successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            state["error"] = f"Failed to generate SQL: {str(e)}"
            return state
    
    async def request_approval(self, state: AgentState) -> AgentState:
        """Create HITL approval request."""
        try:
            session_id = state["session_id"]
            user_query = state["user_query"]
            generated_sql = state["generated_sql"]
            sql_explanation = state["sql_explanation"]
            selected_views = state["selected_views"]
            
            if not generated_sql:
                raise ValueError("No SQL to approve")
            
            # Create approval request
            request_id = await self.hitl_service.create_approval_request(
                session_id=session_id,
                user_query=user_query,
                generated_sql=generated_sql,
                sql_explanation=sql_explanation,
                selected_views=selected_views
            )
            
            state["hitl_request_id"] = request_id
            state["hitl_status"] = "pending"
            
            logger.info(f"Created HITL approval request: {request_id}")
            return state
            
        except Exception as e:
            logger.error(f"Error creating approval request: {e}")
            state["error"] = f"Failed to create approval request: {str(e)}"
            return state
    
    async def wait_for_approval(self, state: AgentState) -> AgentState:
        """Wait for human approval of generated SQL."""
        try:
            request_id = state["hitl_request_id"]
            
            if not request_id:
                raise ValueError("No HITL request ID")
            
            # Wait for approval (this will block until resolved)
            result = await self.hitl_service.wait_for_approval(request_id)
            
            state["hitl_result"] = result
            state["hitl_status"] = result["status"]
            
            logger.info(f"HITL request {request_id} resolved with status: {result['status']}")
            return state
            
        except Exception as e:
            logger.error(f"Error waiting for approval: {e}")
            state["error"] = f"Approval process failed: {str(e)}"
            return state
    
    async def execute_sql(self, state: AgentState) -> AgentState:
        """Execute the approved SQL query."""
        try:
            generated_sql = state["generated_sql"]
            
            if not generated_sql:
                raise ValueError("No SQL to execute")
            
            # For demo purposes, we'll simulate query execution
            # In production, this would call your actual database
            mock_result = self._simulate_query_execution(generated_sql, state["selected_views"])
            
            state["query_result"] = mock_result
            
            logger.info("SQL executed successfully (simulated)")
            return state
            
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            state["error"] = f"Failed to execute SQL: {str(e)}"
            return state
    
    async def format_response(self, state: AgentState) -> AgentState:
        """Format the final response for the user."""
        try:
            query_result = state["query_result"]
            generated_sql = state["generated_sql"]
            sql_explanation = state["sql_explanation"]
            
            # Format response
            response = self._format_final_response(query_result, generated_sql, sql_explanation)
            
            state["formatted_response"] = response
            
            # Add final AI message
            final_message = AIMessage(content=response)
            state["messages"].append(final_message)
            
            # Save session state
            await self._save_session_state(state)
            
            logger.info("Response formatted and session saved")
            return state
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            state["error"] = f"Failed to format response: {str(e)}"
            return state
    
    async def handle_error(self, state: AgentState) -> AgentState:
        """Handle errors in the workflow."""
        error = state.get("error", "Unknown error occurred")
        
        error_message = AIMessage(
            content=f"I encountered an error while processing your request: {error}\n\nPlease try again with a different query or contact support if the issue persists."
        )
        
        state["messages"].append(error_message)
        state["formatted_response"] = error_message.content
        
        # Save session state even on error
        await self._save_session_state(state)
        
        logger.error(f"Handled error: {error}")
        return state
    
    def _approval_router(self, state: AgentState) -> str:
        """Route based on approval result."""
        hitl_result = state.get("hitl_result", {})
        status = hitl_result.get("status", "error")
        
        if status == "approved":
            return "approved"
        elif status == "rejected":
            return "rejected"
        elif status == "expired":
            return "expired"
        else:
            return "error"
    
    def _build_context_message(self, views: List[ViewMetadata]) -> SystemMessage:
        """Build system message with view context."""
        context_parts = ["Available database views for your query:\n"]
        
        for view in views:
            context_parts.append(f"\n**{view.view_name}** ({view.view_type})")
            context_parts.append(f"Description: {view.description}")
            
            if view.columns:
                columns_info = []
                for col in view.columns:
                    col_info = f"{col.name} ({col.type})"
                    if col.notNull:
                        col_info += " NOT NULL"
                    if col.description:
                        col_info += f": {col.description}"
                    columns_info.append(col_info)
                
                context_parts.append("Columns:")
                context_parts.append(", ".join(columns_info))
            
            if view.sample_sql:
                context_parts.append(f"Sample SQL: {view.sample_sql}")
            
            context_parts.append("")
        
        return SystemMessage(content="\n".join(context_parts))
    
    def _build_sql_prompt(self, user_query: str, views: List[ViewMetadata]) -> str:
        """Build prompt for SQL generation."""
        prompt_parts = [
            "You are an expert SQL generator. Generate a SQL query based on the user's request and the available database views.",
            "",
            "IMPORTANT RULES:",
            "1. Only use the provided views and their columns",
            "2. Ensure proper JOIN conditions when using multiple views", 
            "3. Use appropriate WHERE clauses based on the user's criteria",
            "4. Include proper column aliases for clarity",
            "5. Validate that all referenced columns exist in the views",
            "",
            "User Query:",
            user_query,
            "",
            "Available Views:"
        ]
        
        for view in views:
            prompt_parts.append(f"\n{view.view_name} ({view.view_type}):")
            prompt_parts.append(f"Description: {view.description}")
            
            if view.columns:
                prompt_parts.append("Columns:")
                for col in view.columns:
                    col_desc = f"  - {col.name} ({col.type})"
                    if col.notNull:
                        col_desc += " NOT NULL"
                    if col.description:
                        col_desc += f": {col.description}"
                    prompt_parts.append(col_desc)
            
            if view.joins:
                prompt_parts.append("Joins:")
                for join in view.joins:
                    prompt_parts.append(f"  - {join.join_type} JOIN {join.table_name} ON {join.join_condition}")
        
        prompt_parts.extend([
            "",
            "Generate a response in the following JSON format:",
            "{",
            '  "sql": "SELECT ... FROM ...",',
            '  "explanation": "This query does X by joining Y with Z..."',
            "}"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_sql_response(self, response: str) -> Dict[str, str]:
        """Parse SQL response from LLM."""
        try:
            # Try to parse as JSON first
            if response.strip().startswith("{"):
                data = json.loads(response)
                return {
                    "sql": data.get("sql", ""),
                    "explanation": data.get("explanation", "")
                }
            
            # Fallback: extract SQL from markdown code blocks
            sql = ""
            explanation = ""
            
            lines = response.split("\n")
            in_sql_block = False
            sql_lines = []
            
            for line in lines:
                if "```sql" in line.lower():
                    in_sql_block = True
                    continue
                elif "```" in line and in_sql_block:
                    in_sql_block = False
                    continue
                elif in_sql_block:
                    sql_lines.append(line)
                elif not in_sql_block and line.strip():
                    # Consider remaining text as explanation
                    explanation += line + "\n"
            
            sql = "\n".join(sql_lines).strip()
            explanation = explanation.strip()
            
            if not sql:
                # Last resort: assume the whole response is SQL
                sql = response.strip()
                explanation = "Generated SQL query"
            
            return {"sql": sql, "explanation": explanation}
            
        except Exception as e:
            logger.warning(f"Failed to parse SQL response: {e}")
            return {"sql": response, "explanation": "Generated SQL query"}
    
    def _simulate_query_execution(self, sql: str, view_names: List[str]) -> Dict[str, Any]:
        """Simulate query execution for demo purposes."""
        # Generate mock result based on the query
        mock_columns = ["column_1", "column_2", "column_3"]
        mock_rows = [
            ["value_1_1", "value_1_2", "value_1_3"],
            ["value_2_1", "value_2_2", "value_2_3"],
            ["value_3_1", "value_3_2", "value_3_3"]
        ]
        
        return {
            "columns": mock_columns,
            "rows": mock_rows,
            "row_count": len(mock_rows),
            "execution_time_ms": 150,
            "views_used": view_names
        }
    
    def _format_final_response(self, query_result: Dict[str, Any], sql: str, explanation: str) -> str:
        """Format the final response for the user."""
        response_parts = [
            "## Query Results",
            "",
            f"**Generated SQL:**",
            f"```sql",
            sql,
            f"```",
            "",
            f"**Explanation:** {explanation}",
            "",
            f"**Results:** {query_result['row_count']} rows returned in {query_result['execution_time_ms']}ms",
            "",
            "**Data:**"
        ]
        
        # Format the data table
        if query_result.get("columns") and query_result.get("rows"):
            columns = query_result["columns"]
            rows = query_result["rows"]
            
            # Simple table formatting
            response_parts.append("| " + " | ".join(columns) + " |")
            response_parts.append("|" + "|".join(["---" for _ in columns]) + "|")
            
            for row in rows[:10]:  # Limit to first 10 rows
                response_parts.append("| " + " | ".join(str(cell) for cell in row) + " |")
            
            if len(rows) > 10:
                response_parts.append(f"... and {len(rows) - 10} more rows")
        
        return "\n".join(response_parts)
    
    async def _save_session_state(self, state: AgentState):
        """Save session state to database."""
        try:
            session_state = SessionState(
                session_id=state["session_id"],
                current_step="completed",
                user_query=state["user_query"],
                retrieved_views=state.get("retrieved_views", []),
                selected_views=state.get("selected_views", []),
                generated_sql=state.get("generated_sql"),
                sql_explanation=state.get("sql_explanation"),
                hitl_request_id=state.get("hitl_request_id"),
                hitl_status=state.get("hitl_status"),
                query_result=state.get("query_result"),
                formatted_response=state.get("formatted_response")
            )
            
            await self.session_service.save_session(session_state)
            
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")
    
    async def process_query(self, user_query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a user query through the complete workflow."""
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Initialize state
            initial_state = AgentState(
                messages=[HumanMessage(content=user_query)],
                user_query=user_query,
                session_id=session_id,
                retrieved_views=[],
                selected_views=[],
                generated_sql=None,
                sql_explanation=None,
                hitl_request_id=None,
                hitl_status=None,
                hitl_result=None,
                query_result=None,
                formatted_response=None,
                error=None
            )
            
            # Run the workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            return {
                "session_id": session_id,
                "response": final_state.get("formatted_response", "No response generated"),
                "sql": final_state.get("generated_sql"),
                "explanation": final_state.get("sql_explanation"),
                "views_used": final_state.get("selected_views", []),
                "hitl_request_id": final_state.get("hitl_request_id"),
                "error": final_state.get("error")
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "session_id": session_id or "unknown",
                "response": f"An error occurred while processing your query: {str(e)}",
                "error": str(e)
            }