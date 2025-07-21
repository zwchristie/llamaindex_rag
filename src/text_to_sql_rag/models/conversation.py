"""Models for conversation state and human-in-the-loop interactions."""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field


class RequestType(str, Enum):
    """Types of user requests."""
    GENERATE_NEW = "generate_new"
    EXECUTE_SQL = "execute_sql"
    EDIT_SQL = "edit_sql"
    DESCRIBE_SQL = "describe_sql"
    EDIT_PREVIOUS = "edit_previous"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"


class ConversationStatus(str, Enum):
    """Status of conversation."""
    ACTIVE = "active"
    WAITING_FOR_CLARIFICATION = "waiting_for_clarification"
    COMPLETED = "completed"
    ERROR = "error"


class WorkflowStep(str, Enum):
    """Workflow steps in the agent."""
    CLASSIFY_REQUEST = "classify_request"
    GET_METADATA = "get_metadata"
    ASSESS_CONFIDENCE = "assess_confidence"
    REQUEST_CLARIFICATION = "request_clarification"
    GENERATE_SQL = "generate_sql"
    EXECUTE_SQL = "execute_sql"
    FIX_SQL = "fix_sql"
    DESCRIBE_SQL = "describe_sql"
    RETURN_RESULTS = "return_results"


class ConversationMessage(BaseModel):
    """Individual message in conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_type: Optional[str] = None  # "question", "clarification", "sql_result", etc.
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SQLArtifact(BaseModel):
    """SQL artifact with metadata."""
    sql: str
    explanation: str
    confidence: float
    tables_used: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    execution_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class ClarificationRequest(BaseModel):
    """Request for human clarification."""
    question: str
    context: str
    missing_info: List[str]
    suggested_tables: List[str] = Field(default_factory=list)
    confidence_issues: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ConversationState(BaseModel):
    """Enhanced conversation state for agent workflows."""
    
    # Basic conversation info
    conversation_id: str
    session_id: Optional[str] = None
    status: ConversationStatus = ConversationStatus.ACTIVE
    
    # Current request processing
    current_request: str
    request_type: Optional[RequestType] = None
    workflow_step: Optional[WorkflowStep] = None
    
    # Context and history
    messages: List[ConversationMessage] = Field(default_factory=list)
    conversation_context: Dict[str, Any] = Field(default_factory=dict)
    
    # SQL-related state
    current_sql: Optional[str] = None
    sql_history: List[SQLArtifact] = Field(default_factory=list)
    
    # Metadata retrieval
    schema_context: List[Dict[str, Any]] = Field(default_factory=list)
    example_context: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    
    # Confidence and clarification
    confidence_score: float = 0.0
    needs_clarification: bool = False
    clarification_request: Optional[ClarificationRequest] = None
    
    # Execution state
    execution_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Workflow control
    should_execute: bool = False
    requires_human_input: bool = False
    
    # Final result
    final_result: Optional[Dict[str, Any]] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def add_message(self, role: str, content: str, message_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation."""
        message = ConversationMessage(
            role=role,
            content=content,
            message_type=message_type,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
    
    def add_sql_artifact(self, sql: str, explanation: str, confidence: float, tables_used: Optional[List[str]] = None):
        """Add a SQL artifact to history."""
        artifact = SQLArtifact(
            sql=sql,
            explanation=explanation,
            confidence=confidence,
            tables_used=tables_used or []
        )
        self.sql_history.append(artifact)
        self.current_sql = sql
        self.updated_at = datetime.utcnow()
    
    def get_recent_sql(self) -> Optional[SQLArtifact]:
        """Get the most recent SQL artifact."""
        return self.sql_history[-1] if self.sql_history else None
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation for context."""
        if not self.messages:
            return "No previous conversation."
        
        summary_parts = []
        for msg in self.messages[-5:]:  # Last 5 messages
            summary_parts.append(f"{msg.role}: {msg.content[:100]}...")
        
        return "\n".join(summary_parts)


class AgentResponse(BaseModel):
    """Response from the agent."""
    
    conversation_id: str
    status: ConversationStatus
    response_type: str  # "sql_result", "clarification_request", "error", "follow_up"
    
    # Main response content
    content: str
    
    # SQL-related data
    sql: Optional[str] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = None
    execution_result: Optional[Dict[str, Any]] = None
    
    # Clarification data
    clarification_request: Optional[ClarificationRequest] = None
    
    # Metadata
    sources: List[str] = Field(default_factory=list)
    tables_used: List[str] = Field(default_factory=list)
    processing_info: Dict[str, Any] = Field(default_factory=dict)
    
    # Workflow info
    requires_follow_up: bool = False
    next_action: Optional[str] = None
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)