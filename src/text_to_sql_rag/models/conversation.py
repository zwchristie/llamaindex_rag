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
    GET_LOOKUP_DATA = "get_lookup_data"
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


class WorkflowState(BaseModel):
    """Serializable workflow execution state for HITL persistence.
    
    This class encapsulates all intermediate workflow data that needs to be preserved
    during human-in-the-loop clarification requests. When the workflow is interrupted
    for clarification, this state is serialized and stored as a checkpoint, allowing
    the workflow to resume exactly where it left off with complete context.
    
    Key Features:
    - Complete intermediate workflow data preservation
    - Serialization/deserialization for checkpoint storage
    - Separation of request-level state from conversation-level state
    - All data needed to resume LangGraph workflow execution
    """
    
    # Workflow execution context
    workflow_step: Optional[WorkflowStep] = None
    # LEGACY: These fields are maintained for compatibility but replaced by hierarchical context
    schema_context: List[Dict[str, Any]] = Field(default_factory=list)  # LEGACY: Now handled by hierarchical service
    example_context: List[Dict[str, Any]] = Field(default_factory=list)  # LEGACY: Now integrated into hierarchical tiers
    lookup_context: List[Dict[str, Any]] = Field(default_factory=list)  # Still used but populated differently
    sources: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    needs_clarification: bool = False
    clarification_request: Optional[ClarificationRequest] = None
    
    # SQL generation context
    current_sql: Optional[str] = None
    sql_history: List[SQLArtifact] = Field(default_factory=list)
    execution_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Control flags
    should_execute: bool = False
    requires_human_input: bool = False
    
    # Context for resumption
    intermediate_data: Dict[str, Any] = Field(default_factory=dict)
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize workflow state for storage."""
        return self.dict()
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "WorkflowState":
        """Deserialize workflow state from storage."""
        return cls(**data)


class ConversationState(BaseModel):
    """Enhanced conversation state for agent workflows with HITL state management.
    
    This class manages the complete state of a conversation with advanced human-in-the-loop
    capabilities. It implements a dual ID system to separate conversation threads from
    individual request fulfillment, enabling proper state persistence and resumption.
    
    Architecture:
    - conversation_id: Tracks the entire conversation thread with multiple exchanges
    - request_id: Tracks individual request fulfillment within a conversation
    - workflow_state: Contains all intermediate workflow data for checkpoint preservation
    - message_history: Maintains conversation-level message thread continuity
    
    Key HITL Features:
    - Checkpoint-based state persistence before clarification requests
    - Complete workflow context restoration after human input
    - Intelligent routing between clarification responses and new requests
    - Backward compatibility through property delegation
    
    Usage:
    - Start conversation: Creates new conversation_id and initial request_id
    - Request clarification: Saves checkpoint with current workflow_state
    - Resume from checkpoint: Restores exact context and continues workflow
    - New request in thread: Generates new request_id while preserving conversation_id
    """
    
    # Basic conversation info - separate IDs for conversation vs request tracking
    conversation_id: str  # Tracks the entire conversation thread
    request_id: str  # Tracks individual request fulfillment within conversation
    session_id: Optional[str] = None  # Optional user session tracking
    status: ConversationStatus = ConversationStatus.ACTIVE
    
    # Current request processing
    current_request: str
    request_type: Optional[RequestType] = None
    
    # Context and history (conversation-level)
    message_history: List[ConversationMessage] = Field(default_factory=list)
    conversation_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Workflow state (request-level, serializable for HITL)
    workflow_state: WorkflowState = Field(default_factory=WorkflowState)
    
    # Legacy fields for backward compatibility (delegated to workflow_state)
    max_retries: int = 3
    
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
        self.message_history.append(message)
        self.updated_at = datetime.utcnow()
    
    def add_sql_artifact(self, sql: str, explanation: str, confidence: float, tables_used: Optional[List[str]] = None):
        """Add a SQL artifact to history."""
        artifact = SQLArtifact(
            sql=sql,
            explanation=explanation,
            confidence=confidence,
            tables_used=tables_used or []
        )
        self.workflow_state.sql_history.append(artifact)
        self.workflow_state.current_sql = sql
        self.updated_at = datetime.utcnow()
    
    def get_recent_sql(self) -> Optional[SQLArtifact]:
        """Get the most recent SQL artifact."""
        return self.workflow_state.sql_history[-1] if self.workflow_state.sql_history else None
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation for context."""
        if not self.message_history:
            return "No previous conversation."
        
        summary_parts = []
        for msg in self.message_history[-5:]:  # Last 5 messages
            summary_parts.append(f"{msg.role}: {msg.content[:100]}...")
        
        return "\n".join(summary_parts)
    
    # Property delegations for backward compatibility
    @property
    def workflow_step(self) -> Optional[WorkflowStep]:
        return self.workflow_state.workflow_step
    
    @workflow_step.setter
    def workflow_step(self, value: Optional[WorkflowStep]):
        self.workflow_state.workflow_step = value
    
    @property
    def schema_context(self) -> List[Dict[str, Any]]:
        return self.workflow_state.schema_context
    
    @schema_context.setter
    def schema_context(self, value: List[Dict[str, Any]]):
        self.workflow_state.schema_context = value
    
    @property
    def example_context(self) -> List[Dict[str, Any]]:
        return self.workflow_state.example_context
    
    @example_context.setter
    def example_context(self, value: List[Dict[str, Any]]):
        self.workflow_state.example_context = value
    
    @property
    def lookup_context(self) -> List[Dict[str, Any]]:
        return self.workflow_state.lookup_context
    
    @lookup_context.setter
    def lookup_context(self, value: List[Dict[str, Any]]):
        self.workflow_state.lookup_context = value
    
    @property
    def sources(self) -> List[str]:
        return self.workflow_state.sources
    
    @sources.setter
    def sources(self, value: List[str]):
        self.workflow_state.sources = value
    
    @property
    def confidence_score(self) -> float:
        return self.workflow_state.confidence_score
    
    @confidence_score.setter
    def confidence_score(self, value: float):
        self.workflow_state.confidence_score = value
    
    @property
    def needs_clarification(self) -> bool:
        return self.workflow_state.needs_clarification
    
    @needs_clarification.setter
    def needs_clarification(self, value: bool):
        self.workflow_state.needs_clarification = value
    
    @property
    def clarification_request(self) -> Optional[ClarificationRequest]:
        return self.workflow_state.clarification_request
    
    @clarification_request.setter
    def clarification_request(self, value: Optional[ClarificationRequest]):
        self.workflow_state.clarification_request = value
    
    @property
    def current_sql(self) -> Optional[str]:
        return self.workflow_state.current_sql
    
    @current_sql.setter
    def current_sql(self, value: Optional[str]):
        self.workflow_state.current_sql = value
    
    @property
    def sql_history(self) -> List[SQLArtifact]:
        return self.workflow_state.sql_history
    
    @property
    def execution_result(self) -> Optional[Dict[str, Any]]:
        return self.workflow_state.execution_result
    
    @execution_result.setter
    def execution_result(self, value: Optional[Dict[str, Any]]):
        self.workflow_state.execution_result = value
    
    @property
    def error_message(self) -> Optional[str]:
        return self.workflow_state.error_message
    
    @error_message.setter
    def error_message(self, value: Optional[str]):
        self.workflow_state.error_message = value
    
    @property
    def retry_count(self) -> int:
        return self.workflow_state.retry_count
    
    @retry_count.setter
    def retry_count(self, value: int):
        self.workflow_state.retry_count = value
    
    @property
    def should_execute(self) -> bool:
        return self.workflow_state.should_execute
    
    @should_execute.setter
    def should_execute(self, value: bool):
        self.workflow_state.should_execute = value
    
    @property
    def requires_human_input(self) -> bool:
        return self.workflow_state.requires_human_input
    
    @requires_human_input.setter
    def requires_human_input(self, value: bool):
        self.workflow_state.requires_human_input = value
    
    @property
    def messages(self) -> List[ConversationMessage]:
        """Backward compatibility alias for message_history."""
        return self.message_history
    
    def save_workflow_checkpoint(self) -> Dict[str, Any]:
        """Save current workflow state for HITL resumption.
        
        Creates a complete checkpoint of the current workflow state that can be
        stored and later used to resume execution after human clarification.
        This preserves all intermediate data including schema context, confidence
        scores, retrieved documents, and workflow step information.
        
        Returns:
            Dict containing serialized checkpoint data with all necessary information
            to restore the exact workflow state for seamless resumption.
        """
        return {
            "conversation_id": self.conversation_id,
            "request_id": self.request_id,
            "current_request": self.current_request,
            "request_type": self.request_type.value if self.request_type else None,
            "workflow_state": self.workflow_state.serialize(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def restore_from_checkpoint(cls, checkpoint_data: Dict[str, Any], new_request: str) -> "ConversationState":
        """Restore conversation state from checkpoint for HITL continuation.
        
        Reconstructs the complete conversation state from a previously saved checkpoint,
        enabling seamless resumption of workflow execution after human clarification.
        All intermediate data is restored including schema context, confidence scores,
        and workflow step information.
        
        Args:
            checkpoint_data: Previously saved checkpoint data from save_workflow_checkpoint()
            new_request: The user's clarification response or new message
            
        Returns:
            Fully restored ConversationState with preserved context and reset HITL flags
            ready for workflow continuation.
        """
        workflow_state = WorkflowState.deserialize(checkpoint_data["workflow_state"])
        
        # Parse datetime fields
        created_at = datetime.fromisoformat(checkpoint_data["created_at"])
        updated_at = datetime.fromisoformat(checkpoint_data["updated_at"])
        
        state = cls(
            conversation_id=checkpoint_data["conversation_id"],
            request_id=checkpoint_data["request_id"],
            current_request=new_request,  # New request from user
            request_type=RequestType(checkpoint_data["request_type"]) if checkpoint_data["request_type"] else None,
            workflow_state=workflow_state,
            status=ConversationStatus(checkpoint_data["status"]),
            created_at=created_at,
            updated_at=updated_at
        )
        
        # Reset HITL flags for continuation
        state.requires_human_input = False
        state.needs_clarification = False
        state.status = ConversationStatus.ACTIVE
        
        return state


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