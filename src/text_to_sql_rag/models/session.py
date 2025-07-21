"""Session models for user interaction tracking."""

from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean
from sqlalchemy.sql import func

from .database import Base
from .simple_models import HumanInterventionRequest


class SessionStatus(str, Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    PENDING_HUMAN = "pending_human"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class InteractionType(str, Enum):
    """Types of user interactions."""
    QUERY = "query"
    FEEDBACK = "feedback"
    CLARIFICATION = "clarification"
    APPROVAL = "approval"
    MODIFICATION = "modification"


class UserSession(Base):
    """Database model for user sessions."""
    
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(String(255), nullable=True, index=True)  # Optional user identification
    status = Column(String(50), default=SessionStatus.ACTIVE, index=True)
    
    # Session context
    initial_query = Column(Text, nullable=False)
    context = Column(JSON, default={})  # Store conversation context
    
    # Progress tracking
    current_step = Column(String(100), nullable=True)
    total_steps = Column(Integer, default=1)
    completed_steps = Column(Integer, default=0)
    
    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Results
    final_result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)


class SessionInteraction(Base):
    """Database model for session interactions."""
    
    __tablename__ = "session_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    interaction_type = Column(String(50), nullable=False)
    
    # Interaction content
    user_input = Column(Text, nullable=True)
    system_response = Column(Text, nullable=True)
    metadata = Column(JSON, default={})
    
    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    response_time_ms = Column(Integer, nullable=True)
    
    # Human-in-the-loop tracking
    requires_human = Column(Boolean, default=False)
    human_responded = Column(Boolean, default=False)
    human_response_at = Column(DateTime(timezone=True), nullable=True)


# Pydantic models for API

class SessionCreate(BaseModel):
    """Model for creating a new session."""
    initial_query: str
    user_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    expires_in_minutes: int = Field(default=60, ge=5, le=1440)  # 5 min to 24 hours


class SessionUpdate(BaseModel):
    """Model for updating session status."""
    status: Optional[SessionStatus] = None
    current_step: Optional[str] = None
    completed_steps: Optional[int] = None
    context: Optional[Dict[str, Any]] = None
    final_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class InteractionCreate(BaseModel):
    """Model for creating a new interaction."""
    interaction_type: InteractionType
    user_input: Optional[str] = None
    system_response: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    requires_human: bool = False


class SessionResponse(BaseModel):
    """Response model for session queries."""
    id: int
    session_id: str
    user_id: Optional[str]
    status: SessionStatus
    initial_query: str
    context: Dict[str, Any]
    current_step: Optional[str]
    total_steps: int
    completed_steps: int
    created_at: datetime
    last_activity: datetime
    completed_at: Optional[datetime]
    expires_at: Optional[datetime]
    final_result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    
    class Config:
        from_attributes = True


class InteractionResponse(BaseModel):
    """Response model for interaction queries."""
    id: int
    session_id: str
    interaction_type: InteractionType
    user_input: Optional[str]
    system_response: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime
    response_time_ms: Optional[int]
    requires_human: bool
    human_responded: bool
    human_response_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class SessionWithInteractions(BaseModel):
    """Session with its interactions."""
    session: SessionResponse
    interactions: List[InteractionResponse]


class HumanInterventionResponse(BaseModel):
    """Response from human intervention."""
    action: str
    feedback: Optional[str] = None
    modifications: Dict[str, Any] = Field(default_factory=dict)
    continue_processing: bool = True