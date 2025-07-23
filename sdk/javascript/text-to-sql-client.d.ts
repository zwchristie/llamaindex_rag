/**
 * TypeScript definitions for Text-to-SQL RAG API Client
 * 
 * Provides type safety and IntelliSense support for the JavaScript client.
 * 
 * @version 1.0.0
 */

// =============================================================================
// CORE TYPES
// =============================================================================

export interface APIConfig {
  baseUrl: string;
  apiKey?: string;
  timeout?: number;
  debug?: boolean;
}

export interface RequestContext {
  user_id?: string;
  session_context?: string;
  database_schema?: string;
  [key: string]: any;
}

export interface APIError {
  detail: string;
  error_code?: string;
  validation_errors?: ValidationError[];
}

export interface ValidationError {
  field: string;
  message: string;
  type: string;
}

// =============================================================================
// CONVERSATION TYPES
// =============================================================================

export type ConversationStatus = 'completed' | 'waiting_for_clarification' | 'active';
export type ResponseType = 'sql_result' | 'clarification_request' | 'error';
export type MessageRole = 'user' | 'assistant';

export interface ConversationMessage {
  role: MessageRole;
  content: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

export interface ClarificationRequest {
  message: string;
  suggestions: string[];
}

export interface ConversationResult {
  response_type: ResponseType;
  sql?: string;
  explanation?: string;
  confidence?: number;
  execution_results?: QueryExecutionResult;
  clarification?: ClarificationRequest;
  status?: ConversationStatus;
  message?: string;
}

export interface ConversationStartResponse {
  conversation_id: string;
  result: ConversationResult;
}

export interface ConversationDetails {
  conversation_id: string;
  initial_query: string;
  status: ConversationStatus;
  created_at: string;
  last_interaction: string;
  messages: ConversationMessage[];
}

export interface ConversationSummary {
  conversation_id: string;
  initial_query: string;
  status: ConversationStatus;
  created_at: string;
  last_interaction: string;
  message_count: number;
}

export interface ConversationsList {
  conversations: ConversationSummary[];
  total: number;
  filtered_by_status?: string;
}

// =============================================================================
// SQL GENERATION TYPES
// =============================================================================

export interface SQLGenerationRequest {
  query: string;
  session_id?: string;
  context?: RequestContext;
}

export interface SQLMetadata {
  tables_used: string[];
  functions_used: string[];
  complexity: 'simple' | 'intermediate' | 'complex';
}

export interface SQLGenerationResponse {
  sql: string;
  explanation: string;
  confidence: number;
  metadata: SQLMetadata;
}

export interface SQLExecutionOptions {
  auto_execute?: boolean;
  execution_limit?: number;
}

export interface SQLExecutionResponse extends SQLGenerationResponse {
  execution_results: QueryExecutionResult;
}

export interface QueryExecutionResult {
  status: 'success' | 'error' | 'timeout';
  execution_time_ms: number;
  rows_returned: number;
  columns: string[];
  data: Record<string, any>[];
  error_message?: string;
  metadata?: Record<string, any>;
}

export interface SQLValidationResult {
  is_valid: boolean;
  errors: Array<{
    type: string;
    message: string;
    line: number;
    column: number;
  }>;
  suggestions: string[];
}

export interface SQLExplanationResult {
  explanation: string;
  breakdown: {
    tables: string[];
    joins: string[];
    aggregations: string[];
    grouping: string[];
  };
  complexity: 'simple' | 'intermediate' | 'complex';
}

// =============================================================================
// DOCUMENT MANAGEMENT TYPES
// =============================================================================

export type DocumentType = 'schema' | 'report';

export interface DocumentMetadata {
  title: string;
  document_type: DocumentType;
  description?: string;
}

export interface DocumentUploadResponse {
  document_id: number;
  title: string;
  document_type: string;
  file_name: string;
  status: string;
  message: string;
}

export interface DocumentSearchRequest {
  query: string;
  document_types?: DocumentType[];
  limit?: number;
  min_similarity?: number;
}

export interface DocumentSearchResult {
  id: string;
  score: number;
  content: string;
  metadata: Record<string, any>;
}

export interface DocumentSearchResponse {
  query: string;
  results: DocumentSearchResult[];
  total_found: number;
}

// =============================================================================
// SYSTEM TYPES
// =============================================================================

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'starting';
  vector_store: string;
  execution_service: string;
  mongodb: string;
  version: string;
  timestamp: string;
}

export interface DetailedHealthStatus {
  status: string;
  services: {
    vector_store: Record<string, any>;
    execution_service: Record<string, any>;
    opensearch: Record<string, any>;
  };
}

export interface SystemStats {
  vector_store: Record<string, any>;
  active_sessions: number;
  active_conversations: number;
  pending_clarifications: number;
  llm_provider: Record<string, any>;
}

// =============================================================================
// LLM PROVIDER TYPES
// =============================================================================

export type LLMProvider = 'bedrock' | 'custom';

export interface LLMProviderInfo {
  provider_info: {
    provider: string;
    model?: string;
    region?: string;
    endpoint?: string;
  };
  health_status: boolean;
  available_providers: string[];
}

export interface LLMProviderSwitchResponse {
  success: boolean;
  message: string;
  provider_info: Record<string, any>;
}

export interface LLMProviderTestResult {
  success: boolean;
  provider: string;
  test_prompt?: string;
  response?: string;
  response_length?: number;
  error?: string;
}

// =============================================================================
// EVENT TYPES
// =============================================================================

export type EventType = 
  | 'conversation-started'
  | 'message-received'
  | 'clarification-needed'
  | 'sql-generated'
  | 'error';

export interface ConversationStartedEvent {
  conversationId: string;
  query: string;
  result: ConversationResult;
}

export interface MessageReceivedEvent {
  result: ConversationResult;
}

export interface ClarificationNeededEvent {
  message: string;
  suggestions: string[];
}

export interface SQLGeneratedEvent {
  sql: string;
  explanation: string;
  confidence: number;
  results?: QueryExecutionResult;
}

export type EventCallback<T = any> = (data: T) => void;

// =============================================================================
// CHAT UI TYPES
// =============================================================================

export interface ChatMessage {
  role: MessageRole;
  content: string | ConversationResult;
  timestamp: Date;
  metadata?: Record<string, any>;
}

export interface ChatResponse {
  type: 'sql_result' | 'clarification' | 'error';
  timestamp: Date;
  conversationId: string;
  sql?: string;
  explanation?: string;
  confidence?: number;
  canExecute: boolean;
  results?: QueryExecutionResult;
  message?: string;
  suggestions?: string[];
}

// =============================================================================
// CLIENT CLASS DEFINITIONS
// =============================================================================

export class TextToSQLError extends Error {
  status: number;
  code: string;
  
  constructor(message: string, status: number, code: string);
}

export class TextToSQLClient {
  readonly baseUrl: string;
  readonly timeout: number;
  readonly debug: boolean;
  currentConversation: string | null;
  sessionId: string | null;

  constructor(config: APIConfig);

  // Event Management
  on<T>(event: EventType, callback: EventCallback<T>): void;
  off<T>(event: EventType, callback: EventCallback<T>): void;

  // Conversation Management
  startConversation(query: string, context?: RequestContext): Promise<ConversationResult>;
  continueConversation(message: string, conversationId?: string): Promise<ConversationResult>;
  getConversation(conversationId?: string): Promise<ConversationDetails>;
  listConversations(options?: {
    status?: ConversationStatus;
    limit?: number;
  }): Promise<ConversationsList>;
  deleteConversation(conversationId: string): Promise<{ message: string }>;
  resetConversation(): void;

  // SQL Generation
  generateSQL(query: string, options?: Partial<SQLGenerationRequest>): Promise<SQLGenerationResponse>;
  generateAndExecuteSQL(query: string, options?: SQLExecutionOptions): Promise<SQLExecutionResponse>;
  executeSQL(sqlQuery: string, metadata?: Record<string, any>): Promise<QueryExecutionResult>;
  validateSQL(sqlQuery: string): Promise<SQLValidationResult>;
  explainSQL(sqlQuery: string): Promise<SQLExplanationResult>;

  // Document Management
  uploadDocument(file: File, metadata: DocumentMetadata): Promise<DocumentUploadResponse>;
  searchDocuments(query: string, options?: Partial<DocumentSearchRequest>): Promise<DocumentSearchResponse>;

  // System Information
  getHealth(): Promise<HealthStatus>;
  getDetailedHealth(): Promise<DetailedHealthStatus>;
  getStats(): Promise<SystemStats>;

  // LLM Provider Management
  getLLMProviderInfo(): Promise<LLMProviderInfo>;
  switchLLMProvider(provider: LLMProvider): Promise<LLMProviderSwitchResponse>;
  testLLMProvider(): Promise<LLMProviderTestResult>;

  // Session Management
  setSession(sessionId: string): void;
  getSession(): string | null;
  clearSession(): void;

  // Utility Methods
  isAvailable(): Promise<boolean>;
  waitForAvailability(timeout?: number, interval?: number): Promise<boolean>;
}

export class TextToSQLChat {
  readonly client: TextToSQLClient;
  messageHistory: ChatMessage[];
  isWaitingForResponse: boolean;

  constructor(client: TextToSQLClient);

  sendMessage(message: string): Promise<ChatResponse>;
  getHistory(): ChatMessage[];
  reset(): void;
}

// =============================================================================
// UTILITY TYPES
// =============================================================================

export interface ListConversationsOptions {
  status?: ConversationStatus;
  limit?: number;
}

export interface SearchDocumentsOptions extends Partial<DocumentSearchRequest> {}

export interface WaitForAvailabilityOptions {
  timeout?: number;
  interval?: number;
}

// =============================================================================
// EXPORTS
// =============================================================================

export { TextToSQLClient as default };

declare global {
  interface Window {
    TextToSQLClient: typeof TextToSQLClient;
    TextToSQLChat: typeof TextToSQLChat;
    TextToSQLError: typeof TextToSQLError;
  }
}

// =============================================================================
// REACT TYPES (if using React)
// =============================================================================

export interface UseTextToSQLResult {
  sendMessage: (message: string) => Promise<ConversationResult>;
  resetConversation: () => void;
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  conversationId: string | null;
}

// =============================================================================
// CONFIGURATION PRESETS
// =============================================================================

export interface ClientPresets {
  development: APIConfig;
  production: APIConfig;
  testing: APIConfig;
}

export const presets: ClientPresets;