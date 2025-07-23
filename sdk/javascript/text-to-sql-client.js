/**
 * Text-to-SQL RAG API Client
 * 
 * A comprehensive JavaScript client for integrating with the Text-to-SQL RAG API.
 * Supports conversation management, SQL generation, and real-time chat integration.
 * 
 * @version 1.0.0
 * @author Text-to-SQL RAG Team
 */

class TextToSQLError extends Error {
  constructor(message, status, code) {
    super(message);
    this.name = 'TextToSQLError';
    this.status = status;
    this.code = code;
  }
}

class TextToSQLClient {
  /**
   * Initialize the Text-to-SQL API client
   * 
   * @param {Object} config - Configuration options
   * @param {string} config.baseUrl - API base URL (e.g., 'http://localhost:8000')
   * @param {string} [config.apiKey] - API key for authentication (if required)
   * @param {number} [config.timeout=30000] - Request timeout in milliseconds
   * @param {boolean} [config.debug=false] - Enable debug logging
   */
  constructor(config) {
    this.baseUrl = config.baseUrl.replace(/\/$/, ''); // Remove trailing slash
    this.apiKey = config.apiKey;
    this.timeout = config.timeout || 30000;
    this.debug = config.debug || false;
    
    // Internal state
    this.currentConversation = null;
    this.sessionId = null;
    
    // Event listeners
    this.listeners = {
      'conversation-started': [],
      'message-received': [],
      'clarification-needed': [],
      'sql-generated': [],
      'error': []
    };
  }

  /**
   * Add event listener
   * 
   * @param {string} event - Event name
   * @param {Function} callback - Event callback
   */
  on(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event].push(callback);
    }
  }

  /**
   * Remove event listener
   * 
   * @param {string} event - Event name
   * @param {Function} callback - Event callback to remove
   */
  off(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
    }
  }

  /**
   * Emit event to listeners
   * 
   * @private
   * @param {string} event - Event name
   * @param {*} data - Event data
   */
  _emit(event, data) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(callback => callback(data));
    }
  }

  /**
   * Make HTTP request to the API
   * 
   * @private
   * @param {string} endpoint - API endpoint
   * @param {Object} options - Fetch options
   * @returns {Promise<Object>} Response data
   */
  async _request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    if (this.sessionId) {
      headers['X-Session-ID'] = this.sessionId;
    }

    const requestOptions = {
      timeout: this.timeout,
      ...options,
      headers
    };

    if (this.debug) {
      console.log(`[TextToSQL] ${options.method || 'GET'} ${url}`, requestOptions);
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(url, {
        ...requestOptions,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new TextToSQLError(
          errorData.detail || `HTTP ${response.status}`,
          response.status,
          errorData.error_code
        );
      }

      const data = await response.json();
      
      if (this.debug) {
        console.log(`[TextToSQL] Response:`, data);
      }

      return data;
    } catch (error) {
      if (error.name === 'AbortError') {
        throw new TextToSQLError('Request timeout', 408, 'TIMEOUT');
      } else if (error instanceof TextToSQLError) {
        throw error;
      } else {
        throw new TextToSQLError(`Network error: ${error.message}`, 0, 'NETWORK_ERROR');
      }
    }
  }

  // =============================================================================
  // CONVERSATION MANAGEMENT
  // =============================================================================

  /**
   * Start a new conversation
   * 
   * @param {string} query - Natural language query
   * @param {Object} [context] - Additional context
   * @returns {Promise<Object>} Conversation result
   */
  async startConversation(query, context = {}) {
    try {
      const response = await this._request('/conversations/start', {
        method: 'POST',
        body: JSON.stringify({
          query,
          context: {
            user_id: this.sessionId,
            ...context
          }
        })
      });

      this.currentConversation = response.conversation_id;
      
      this._emit('conversation-started', {
        conversationId: response.conversation_id,
        query,
        result: response.result
      });

      return this._processConversationResult(response.result);
    } catch (error) {
      this._emit('error', error);
      throw error;
    }
  }

  /**
   * Continue an existing conversation
   * 
   * @param {string} message - User's response or clarification
   * @param {string} [conversationId] - Conversation ID (uses current if not provided)
   * @returns {Promise<Object>} Conversation result
   */
  async continueConversation(message, conversationId = null) {
    const convId = conversationId || this.currentConversation;
    
    if (!convId) {
      throw new TextToSQLError('No active conversation. Start a new conversation first.', 400, 'NO_CONVERSATION');
    }

    try {
      const response = await this._request(`/conversations/${convId}/continue`, {
        method: 'POST',
        body: JSON.stringify({ message })
      });

      return this._processConversationResult(response);
    } catch (error) {
      this._emit('error', error);
      throw error;
    }
  }

  /**
   * Get conversation details and history
   * 
   * @param {string} [conversationId] - Conversation ID (uses current if not provided)
   * @returns {Promise<Object>} Conversation details
   */
  async getConversation(conversationId = null) {
    const convId = conversationId || this.currentConversation;
    
    if (!convId) {
      throw new TextToSQLError('No conversation ID provided', 400, 'NO_CONVERSATION_ID');
    }

    return await this._request(`/conversations/${convId}`);
  }

  /**
   * List user's conversations
   * 
   * @param {Object} [options] - Filter options
   * @param {string} [options.status] - Filter by status
   * @param {number} [options.limit=20] - Maximum number of conversations
   * @returns {Promise<Object>} Conversations list
   */
  async listConversations(options = {}) {
    const params = new URLSearchParams();
    
    if (options.status) {
      params.append('status', options.status);
    }
    
    if (options.limit) {
      params.append('limit', options.limit.toString());
    }

    const query = params.toString() ? `?${params.toString()}` : '';
    return await this._request(`/conversations${query}`);
  }

  /**
   * Delete a conversation
   * 
   * @param {string} conversationId - Conversation ID to delete
   * @returns {Promise<Object>} Deletion result
   */
  async deleteConversation(conversationId) {
    const response = await this._request(`/conversations/${conversationId}`, {
      method: 'DELETE'
    });

    if (conversationId === this.currentConversation) {
      this.currentConversation = null;
    }

    return response;
  }

  /**
   * Reset current conversation
   */
  resetConversation() {
    this.currentConversation = null;
  }

  /**
   * Process conversation result and emit appropriate events
   * 
   * @private
   * @param {Object} result - Conversation result
   * @returns {Object} Processed result
   */
  _processConversationResult(result) {
    this._emit('message-received', result);

    switch (result.response_type) {
      case 'sql_result':
        this._emit('sql-generated', {
          sql: result.sql,
          explanation: result.explanation,
          confidence: result.confidence,
          results: result.execution_results
        });
        break;
      
      case 'clarification_request':
        this._emit('clarification-needed', {
          message: result.clarification.message,
          suggestions: result.clarification.suggestions
        });
        break;
      
      case 'error':
        this._emit('error', new TextToSQLError(result.message, 422, 'CONVERSATION_ERROR'));
        break;
    }

    return result;
  }

  // =============================================================================
  // DIRECT SQL GENERATION
  // =============================================================================

  /**
   * Generate SQL from natural language (without conversation)
   * 
   * @param {string} query - Natural language query
   * @param {Object} [options] - Generation options
   * @returns {Promise<Object>} SQL generation result
   */
  async generateSQL(query, options = {}) {
    return await this._request('/query/generate', {
      method: 'POST',
      body: JSON.stringify({
        query,
        session_id: this.sessionId,
        ...options
      })
    });
  }

  /**
   * Generate and execute SQL
   * 
   * @param {string} query - Natural language query
   * @param {Object} [options] - Generation and execution options
   * @param {boolean} [options.auto_execute=true] - Auto-execute if confidence is high
   * @param {number} [options.execution_limit=100] - Maximum rows to return
   * @returns {Promise<Object>} SQL generation and execution result
   */
  async generateAndExecuteSQL(query, options = {}) {
    return await this._request('/query/generate-and-execute', {
      method: 'POST',
      body: JSON.stringify({
        query,
        auto_execute: options.auto_execute !== false,
        execution_limit: options.execution_limit || 100,
        ...options
      })
    });
  }

  /**
   * Execute SQL query directly
   * 
   * @param {string} sqlQuery - SQL query to execute
   * @param {Object} [metadata] - Additional metadata
   * @returns {Promise<Object>} Execution result
   */
  async executeSQL(sqlQuery, metadata = {}) {
    return await this._request('/query/execute', {
      method: 'POST',
      body: JSON.stringify({
        sql_query: sqlQuery,
        session_id: this.sessionId,
        metadata
      })
    });
  }

  /**
   * Validate SQL query syntax
   * 
   * @param {string} sqlQuery - SQL query to validate
   * @returns {Promise<Object>} Validation result
   */
  async validateSQL(sqlQuery) {
    return await this._request('/query/validate', {
      method: 'POST',
      body: JSON.stringify({ sql_query: sqlQuery })
    });
  }

  /**
   * Get explanation for SQL query
   * 
   * @param {string} sqlQuery - SQL query to explain
   * @returns {Promise<Object>} Explanation result
   */
  async explainSQL(sqlQuery) {
    return await this._request('/query/explain', {
      method: 'POST',
      body: JSON.stringify({ sql_query: sqlQuery })
    });
  }

  // =============================================================================
  // DOCUMENT MANAGEMENT
  // =============================================================================

  /**
   * Upload document for RAG context
   * 
   * @param {File} file - File to upload
   * @param {Object} metadata - Document metadata
   * @param {string} metadata.title - Document title
   * @param {string} metadata.document_type - Document type ('schema' or 'report')
   * @param {string} [metadata.description] - Document description
   * @returns {Promise<Object>} Upload result
   */
  async uploadDocument(file, metadata) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', metadata.title);
    formData.append('document_type', metadata.document_type);
    
    if (metadata.description) {
      formData.append('description', metadata.description);
    }

    return await this._request('/documents/upload', {
      method: 'POST',
      headers: {}, // Remove Content-Type to let browser set it for FormData
      body: formData
    });
  }

  /**
   * Search documents
   * 
   * @param {string} query - Search query
   * @param {Object} [options] - Search options
   * @param {Array<string>} [options.document_types] - Filter by document types
   * @param {number} [options.limit=10] - Maximum results
   * @param {number} [options.min_similarity=0.7] - Minimum similarity threshold
   * @returns {Promise<Object>} Search results
   */
  async searchDocuments(query, options = {}) {
    return await this._request('/search/documents', {
      method: 'POST',
      body: JSON.stringify({
        query,
        limit: options.limit || 10,
        min_similarity: options.min_similarity || 0.7,
        ...options
      })
    });
  }

  // =============================================================================
  // SYSTEM INFORMATION
  // =============================================================================

  /**
   * Get API health status
   * 
   * @returns {Promise<Object>} Health status
   */
  async getHealth() {
    return await this._request('/health');
  }

  /**
   * Get detailed health information
   * 
   * @returns {Promise<Object>} Detailed health status
   */
  async getDetailedHealth() {
    return await this._request('/health/detailed');
  }

  /**
   * Get system statistics
   * 
   * @returns {Promise<Object>} System statistics
   */
  async getStats() {
    return await this._request('/stats');
  }

  // =============================================================================
  // LLM PROVIDER MANAGEMENT
  // =============================================================================

  /**
   * Get current LLM provider information
   * 
   * @returns {Promise<Object>} Provider information
   */
  async getLLMProviderInfo() {
    return await this._request('/llm-provider/info');
  }

  /**
   * Switch LLM provider
   * 
   * @param {string} provider - Provider name ('bedrock' or 'custom')
   * @returns {Promise<Object>} Switch result
   */
  async switchLLMProvider(provider) {
    return await this._request('/llm-provider/switch', {
      method: 'POST',
      body: JSON.stringify({ provider })
    });
  }

  /**
   * Test current LLM provider
   * 
   * @returns {Promise<Object>} Test result
   */
  async testLLMProvider() {
    return await this._request('/llm-provider/test');
  }

  // =============================================================================
  // SESSION MANAGEMENT
  // =============================================================================

  /**
   * Set session ID for requests
   * 
   * @param {string} sessionId - Session identifier
   */
  setSession(sessionId) {
    this.sessionId = sessionId;
  }

  /**
   * Get current session ID
   * 
   * @returns {string|null} Current session ID
   */
  getSession() {
    return this.sessionId;
  }

  /**
   * Clear session
   */
  clearSession() {
    this.sessionId = null;
    this.currentConversation = null;
  }

  // =============================================================================
  // UTILITY METHODS
  // =============================================================================

  /**
   * Check if API is available
   * 
   * @returns {Promise<boolean>} API availability status
   */
  async isAvailable() {
    try {
      const health = await this.getHealth();
      return health.status === 'healthy' || health.status === 'degraded';
    } catch (error) {
      return false;
    }
  }

  /**
   * Wait for API to become available
   * 
   * @param {number} [timeout=30000] - Timeout in milliseconds
   * @param {number} [interval=1000] - Check interval in milliseconds
   * @returns {Promise<boolean>} Whether API became available
   */
  async waitForAvailability(timeout = 30000, interval = 1000) {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      if (await this.isAvailable()) {
        return true;
      }
      
      await new Promise(resolve => setTimeout(resolve, interval));
    }
    
    return false;
  }
}

// =============================================================================
// CHAT UI HELPER CLASSES
// =============================================================================

/**
 * High-level chat interface for UI integration
 */
class TextToSQLChat {
  constructor(client) {
    this.client = client;
    this.messageHistory = [];
    this.isWaitingForResponse = false;
    
    // Set up event listeners
    this.client.on('message-received', (result) => {
      this.messageHistory.push({
        role: 'assistant',
        content: this._formatAssistantMessage(result),
        timestamp: new Date(),
        metadata: result
      });
    });
  }

  /**
   * Send a message and get response
   * 
   * @param {string} message - User message
   * @returns {Promise<Object>} Chat response
   */
  async sendMessage(message) {
    if (this.isWaitingForResponse) {
      throw new Error('Please wait for the current response to complete');
    }

    this.isWaitingForResponse = true;
    
    try {
      // Add user message to history
      this.messageHistory.push({
        role: 'user',
        content: message,
        timestamp: new Date()
      });

      let response;
      
      if (!this.client.currentConversation) {
        response = await this.client.startConversation(message);
      } else {
        response = await this.client.continueConversation(message);
      }

      return this._processChatResponse(response);
    } finally {
      this.isWaitingForResponse = false;
    }
  }

  /**
   * Get formatted chat history
   * 
   * @returns {Array} Chat message history
   */
  getHistory() {
    return this.messageHistory;
  }

  /**
   * Clear chat history and reset conversation
   */
  reset() {
    this.messageHistory = [];
    this.client.resetConversation();
    this.isWaitingForResponse = false;
  }

  /**
   * Process chat response for UI display
   * 
   * @private
   * @param {Object} response - API response
   * @returns {Object} Formatted chat response
   */
  _processChatResponse(response) {
    const baseResponse = {
      timestamp: new Date(),
      conversationId: this.client.currentConversation
    };

    switch (response.response_type) {
      case 'sql_result':
        return {
          ...baseResponse,
          type: 'sql_result',
          sql: response.sql,
          explanation: response.explanation,
          confidence: response.confidence,
          canExecute: true,
          results: response.execution_results
        };

      case 'clarification_request':
        return {
          ...baseResponse,
          type: 'clarification',
          message: response.clarification.message,
          suggestions: response.clarification.suggestions,
          canExecute: false
        };

      default:
        return {
          ...baseResponse,
          type: 'error',
          message: response.message || 'Unknown response type',
          canExecute: false
        };
    }
  }

  /**
   * Format assistant message for display
   * 
   * @private
   * @param {Object} result - Assistant response
   * @returns {string} Formatted message
   */
  _formatAssistantMessage(result) {
    switch (result.response_type) {
      case 'sql_result':
        return `I've generated this SQL query:\n\n\`\`\`sql\n${result.sql}\n\`\`\`\n\n${result.explanation}`;
      
      case 'clarification_request':
        return result.clarification.message;
      
      default:
        return result.message || 'I encountered an issue processing your request.';
    }
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

// For ES6 modules
export { TextToSQLClient, TextToSQLChat, TextToSQLError };

// For CommonJS
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { TextToSQLClient, TextToSQLChat, TextToSQLError };
}

// For browser global
if (typeof window !== 'undefined') {
  window.TextToSQLClient = TextToSQLClient;
  window.TextToSQLChat = TextToSQLChat;
  window.TextToSQLError = TextToSQLError;
}