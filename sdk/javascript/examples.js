/**
 * Text-to-SQL RAG API Integration Examples
 * 
 * This file contains practical examples for integrating the Text-to-SQL API
 * into various UI frameworks and chat applications.
 */

// =============================================================================
// BASIC USAGE EXAMPLES
// =============================================================================

/**
 * Example 1: Basic Client Setup
 */
async function basicSetup() {
  // Initialize client
  const client = new TextToSQLClient({
    baseUrl: 'http://localhost:8000',
    timeout: 30000,
    debug: true
  });

  // Check if API is available
  const isAvailable = await client.isAvailable();
  console.log('API Available:', isAvailable);

  // Get health status
  const health = await client.getHealth();
  console.log('Health Status:', health);
}

/**
 * Example 2: Simple SQL Generation
 */
async function simpleSQLGeneration() {
  const client = new TextToSQLClient({
    baseUrl: 'http://localhost:8000'
  });

  try {
    const result = await client.generateSQL(
      "Show me all users who registered in the last 30 days"
    );
    
    console.log('Generated SQL:', result.sql);
    console.log('Explanation:', result.explanation);
    console.log('Confidence:', result.confidence);
  } catch (error) {
    console.error('Error generating SQL:', error.message);
  }
}

/**
 * Example 3: Conversation-Based Interaction
 */
async function conversationExample() {
  const client = new TextToSQLClient({
    baseUrl: 'http://localhost:8000'
  });

  // Start conversation
  const response = await client.startConversation(
    "I want to see user activity data"
  );

  if (response.response_type === 'clarification_request') {
    console.log('Clarification needed:', response.clarification.message);
    console.log('Suggestions:', response.clarification.suggestions);
    
    // Continue with clarification
    const clarifiedResponse = await client.continueConversation(
      "Show me login activity for the past week"
    );
    
    console.log('Final SQL:', clarifiedResponse.sql);
  }
}

// =============================================================================
// REACT INTEGRATION EXAMPLES
// =============================================================================

/**
 * Example 4: React Hook for Text-to-SQL
 */
function useTextToSQL() {
  const [client] = useState(() => new TextToSQLClient({
    baseUrl: process.env.REACT_APP_API_URL || 'http://localhost:8000'
  }));
  
  const [conversation, setConversation] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const sendMessage = useCallback(async (message) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = conversation 
        ? await client.continueConversation(message)
        : await client.startConversation(message);

      if (!conversation) {
        setConversation(client.currentConversation);
      }

      setMessages(prev => [
        ...prev,
        { role: 'user', content: message, timestamp: new Date() },
        { role: 'assistant', content: response, timestamp: new Date() }
      ]);

      return response;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [client, conversation]);

  const resetConversation = useCallback(() => {
    client.resetConversation();
    setConversation(null);
    setMessages([]);
    setError(null);
  }, [client]);

  return {
    sendMessage,
    resetConversation,
    messages,
    isLoading,
    error,
    conversationId: conversation
  };
}

/**
 * Example 5: React Chat Component
 */
function TextToSQLChatComponent() {
  const { sendMessage, messages, isLoading, error, resetConversation } = useTextToSQL();
  const [input, setInput] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');

    try {
      await sendMessage(userMessage);
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  const renderMessage = (message) => {
    if (message.role === 'user') {
      return (
        <div className="user-message">
          <div className="message-content">{message.content}</div>
        </div>
      );
    }

    const response = message.content;
    
    if (response.response_type === 'sql_result') {
      return (
        <div className="assistant-message sql-result">
          <div className="explanation">{response.explanation}</div>
          <div className="confidence">Confidence: {(response.confidence * 100).toFixed(1)}%</div>
          <pre className="sql-code">{response.sql}</pre>
          {response.execution_results && (
            <div className="execution-results">
              <h4>Results ({response.execution_results.rows_returned} rows):</h4>
              <table>
                <thead>
                  <tr>
                    {response.execution_results.columns.map(col => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {response.execution_results.data.slice(0, 10).map((row, idx) => (
                    <tr key={idx}>
                      {response.execution_results.columns.map(col => (
                        <td key={col}>{row[col]}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      );
    }

    if (response.response_type === 'clarification_request') {
      return (
        <div className="assistant-message clarification">
          <div className="clarification-message">{response.clarification.message}</div>
          <div className="suggestions">
            {response.clarification.suggestions.map((suggestion, idx) => (
              <button 
                key={idx}
                onClick={() => sendMessage(suggestion)}
                className="suggestion-button"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      );
    }

    return (
      <div className="assistant-message error">
        Error: {response.message}
      </div>
    );
  };

  return (
    <div className="text-to-sql-chat">
      <div className="chat-header">
        <h3>Text-to-SQL Assistant</h3>
        <button onClick={resetConversation}>New Conversation</button>
      </div>
      
      <div className="chat-messages">
        {messages.map((message, idx) => (
          <div key={idx} className={`message ${message.role}`}>
            {renderMessage(message)}
          </div>
        ))}
        
        {isLoading && (
          <div className="message assistant loading">
            <div className="typing-indicator">Generating SQL...</div>
          </div>
        )}
        
        {error && (
          <div className="message error">
            <div className="error-message">Error: {error}</div>
          </div>
        )}
      </div>
      
      <form onSubmit={handleSubmit} className="chat-input">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask me anything about your data..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}

// =============================================================================
// VANILLA JAVASCRIPT EXAMPLES
// =============================================================================

/**
 * Example 6: Vanilla JavaScript Chat Interface
 */
class VanillaChatInterface {
  constructor(containerId, apiBaseUrl) {
    this.container = document.getElementById(containerId);
    this.client = new TextToSQLClient({ baseUrl: apiBaseUrl });
    this.chat = new TextToSQLChat(this.client);
    
    this.setupUI();
    this.setupEventListeners();
  }

  setupUI() {
    this.container.innerHTML = `
      <div class="chat-interface">
        <div class="chat-header">
          <h3>SQL Assistant</h3>
          <button id="reset-btn" class="reset-button">Reset</button>
        </div>
        <div id="messages" class="messages-container"></div>
        <div class="input-container">
          <input 
            id="message-input" 
            type="text" 
            placeholder="Ask me about your data..."
            autocomplete="off"
          />
          <button id="send-btn" class="send-button">Send</button>
        </div>
        <div id="status" class="status-bar"></div>
      </div>
    `;
  }

  setupEventListeners() {
    const input = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const resetBtn = document.getElementById('reset-btn');

    sendBtn.addEventListener('click', () => this.sendMessage());
    resetBtn.addEventListener('click', () => this.resetChat());
    
    input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    // Set up client event listeners
    this.client.on('sql-generated', (data) => {
      this.updateStatus(`SQL generated with ${(data.confidence * 100).toFixed(1)}% confidence`);
    });

    this.client.on('clarification-needed', (data) => {
      this.updateStatus('Clarification needed');
    });

    this.client.on('error', (error) => {
      this.updateStatus(`Error: ${error.message}`, 'error');
    });
  }

  async sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    
    if (!message) return;

    input.value = '';
    this.addMessage('user', message);
    this.updateStatus('Generating response...');

    try {
      const response = await this.chat.sendMessage(message);
      this.addAssistantMessage(response);
      this.updateStatus('Ready');
    } catch (error) {
      this.addMessage('error', `Error: ${error.message}`);
      this.updateStatus('Error occurred', 'error');
    }
  }

  addMessage(role, content) {
    const messagesContainer = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    if (typeof content === 'string') {
      messageDiv.textContent = content;
    } else {
      messageDiv.appendChild(content);
    }
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  addAssistantMessage(response) {
    const messageContent = document.createElement('div');
    
    switch (response.type) {
      case 'sql_result':
        messageContent.innerHTML = `
          <div class="sql-response">
            <div class="explanation">${response.explanation}</div>
            <div class="confidence">Confidence: ${(response.confidence * 100).toFixed(1)}%</div>
            <pre class="sql-code">${response.sql}</pre>
            ${this.renderExecutionResults(response.results)}
          </div>
        `;
        break;
        
      case 'clarification':
        messageContent.innerHTML = `
          <div class="clarification">
            <p>${response.message}</p>
            <div class="suggestions">
              ${response.suggestions.map(suggestion => 
                `<button class="suggestion" onclick="this.closest('.chat-interface').querySelector('#message-input').value='${suggestion}'">${suggestion}</button>`
              ).join('')}
            </div>
          </div>
        `;
        break;
        
      default:
        messageContent.textContent = response.message || 'Unknown response';
    }
    
    this.addMessage('assistant', messageContent);
  }

  renderExecutionResults(results) {
    if (!results || !results.data) return '';
    
    return `
      <div class="execution-results">
        <h4>Results (${results.rows_returned} rows):</h4>
        <table>
          <thead>
            <tr>${results.columns.map(col => `<th>${col}</th>`).join('')}</tr>
          </thead>
          <tbody>
            ${results.data.slice(0, 10).map(row => 
              `<tr>${results.columns.map(col => `<td>${row[col] || ''}</td>`).join('')}</tr>`
            ).join('')}
          </tbody>
        </table>
      </div>
    `;
  }

  updateStatus(message, type = 'info') {
    const statusBar = document.getElementById('status');
    statusBar.textContent = message;
    statusBar.className = `status-bar ${type}`;
  }

  resetChat() {
    this.chat.reset();
    document.getElementById('messages').innerHTML = '';
    this.updateStatus('Ready');
  }
}

// =============================================================================
// VUE.JS INTEGRATION EXAMPLE
// =============================================================================

/**
 * Example 7: Vue.js Composition API Integration
 */
const { ref, reactive, onMounted, computed } = Vue;

function useTextToSQLVue() {
  const client = ref(null);
  const messages = ref([]);
  const isLoading = ref(false);
  const error = ref(null);
  const conversationId = ref(null);

  onMounted(() => {
    client.value = new TextToSQLClient({
      baseUrl: process.env.VUE_APP_API_URL || 'http://localhost:8000'
    });
  });

  const sendMessage = async (message) => {
    if (!client.value) return;

    isLoading.value = true;
    error.value = null;

    try {
      const response = conversationId.value
        ? await client.value.continueConversation(message)
        : await client.value.startConversation(message);

      if (!conversationId.value) {
        conversationId.value = client.value.currentConversation;
      }

      messages.value.push(
        { role: 'user', content: message, timestamp: new Date() },
        { role: 'assistant', content: response, timestamp: new Date() }
      );

      return response;
    } catch (err) {
      error.value = err.message;
      throw err;
    } finally {
      isLoading.value = false;
    }
  };

  const resetConversation = () => {
    if (client.value) {
      client.value.resetConversation();
    }
    conversationId.value = null;
    messages.value = [];
    error.value = null;
  };

  return {
    sendMessage,
    resetConversation,
    messages: computed(() => messages.value),
    isLoading: computed(() => isLoading.value),
    error: computed(() => error.value),
    conversationId: computed(() => conversationId.value)
  };
}

// =============================================================================
// ADVANCED USAGE EXAMPLES
// =============================================================================

/**
 * Example 8: Document Upload and Context Enhancement
 */
async function documentUploadExample() {
  const client = new TextToSQLClient({
    baseUrl: 'http://localhost:8000'
  });

  // Upload a schema document
  const fileInput = document.getElementById('schema-file');
  const file = fileInput.files[0];

  if (file) {
    try {
      const uploadResult = await client.uploadDocument(file, {
        title: 'User Database Schema',
        document_type: 'schema',
        description: 'Main user table structure and relationships'
      });

      console.log('Document uploaded:', uploadResult);

      // Now queries will have better context
      const response = await client.generateSQL(
        "Show me user registration trends"
      );

      console.log('SQL with enhanced context:', response.sql);
    } catch (error) {
      console.error('Upload failed:', error);
    }
  }
}

/**
 * Example 9: LLM Provider Management
 */
async function providerManagementExample() {
  const client = new TextToSQLClient({
    baseUrl: 'http://localhost:8000'
  });

  // Get current provider info
  const providerInfo = await client.getLLMProviderInfo();
  console.log('Current provider:', providerInfo.provider_info.provider);

  // Switch to custom provider
  try {
    await client.switchLLMProvider('custom');
    console.log('Switched to custom provider');

    // Test the new provider
    const testResult = await client.testLLMProvider();
    console.log('Provider test:', testResult.success);
  } catch (error) {
    console.error('Provider switch failed:', error);
  }
}

/**
 * Example 10: Error Handling and Retry Logic
 */
class RobustTextToSQLClient {
  constructor(config) {
    this.client = new TextToSQLClient(config);
    this.maxRetries = config.maxRetries || 3;
    this.retryDelay = config.retryDelay || 1000;
  }

  async sendMessageWithRetry(message, retries = 0) {
    try {
      return await this.client.startConversation(message);
    } catch (error) {
      if (retries < this.maxRetries && this.isRetryableError(error)) {
        console.log(`Retry attempt ${retries + 1}/${this.maxRetries}`);
        await this.delay(this.retryDelay * Math.pow(2, retries)); // Exponential backoff
        return this.sendMessageWithRetry(message, retries + 1);
      }
      throw error;
    }
  }

  isRetryableError(error) {
    return error.status === 500 || 
           error.status === 502 || 
           error.status === 503 || 
           error.status === 504 ||
           error.code === 'TIMEOUT' ||
           error.code === 'NETWORK_ERROR';
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Example 11: Real-time Collaboration Features
 */
class CollaborativeTextToSQL {
  constructor(config) {
    this.client = new TextToSQLClient(config);
    this.collaborators = new Map();
    this.sharedConversations = new Map();
  }

  async shareConversation(conversationId, userId) {
    const conversation = await this.client.getConversation(conversationId);
    this.sharedConversations.set(conversationId, {
      ...conversation,
      shared_with: [userId],
      owner: this.client.getSession()
    });

    // Notify collaborators
    this.notifyCollaborators(conversationId, 'conversation_shared');
  }

  async joinSharedConversation(conversationId, userId) {
    const shared = this.sharedConversations.get(conversationId);
    if (shared && shared.shared_with.includes(userId)) {
      this.client.currentConversation = conversationId;
      return shared;
    }
    throw new Error('Access denied to conversation');
  }

  notifyCollaborators(conversationId, event) {
    // Implementation would depend on your real-time system (WebSocket, etc.)
    console.log(`Notifying collaborators about ${event} for ${conversationId}`);
  }
}

// =============================================================================
// INITIALIZATION HELPER
// =============================================================================

/**
 * Initialize Text-to-SQL client with environment detection
 */
function initializeTextToSQL(customConfig = {}) {
  const defaultConfig = {
    baseUrl: 'http://localhost:8000',
    timeout: 30000,
    debug: false
  };

  // Detect environment
  if (typeof process !== 'undefined' && process.env) {
    // Node.js environment
    defaultConfig.baseUrl = process.env.TEXT_TO_SQL_API_URL || defaultConfig.baseUrl;
    defaultConfig.debug = process.env.NODE_ENV === 'development';
  } else if (typeof window !== 'undefined') {
    // Browser environment
    defaultConfig.baseUrl = window.TEXT_TO_SQL_API_URL || defaultConfig.baseUrl;
    defaultConfig.debug = window.location.hostname === 'localhost';
  }

  return new TextToSQLClient({ ...defaultConfig, ...customConfig });
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    basicSetup,
    simpleSQLGeneration,
    conversationExample,
    useTextToSQL,
    VanillaChatInterface,
    RobustTextToSQLClient,
    CollaborativeTextToSQL,
    initializeTextToSQL
  };
}