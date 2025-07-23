# React Integration Guide for Text-to-SQL RAG API

## AI Agent Instructions

This guide provides structured patterns and code templates for integrating the Text-to-SQL RAG API into React applications. Use these patterns to generate production-ready React components.

## Quick Start Template

### 1. API Client Setup

```javascript
// src/services/textToSQLAPI.js
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_TEXT_TO_SQL_API_URL || 'http://localhost:8000';

// Create axios instance with default config
export const textToSQLAPI = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Generate session ID for user tracking
const SESSION_ID = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

// Add session ID to all requests
textToSQLAPI.interceptors.request.use((config) => {
  config.headers['X-Session-ID'] = SESSION_ID;
  return config;
});

// Error handling interceptor
textToSQLAPI.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 422) {
      // This is likely a clarification request, not an error
      return Promise.resolve(error.response);
    }
    
    // Log error for debugging
    console.error('API Error:', error.response?.data || error.message);
    
    return Promise.reject(error);
  }
);

export default textToSQLAPI;
```

### 2. Custom Hooks

```javascript
// src/hooks/useTextToSQL.js
import { useState, useCallback } from 'react';
import textToSQLAPI from '../services/textToSQLAPI';

export const useTextToSQL = () => {
  const [conversationId, setConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const sendMessage = useCallback(async (query) => {
    setIsLoading(true);
    setError(null);

    try {
      let response;
      
      if (!conversationId) {
        // Start new conversation
        response = await textToSQLAPI.post('/conversations/start', {
          query,
          context: {
            user_id: `user_${Date.now()}`,
            session_context: 'chat_interface'
          }
        });
        setConversationId(response.data.conversation_id);
      } else {
        // Continue existing conversation
        response = await textToSQLAPI.post(`/conversations/${conversationId}/continue`, {
          message: query
        });
      }

      // Add messages to history
      const userMessage = {
        id: Date.now(),
        role: 'user',
        content: query,
        timestamp: new Date()
      };

      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: response.data.result || response.data,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, userMessage, assistantMessage]);
      
      return response.data.result || response.data;
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message || 'An error occurred';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [conversationId]);

  const resetConversation = useCallback(() => {
    setConversationId(null);
    setMessages([]);
    setError(null);
  }, []);

  const generateSQL = useCallback(async (query) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await textToSQLAPI.post('/query/generate', { query });
      return response.data;
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message;
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    // State
    conversationId,
    messages,
    isLoading,
    error,
    
    // Actions
    sendMessage,
    resetConversation,
    generateSQL,
  };
};

// Health monitoring hook
export const useAPIHealth = () => {
  const [health, setHealth] = useState(null);
  const [isChecking, setIsChecking] = useState(false);

  const checkHealth = useCallback(async () => {
    setIsChecking(true);
    try {
      const response = await textToSQLAPI.get('/health');
      setHealth(response.data);
      return response.data;
    } catch (error) {
      setHealth({ 
        status: 'error', 
        error: error.message,
        timestamp: new Date().toISOString()
      });
      return null;
    } finally {
      setIsChecking(false);
    }
  }, []);

  return { health, isChecking, checkHealth };
};
```

### 3. Core Components

```javascript
// src/components/TextToSQLChat.jsx
import React, { useState, useRef, useEffect } from 'react';
import { useTextToSQL } from '../hooks/useTextToSQL';
import './TextToSQLChat.css';

const TextToSQLChat = ({ onSQLGenerated, className = '' }) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  
  const {
    messages,
    isLoading,
    error,
    sendMessage,
    resetConversation,
    conversationId
  } = useTextToSQL();

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!input.trim() || isLoading) return;

    const userInput = input.trim();
    setInput('');

    try {
      const result = await sendMessage(userInput);
      
      // Notify parent component if SQL was generated
      if (result.response_type === 'sql_result' && onSQLGenerated) {
        onSQLGenerated(result);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setInput(suggestion);
    inputRef.current?.focus();
  };

  return (
    <div className={`text-to-sql-chat ${className}`}>
      {/* Header */}
      <div className="chat-header">
        <h3>SQL Assistant</h3>
        <div className="chat-controls">
          {conversationId && (
            <button 
              onClick={resetConversation}
              className="btn btn-secondary btn-sm"
              title="Start new conversation"
            >
              New Chat
            </button>
          )}
        </div>
      </div>

      {/* Messages Container */}
      <div className="messages-container">
        {messages.length === 0 && (
          <div className="welcome-message">
            <h4>👋 Welcome to SQL Assistant</h4>
            <p>Ask me anything about your data in plain English!</p>
            <div className="example-queries">
              <p>Try asking:</p>
              <button 
                className="example-query"
                onClick={() => handleSuggestionClick("How many users registered last month?")}
              >
                "How many users registered last month?"
              </button>
              <button 
                className="example-query"
                onClick={() => handleSuggestionClick("Show me top selling products")}
              >
                "Show me top selling products"
              </button>
            </div>
          </div>
        )}

        {messages.map((message) => (
          <MessageComponent 
            key={message.id} 
            message={message}
            onSuggestionClick={handleSuggestionClick}
          />
        ))}

        {isLoading && (
          <div className="message assistant">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
              <span>Generating SQL...</span>
            </div>
          </div>
        )}

        {error && (
          <div className="message error">
            <div className="message-content">
              <strong>Error:</strong> {error}
              <button 
                onClick={() => window.location.reload()}
                className="btn btn-sm btn-outline-primary mt-2"
              >
                Retry
              </button>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="chat-input-form">
        <div className="input-group">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask me about your data..."
            disabled={isLoading}
            className="form-control"
            maxLength={1000}
          />
          <button 
            type="submit" 
            disabled={isLoading || !input.trim()}
            className="btn btn-primary"
          >
            {isLoading ? (
              <span className="spinner-border spinner-border-sm" />
            ) : (
              'Send'
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

// Message Component
const MessageComponent = ({ message, onSuggestionClick }) => {
  const { role, content } = message;

  if (role === 'user') {
    return (
      <div className="message user">
        <div className="message-content">
          {content}
        </div>
      </div>
    );
  }

  // Assistant message
  const result = content;

  return (
    <div className="message assistant">
      <div className="message-content">
        {result.response_type === 'sql_result' && (
          <SQLResultDisplay result={result} />
        )}

        {result.response_type === 'clarification_request' && (
          <ClarificationDisplay 
            clarification={result.clarification}
            onSuggestionClick={onSuggestionClick}
          />
        )}

        {result.response_type === 'error' && (
          <div className="error-display">
            <strong>Error:</strong> {result.message}
          </div>
        )}
      </div>
    </div>
  );
};

// SQL Result Display Component
const SQLResultDisplay = ({ result }) => {
  const [showResults, setShowResults] = useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(result.sql);
    // You could add a toast notification here
  };

  return (
    <div className="sql-result-display">
      {/* Confidence and Explanation */}
      <div className="result-header">
        <div className="confidence-badge">
          Confidence: {(result.confidence * 100).toFixed(1)}%
        </div>
      </div>

      <div className="explanation">
        {result.explanation}
      </div>

      {/* SQL Code */}
      <div className="sql-code-container">
        <div className="sql-code-header">
          <span>Generated SQL</span>
          <button 
            onClick={copyToClipboard}
            className="btn btn-sm btn-outline-secondary"
            title="Copy to clipboard"
          >
            📋 Copy
          </button>
        </div>
        <pre className="sql-code">
          <code>{result.sql}</code>
        </pre>
      </div>

      {/* Execution Results */}
      {result.execution_results && (
        <div className="execution-results">
          <button 
            onClick={() => setShowResults(!showResults)}
            className="btn btn-sm btn-outline-primary"
          >
            {showResults ? 'Hide' : 'Show'} Results ({result.execution_results.rows_returned} rows)
          </button>
          
          {showResults && (
            <ResultsTable results={result.execution_results} />
          )}
        </div>
      )}
    </div>
  );
};

// Clarification Display Component
const ClarificationDisplay = ({ clarification, onSuggestionClick }) => {
  return (
    <div className="clarification-display">
      <div className="clarification-message">
        {clarification.message}
      </div>
      
      {clarification.suggestions && clarification.suggestions.length > 0 && (
        <div className="suggestions">
          <p>Try one of these:</p>
          {clarification.suggestions.map((suggestion, index) => (
            <button
              key={index}
              onClick={() => onSuggestionClick(suggestion)}
              className="suggestion-btn"
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

// Results Table Component
const ResultsTable = ({ results }) => {
  if (!results.data || results.data.length === 0) {
    return <div className="no-results">No data returned</div>;
  }

  const displayData = results.data.slice(0, 100); // Limit to 100 rows for performance

  return (
    <div className="results-table-container">
      <table className="results-table">
        <thead>
          <tr>
            {results.columns.map(column => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {displayData.map((row, index) => (
            <tr key={index}>
              {results.columns.map(column => (
                <td key={column}>
                  {row[column] !== null && row[column] !== undefined 
                    ? String(row[column]) 
                    : '-'
                  }
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      
      {results.data.length > 100 && (
        <div className="table-footer">
          Showing first 100 of {results.rows_returned} rows
        </div>
      )}
    </div>
  );
};

export default TextToSQLChat;
```

### 4. CSS Styles

```css
/* src/components/TextToSQLChat.css */
.text-to-sql-chat {
  display: flex;
  flex-direction: column;
  height: 600px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  background: white;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  border-bottom: 1px solid #e0e0e0;
  background: #f8f9fa;
}

.chat-header h3 {
  margin: 0;
  color: #2c3e50;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.message {
  display: flex;
  margin-bottom: 1rem;
}

.message.user {
  justify-content: flex-end;
}

.message.assistant {
  justify-content: flex-start;
}

.message-content {
  max-width: 80%;
  padding: 0.75rem 1rem;
  border-radius: 1rem;
  word-wrap: break-word;
}

.message.user .message-content {
  background: #007bff;
  color: white;
  border-bottom-right-radius: 0.25rem;
}

.message.assistant .message-content {
  background: #f1f3f5;
  color: #333;
  border-bottom-left-radius: 0.25rem;
}

.message.error .message-content {
  background: #f8d7da;
  color: #721c24;
  border: 1px solid #f5c6cb;
}

/* Welcome Message */
.welcome-message {
  text-align: center;
  padding: 2rem;
  color: #6c757d;
}

.example-queries {
  margin-top: 1rem;
}

.example-query {
  display: block;
  margin: 0.5rem auto;
  padding: 0.5rem 1rem;
  background: #e9ecef;
  border: none;
  border-radius: 0.5rem;
  cursor: pointer;
  transition: background-color 0.2s;
}

.example-query:hover {
  background: #dee2e6;
}

/* SQL Result Display */
.sql-result-display {
  margin: 0.5rem 0;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.confidence-badge {
  background: #28a745;
  color: white;
  padding: 0.25rem 0.5rem;
  border-radius: 0.25rem;
  font-size: 0.875rem;
  font-weight: 500;
}

.explanation {
  margin-bottom: 1rem;
  font-style: italic;
  color: #6c757d;
}

.sql-code-container {
  border: 1px solid #e0e0e0;
  border-radius: 0.5rem;
  overflow: hidden;
  margin-bottom: 1rem;
}

.sql-code-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem;
  background: #f8f9fa;
  border-bottom: 1px solid #e0e0e0;
  font-weight: 500;
}

.sql-code {
  margin: 0;
  padding: 1rem;
  background: #f8f9fa;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 0.875rem;
  line-height: 1.4;
  overflow-x: auto;
}

/* Clarification Display */
.clarification-display {
  margin: 0.5rem 0;
}

.clarification-message {
  margin-bottom: 1rem;
  font-weight: 500;
}

.suggestions {
  margin-top: 0.5rem;
}

.suggestion-btn {
  display: inline-block;
  margin: 0.25rem;
  padding: 0.5rem 0.75rem;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 0.25rem;
  cursor: pointer;
  transition: background-color 0.2s;
}

.suggestion-btn:hover {
  background: #0056b3;
}

/* Results Table */
.results-table-container {
  margin-top: 1rem;
  border: 1px solid #e0e0e0;
  border-radius: 0.5rem;
  overflow: hidden;
}

.results-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
}

.results-table th,
.results-table td {
  padding: 0.5rem;
  text-align: left;
  border-bottom: 1px solid #e0e0e0;
}

.results-table th {
  background: #f8f9fa;
  font-weight: 600;
  position: sticky;
  top: 0;
}

.results-table tbody tr:hover {
  background: #f8f9fa;
}

.table-footer {
  padding: 0.5rem;
  background: #f8f9fa;
  text-align: center;
  font-size: 0.875rem;
  color: #6c757d;
}

/* Input Form */
.chat-input-form {
  padding: 1rem;
  border-top: 1px solid #e0e0e0;
  background: white;
}

.input-group {
  display: flex;
  gap: 0.5rem;
}

.form-control {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #ced4da;
  border-radius: 0.5rem;
  font-size: 1rem;
  outline: none;
}

.form-control:focus {
  border-color: #007bff;
  box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
}

.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 0.5rem;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.2s;
  text-decoration: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.btn-primary {
  background: #007bff;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: #0056b3;
}

.btn-secondary {
  background: #6c757d;
  color: white;
}

.btn-outline-primary {
  background: transparent;
  color: #007bff;
  border: 1px solid #007bff;
}

.btn-outline-secondary {
  background: transparent;
  color: #6c757d;
  border: 1px solid #6c757d;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-sm {
  padding: 0.375rem 0.75rem;
  font-size: 0.875rem;
}

/* Typing indicator */
.typing-indicator {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  margin-right: 0.5rem;
}

.typing-indicator span {
  width: 0.5rem;
  height: 0.5rem;
  background: #6c757d;
  border-radius: 50%;
  animation: typing 1.4s infinite;
}

.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0%, 60%, 100% {
    transform: translateY(0);
  }
  30% {
    transform: translateY(-0.5rem);
  }
}

/* Spinner */
.spinner-border {
  width: 1rem;
  height: 1rem;
  border: 0.125rem solid currentColor;
  border-right-color: transparent;
  border-radius: 50%;
  animation: spinner 0.75s linear infinite;
}

.spinner-border-sm {
  width: 0.875rem;
  height: 0.875rem;
  border-width: 0.125rem;
}

@keyframes spinner {
  to {
    transform: rotate(360deg);
  }
}

/* Responsive */
@media (max-width: 768px) {
  .text-to-sql-chat {
    height: 500px;
  }
  
  .message-content {
    max-width: 90%;
  }
  
  .sql-code {
    font-size: 0.75rem;
  }
  
  .results-table {
    font-size: 0.75rem;
  }
  
  .results-table th,
  .results-table td {
    padding: 0.25rem;
  }
}
```

## Usage in Your React App

### 1. Basic Integration

```javascript
// src/App.js
import React from 'react';
import TextToSQLChat from './components/TextToSQLChat';
import './App.css';

function App() {
  const handleSQLGenerated = (result) => {
    console.log('SQL Generated:', result);
    // You can do something with the generated SQL here
    // e.g., save to state, execute automatically, etc.
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>My Data Analytics Dashboard</h1>
      </header>
      
      <main className="App-main">
        <TextToSQLChat 
          onSQLGenerated={handleSQLGenerated}
          className="main-chat"
        />
      </main>
    </div>
  );
}

export default App;
```

### 2. Advanced Integration with Context

```javascript
// src/context/TextToSQLContext.js
import React, { createContext, useContext } from 'react';
import { useTextToSQL, useAPIHealth } from '../hooks/useTextToSQL';

const TextToSQLContext = createContext();

export const TextToSQLProvider = ({ children }) => {
  const textToSQL = useTextToSQL();
  const apiHealth = useAPIHealth();

  const value = {
    ...textToSQL,
    ...apiHealth,
  };

  return (
    <TextToSQLContext.Provider value={value}>
      {children}
    </TextToSQLContext.Provider>
  );
};

export const useTextToSQLContext = () => {
  const context = useContext(TextToSQLContext);
  if (!context) {
    throw new Error('useTextToSQLContext must be used within TextToSQLProvider');
  }
  return context;
};

// Usage in App.js
import { TextToSQLProvider } from './context/TextToSQLContext';

function App() {
  return (
    <TextToSQLProvider>
      <YourAppComponents />
    </TextToSQLProvider>
  );
}
```

### 3. Environment Configuration

```javascript
// .env
REACT_APP_TEXT_TO_SQL_API_URL=http://localhost:8000

// .env.production
REACT_APP_TEXT_TO_SQL_API_URL=https://your-api-domain.com
```

## AI Agent Code Generation Templates

When generating React components for this API, use these patterns:

### Pattern 1: Simple SQL Generator
```javascript
const SimpleSQLGenerator = () => {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const generateSQL = async () => {
    setLoading(true);
    try {
      const response = await textToSQLAPI.post('/query/generate', { query });
      setResult(response.data);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input 
        value={query} 
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Enter your question..."
      />
      <button onClick={generateSQL} disabled={loading}>
        {loading ? 'Generating...' : 'Generate SQL'}
      </button>
      {result && <SQLDisplay result={result} />}
    </div>
  );
};
```

### Pattern 2: Document Upload Component
```javascript
const DocumentUploader = ({ onUploadComplete }) => {
  const [uploading, setUploading] = useState(false);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploading(true);
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', file.name);
    formData.append('document_type', file.name.includes('schema') ? 'schema' : 'report');

    try {
      const response = await textToSQLAPI.post('/documents/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      onUploadComplete?.(response.data);
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <input 
        type="file" 
        accept=".json,.txt,.md,.sql"
        onChange={handleFileUpload}
        disabled={uploading}
      />
      {uploading && <span>Uploading...</span>}
    </div>
  );
};
```

### Pattern 3: Health Status Component
```javascript
const APIHealthStatus = () => {
  const { health, checkHealth } = useAPIHealth();

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30s
    return () => clearInterval(interval);
  }, [checkHealth]);

  if (!health) return <div>Checking API status...</div>;

  return (
    <div className={`health-status ${health.status}`}>
      <span>API Status: {health.status}</span>
      {health.status !== 'healthy' && (
        <button onClick={checkHealth}>Retry</button>
      )}
    </div>
  );
};
```

This guide provides everything needed for AI agents to generate production-ready React components that integrate seamlessly with the Text-to-SQL RAG API.