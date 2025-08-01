# Bedrock Endpoint Integration Guide

## Overview

The system now supports accessing AWS Bedrock through HTTP endpoints instead of requiring local AWS credentials. This eliminates the need for AWS profile configuration and credential management.

## Key Benefits

✅ **No AWS Credentials Required**: No need to configure AWS profiles or manage credentials locally  
✅ **Simplified Deployment**: Easier to deploy in environments without AWS CLI  
✅ **Centralized Access**: All Bedrock access goes through your controlled endpoint  
✅ **Consistent Interface**: Same LLM interface as before, just different transport mechanism  

## Configuration

### Environment Variables

Set these variables in your `.env` file:

```env
# Use bedrock_endpoint provider
LLM_PROVIDER=bedrock_endpoint

# Your Bedrock endpoint URL
BEDROCK_ENDPOINT_URL=https://your-base-host.com

# Model ID (same as before)
AWS_LLM_MODEL=us.anthropic.claude-3-haiku-20240307-v1:0
```

### Supported Models

The endpoint should support these Claude models:
- `us.anthropic.claude-3-haiku-20240307-v1:0` (Fast, cost-effective)
- `us.anthropic.claude-3-sonnet-20240229-v1:0` (Balanced performance)
- `us.anthropic.claude-3-opus-20240229-v1:0` (Highest capability)

## Endpoint API Contract

### Request Format

Your endpoint should expect POST requests to `/invokeBedrock/` with this format:

```json
{
  "model_id": "us.anthropic.claude-3-haiku-20240307-v1:0",
  "query": "Compare between Chat GPT models and Claude models?",
  "model_kwargs": {
    "max_tokens": 2048,
    "temperature": 0.1,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"]
  }
}
```

### Response Format

The endpoint should return JSON in one of these formats:

**Option 1 - Simple text response:**
```json
{
  "text": "Generated response text here..."
}
```

**Option 2 - Structured response:**
```json
{
  "content": "Generated response text here...",
  "model_id": "us.anthropic.claude-3-haiku-20240307-v1:0",
  "usage": {
    "input_tokens": 150,
    "output_tokens": 200
  }
}
```

The system will automatically extract text from common response fields: `text`, `content`, `response`, `output`, or `completion`.

## Integration Details

### New Service Classes

1. **`BedrockEndpointService`**: Core HTTP client for Bedrock endpoint
2. **`BedrockEndpointLLMWrapper`**: LLM interface wrapper for compatibility
3. **`BedrockEndpointEmbeddingService`**: Embedding service with Bedrock fallback

### Updated Components

1. **`LLMProviderFactory`**: Added `bedrock_endpoint` provider support
2. **`Settings`**: Added `BEDROCK_ENDPOINT_URL` configuration
3. **`HierarchicalContextService`**: Works seamlessly with endpoint service
4. **`LlamaIndexVectorService`**: Updated custom wrappers for endpoint compatibility
5. **`CustomBedrockEmbedding`**: Enhanced to support both inference ARNs and endpoints
6. **`CustomBedrockLLM`**: Enhanced to support both inference ARNs and endpoints

### LlamaIndex Integration Features

✅ **Inference Profile ARN Support**: Maintains existing support for inference profile ARNs  
✅ **Endpoint Service Support**: New support for HTTP-based Bedrock access  
✅ **Automatic Fallback**: Embeddings automatically fall back to Bedrock if endpoint doesn't support them  
✅ **Seamless Switching**: Change `LLM_PROVIDER` to switch between modes  
✅ **Custom Wrapper Compatibility**: Works with existing LlamaIndex custom wrappers

## Usage Examples

### Basic Text Generation

```python
from src.text_to_sql_rag.services.llm_provider_factory import llm_factory

# Generate text using endpoint
response = llm_factory.generate_text(
    prompt="Explain hierarchical metadata architecture",
    max_tokens=1000,
    temperature=0.1
)
print(response)
```

### SQL Generation

```python
# Generate SQL using the text-to-SQL system
from src.text_to_sql_rag.core.langgraph_agent import TextToSQLAgent

agent = TextToSQLAgent(vector_service, query_execution_service)
result = await agent.generate_sql("Show me all active trades from last month")
print(result["sql"])
```

### Direct Endpoint Access

```python
from src.text_to_sql_rag.services.bedrock_endpoint_service import BedrockEndpointService

# Direct endpoint usage
bedrock = BedrockEndpointService("https://your-endpoint.com")
response = bedrock.invoke_model(
    model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
    query="What is hierarchical metadata?",
    max_tokens=500
)
```

## Migration from AWS Credentials

### Old Configuration (.env)
```env
LLM_PROVIDER=bedrock
AWS_REGION=us-east-1
AWS_USE_PROFILE=true
AWS_PROFILE=your-profile
AWS_LLM_MODEL=us.anthropic.claude-3-haiku-20240307-v1:0
AWS_EMBEDDING_MODEL=amazon.titan-embed-text-v1
```

### New Configuration (.env)
```env
LLM_PROVIDER=bedrock_endpoint
BEDROCK_ENDPOINT_URL=https://your-endpoint.com
AWS_LLM_MODEL=us.anthropic.claude-3-haiku-20240307-v1:0
AWS_EMBEDDING_MODEL=amazon.titan-embed-text-v1
```

**Important Notes:**
- Keep `AWS_LLM_MODEL` and `AWS_EMBEDDING_MODEL` - they're still needed for model IDs
- Embeddings automatically fall back to Bedrock if endpoint doesn't support them
- All existing code continues to work - LlamaIndex integration is fully compatible
- Inference profile ARNs continue to work with both modes

## Error Handling

The endpoint service includes comprehensive error handling:

### Connection Errors
```python
# Automatic retry and fallback handling
try:
    response = bedrock_service.generate_text(prompt)
except requests.exceptions.RequestException as e:
    logger.error(f"Endpoint connection failed: {e}")
    # Implement fallback logic
```

### Response Parsing
```python
# Flexible response format handling
if "text" in response:
    return response["text"]
elif "content" in response:
    return response["content"]
# ... handles multiple response formats
```

## Performance Considerations

### Timeouts
- Default request timeout: 60 seconds
- Configurable per request
- Suitable for long-running LLM requests

### Connection Pooling
- Uses `requests.Session` for connection reuse
- Persistent connections to endpoint
- Better performance for multiple requests

### Caching
- No automatic caching (implement at endpoint level if needed)
- Session-based connection reuse
- Compatible with existing caching layers

## Troubleshooting

### Common Issues

**1. "Bedrock endpoint URL not configured"**
```bash
# Solution: Set the endpoint URL
export BEDROCK_ENDPOINT_URL=https://your-endpoint.com
```

**2. "Connection refused"**
```bash
# Check endpoint is accessible
curl -X POST https://your-endpoint.com/invokeBedrock/ \
  -H "Content-Type: application/json" \
  -d '{"model_id":"us.anthropic.claude-3-haiku-20240307-v1:0","query":"test"}'
```

**3. "Unexpected response format"**
- Check endpoint response matches expected JSON format
- Look for `text`, `content`, or `response` fields in JSON
- Enable debug logging to see raw responses

### Debug Logging

Enable detailed logging:
```python
import logging
logging.getLogger("text_to_sql_rag.services.bedrock_endpoint_service").setLevel(logging.DEBUG)
```

## Security Considerations

### HTTPS Required
- Always use HTTPS endpoints in production
- Validate SSL certificates
- Consider client certificate authentication

### Authentication
- Endpoint should handle authentication
- No credentials stored in application
- Consider API key headers if needed

### Network Security
- Restrict endpoint access to authorized sources
- Use VPN or private networks where possible
- Monitor endpoint access logs

## Testing

### Health Check
```python
from src.text_to_sql_rag.services.llm_provider_factory import llm_factory

# Test endpoint connectivity
is_healthy = llm_factory.health_check()
print(f"Endpoint health: {is_healthy}")
```

### Provider Info
```python
# Get current configuration
info = llm_factory.get_provider_info()
print(f"Provider: {info['provider']}")
print(f"Endpoint: {info['endpoint_url']}")
print(f"Model: {info['model']}")
```

## Complete Example

Here's a complete example of setting up and using the endpoint integration:

```python
# 1. Set environment variables
import os
os.environ['LLM_PROVIDER'] = 'bedrock_endpoint'
os.environ['BEDROCK_ENDPOINT_URL'] = 'https://your-endpoint.com'
os.environ['AWS_LLM_MODEL'] = 'us.anthropic.claude-3-haiku-20240307-v1:0'

# 2. Initialize services (automatic from settings)
from src.text_to_sql_rag.services.llm_provider_factory import llm_factory
from src.text_to_sql_rag.services.hierarchical_context_service import HierarchicalContextService

# 3. Test basic functionality
response = llm_factory.generate_text("Hello, world!")
print(f"Response: {response}")

# 4. Use with hierarchical context (same as before)
# The hierarchical context service automatically uses the endpoint
```

The endpoint integration provides a seamless way to use Bedrock without AWS credential complexity while maintaining full compatibility with the existing hierarchical metadata architecture.