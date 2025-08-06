# Connection Test Suite

This directory contains comprehensive connection tests for the text-to-SQL RAG application. These tests validate connectivity to all external services required for the application to function properly.

## Test Scripts

### Individual Test Scripts

1. **`test_mongodb_connection.py`** - Tests MongoDB connectivity and operations
   - Connection validation
   - Database access permissions
   - CRUD operations
   - Index management
   - MongoDB service wrapper testing

2. **`test_opensearch_connection.py`** - Tests OpenSearch/Elasticsearch connectivity
   - Cluster connection and health
   - Index creation and management
   - Document indexing and search
   - Vector/KNN search capabilities
   - SSL/authentication validation

3. **`test_bedrock_llm.py`** - Tests Bedrock endpoint LLM connectivity
   - Bedrock endpoint configuration validation
   - Endpoint connectivity testing
   - Text generation with various prompts
   - Custom endpoint service testing
   - LLM factory service integration

4. **`test_bedrock_embedding.py`** - Tests Bedrock endpoint embedding model connectivity
   - Endpoint embedding generation with various text inputs
   - Vector similarity calculations
   - Dimension consistency validation
   - Custom endpoint service testing

### Master Test Runner

**`run_all_connection_tests.py`** - Comprehensive test suite runner
- Runs all connection tests automatically
- Provides detailed results and configuration guidance
- Supports selective test execution
- Saves results to JSON for analysis

## Usage

### Running Individual Tests

```bash
# Run from project root directory
cd /path/to/llamaindex_proj

# Test MongoDB connection
python tests/test_mongodb_connection.py

# Test OpenSearch connection  
python tests/test_opensearch_connection.py

# Test Bedrock LLM
python tests/test_bedrock_llm.py

# Test Bedrock Embeddings
python tests/test_bedrock_embedding.py
```

### Running All Tests

```bash
# Run all tests
python tests/run_all_connection_tests.py

# Run specific tests only
python tests/run_all_connection_tests.py --tests mongodb opensearch

# Quick mode (minimal output)
python tests/run_all_connection_tests.py --quick

# Save results to JSON file
python tests/run_all_connection_tests.py --save-results
```

## Configuration Requirements

### MongoDB
```bash
export MONGODB_URL="mongodb://localhost:27017"
export MONGODB_DATABASE="text_to_sql_rag"
```

### OpenSearch
```bash
export OPENSEARCH_HOST="localhost"
export OPENSEARCH_PORT="9200"
export OPENSEARCH_USE_SSL="false"
export OPENSEARCH_USERNAME="admin"
export OPENSEARCH_PASSWORD="admin"
export OPENSEARCH_VECTOR_SIZE="1024"
```

### Bedrock Endpoint
```bash
export BEDROCK_ENDPOINT_URL="https://your-endpoint.com/invokeBedrock/"
export AWS_LLM_MODEL="anthropic.claude-3-5-sonnet-20241022-v2:0"
export AWS_EMBEDDING_MODEL="amazon.titan-embed-text-v2:0"
export LLM_PROVIDER="bedrock"
```

## Test Output Examples

### Successful Test Run
```
🔍 Starting MongoDB Connection Tests
==================================================
✅ PASS: Connection Parameters - Configuration loaded successfully
✅ PASS: Basic Connection - Connected successfully
✅ PASS: Database Access - Database accessible
✅ PASS: Collection Operations - CRUD operations successful
✅ PASS: Index Operations - Index operations successful
✅ PASS: MongoDB Service - MongoDB service working correctly
✅ PASS: Cleanup - Test cleanup completed successfully

📊 Test Results: 6/6 tests passed
🎉 All MongoDB tests passed!
```

### Failed Test with Guidance
```
❌ FAIL: Basic Connection - Connection failed: [Errno 111] Connection refused

🔧 MongoDB Configuration:
   Environment Variables:
     MONGODB_URL=mongodb://localhost:27017
     MONGODB_DATABASE=text_to_sql_rag
   Notes:
     • Ensure MongoDB is running and accessible
     • Check network connectivity to MongoDB host
     • Verify authentication credentials if required
```

## Dependencies

The test scripts require the following packages:
- `pymongo` - MongoDB connectivity
- `opensearch-py` - OpenSearch connectivity  
- `requests` - HTTP requests (for Bedrock endpoint connectivity)
- `numpy` - Vector operations

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root directory
2. **Connection Refused**: Check if services are running and accessible
3. **Authentication Errors**: Verify credentials and permissions
4. **Timeout Errors**: Check network connectivity and increase timeouts

### Debug Mode

Most test scripts provide detailed error information when tests fail. Check the error messages and details for specific configuration guidance.

### Log Files

Test results can be saved to JSON files using the `--save-results` flag with the master test runner for detailed analysis and debugging.

## Integration with CI/CD

These test scripts can be integrated into CI/CD pipelines:

```bash
# Example CI/CD usage
python tests/run_all_connection_tests.py --quick --save-results
exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "All connection tests passed"
else
    echo "Connection tests failed - check configuration"
    exit 1
fi
```