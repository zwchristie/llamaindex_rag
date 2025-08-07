import json
import boto3
import logging
import time
from typing import Dict, Any, Optional, Union

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize Bedrock client with retry configuration
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    config=boto3.session.Config(
        retries={
            'max_attempts': 3,
            'mode': 'adaptive'
        }
    )
)

class BedrockError(Exception):
    """Custom exception for Bedrock-related errors"""
    pass

def validate_input(payload: Dict[str, Any]) -> Dict[str, str]:
    """Validate and sanitize input parameters"""
    errors = []
    
    model_id = payload.get('model_id')
    query = payload.get('query')
    invoke_type = payload.get('invoke_type', 'llm')
    
    if not model_id or not isinstance(model_id, str):
        errors.append("model_id is required and must be a string")
    
    if not query or not isinstance(query, str):
        errors.append("query is required and must be a string")
    elif len(query.strip()) == 0:
        errors.append("query cannot be empty")
    elif len(query) > 100000:  # 100KB limit
        errors.append("query exceeds maximum length of 100,000 characters")
    
    if invoke_type not in ['llm', 'embedding']:
        errors.append("invoke_type must be either 'llm' or 'embedding'")
    
    # Validate model_kwargs if present
    model_kwargs = payload.get('model_kwargs', {})
    if not isinstance(model_kwargs, dict):
        errors.append("model_kwargs must be a dictionary")
    
    if errors:
        raise ValueError("; ".join(errors))
    
    return {
        'model_id': model_id.strip(),
        'query': query.strip(),
        'invoke_type': invoke_type,
        'model_kwargs': model_kwargs
    }

def sanitize_query(query: str) -> str:
    """Sanitize query to prevent potential issues"""
    # Remove null bytes and other problematic characters
    sanitized = query.replace('\x00', '').strip()
    return sanitized

def invoke_embedding_model(model_id: str, query: str) -> Dict[str, Any]:
    """Handle embedding model invocation"""
    try:
        logger.info(f"Invoking embedding model: {model_id}")
        
        body = {
            "inputText": sanitize_query(query)
        }
        
        logger.info(f"Embedding request body size: {len(json.dumps(body))} characters")
        
        start_time = time.time()
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            accept='application/json',
            contentType='application/json'
        )
        
        duration = time.time() - start_time
        logger.info(f"Embedding model response received in {duration:.2f} seconds")
        
        response_body = json.loads(response.get('body').read())
        
        if 'embedding' not in response_body:
            raise BedrockError("No embedding found in response")
        
        embedding = response_body.get('embedding', [])
        logger.info(f"Embedding vector length: {len(embedding)}")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache'
            },
            'body': json.dumps({
                'embedding': embedding,
                'model_id': model_id,
                'processing_time': round(duration, 2)
            })
        }
        
    except Exception as e:
        logger.error(f"Error in embedding model invocation: {str(e)}")
        raise BedrockError(f"Embedding model invocation failed: {str(e)}")

def invoke_llm_model(model_id: str, query: str, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Handle LLM model invocation"""
    try:
        logger.info(f"Invoking LLM model: {model_id}")
        
        # Set default LLM parameters
        default_kwargs = {
            "max_tokens": 2048,
            "temperature": 0.1,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": ["\n\nHuman"]
        }
        
        # Merge and validate kwargs
        final_kwargs = {**default_kwargs, **model_kwargs}
        
        # Validate parameter ranges
        if final_kwargs.get('max_tokens', 0) > 4096:
            final_kwargs['max_tokens'] = 4096
        if final_kwargs.get('temperature', 0) < 0 or final_kwargs.get('temperature', 0) > 1:
            final_kwargs['temperature'] = 0.1
        
        sanitized_query = sanitize_query(query)
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": sanitized_query}]}
            ],
            **final_kwargs
        }
        
        logger.info(f"LLM request body size: {len(json.dumps(body))} characters")
        logger.info(f"LLM parameters: max_tokens={final_kwargs['max_tokens']}, temperature={final_kwargs['temperature']}")
        
        start_time = time.time()
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )
        
        duration = time.time() - start_time
        logger.info(f"LLM model response received in {duration:.2f} seconds")
        
        response_body = json.loads(response.get("body").read())
        
        if 'content' not in response_body or not response_body['content']:
            raise BedrockError("No content found in response")
        
        result = response_body.get("content", [])[0].get("text", "")
        
        if not result:
            raise BedrockError("Empty response from model")
        
        logger.info(f"Model result length: {len(result)} characters")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache'
            },
            'body': json.dumps({
                'result': result,
                'model_id': model_id,
                'processing_time': round(duration, 2),
                'usage': response_body.get('usage', {})
            })
        }
        
    except Exception as e:
        logger.error(f"Error in LLM model invocation: {str(e)}")
        raise BedrockError(f"LLM model invocation failed: {str(e)}")

def parse_event_body(event: Dict[str, Any]) -> Dict[str, Any]:
    """Parse event body from ALB or direct invocation"""
    try:
        if 'body' in event:
            if event['body'] is None:
                return {}
            
            # Handle base64 encoded body
            if event.get('isBase64Encoded', False):
                import base64
                decoded_body = base64.b64decode(event['body']).decode('utf-8')
                return json.loads(decoded_body)
            
            # Handle regular JSON string
            if isinstance(event['body'], str):
                return json.loads(event['body'])
            
            # Handle already parsed body
            if isinstance(event['body'], dict):
                return event['body']
        
        # Direct Lambda invocation
        return event
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise ValueError(f"Invalid JSON in request body: {str(e)}")
    except Exception as e:
        logger.error(f"Error parsing event body: {e}")
        raise ValueError(f"Failed to parse request body: {str(e)}")

def execute(event, context):
    """
    Lambda handler to invoke a Bedrock model based on event input.
    Supports Claude LLM models and Titan embedding models with improved error handling.
    """
    request_id = context.aws_request_id if context else 'unknown'
    logger.info(f"Request ID: {request_id}")
    logger.info(f"Received event keys: {list(event.keys())}")
    
    try:
        # Parse the event body
        payload = parse_event_body(event)
        logger.info(f"Parsed payload keys: {list(payload.keys())}")
        
        # Validate input parameters
        validated_params = validate_input(payload)
        model_id = validated_params['model_id']
        query = validated_params['query']
        invoke_type = validated_params['invoke_type']
        model_kwargs = validated_params['model_kwargs']
        
        logger.info(f"Processing {invoke_type} request for model: {model_id}")
        logger.info(f"Query length: {len(query)} characters")
        
        # Route to appropriate handler
        if invoke_type == 'embedding':
            return invoke_embedding_model(model_id, query)
        else:
            return invoke_llm_model(model_id, query, model_kwargs)
            
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': 'Bad Request',
                'message': str(e),
                'request_id': request_id
            })
        }
    
    except BedrockError as e:
        logger.error(f"Bedrock error: {e}")
        return {
            'statusCode': 502,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': 'Bedrock Service Error',
                'message': str(e),
                'request_id': request_id
            })
        }
    
    except Exception as e:
        logger.exception(f"Unexpected error in request {request_id}: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred',
                'request_id': request_id
            })
        }

# Lambda handler alias
lambda_handler = execute