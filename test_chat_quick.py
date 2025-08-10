#!/usr/bin/env python3
"""Quick test of the chat interface."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chat_interface import TextToSQLChat

async def quick_chat_test():
    """Quick test of the chat system."""
    print("TESTING CHAT INTERFACE")
    print("="*30)
    
    chat = TextToSQLChat()
    
    try:
        # Initialize
        if not await chat.initialize():
            print("Failed to initialize chat system")
            return False
        
        # Test a query
        test_query = "Show me user engagement metrics"
        print(f"\nTesting query: '{test_query}'")
        
        result = await chat.process_query(test_query)
        
        if result:
            print(f"\nChat test successful!")
            print(f"Generated SQL preview: {result['sql'][:100]}...")
            print(f"Request ID: {result['request_id']}")
            return True
        else:
            print("Chat test failed")
            return False
            
    finally:
        await chat.cleanup()

if __name__ == "__main__":
    success = asyncio.run(quick_chat_test())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")