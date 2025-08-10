#!/usr/bin/env python3
"""
Interactive chat interface for text-to-SQL RAG system.
"""

import asyncio
import os
import sys
from pathlib import Path
import logging
import uuid
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
import motor.motor_asyncio
from opensearchpy import AsyncOpenSearch

# Load .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise for interactive mode
logger = logging.getLogger(__name__)

class TextToSQLChat:
    """Interactive chat interface for text-to-SQL conversions."""
    
    def __init__(self):
        self.mongo_client = None
        self.db = None
        self.opensearch_client = None
        self.services = {}
        self.current_session_id = str(uuid.uuid4())
        
    async def initialize(self):
        """Initialize all services."""
        try:
            print("Initializing Text-to-SQL RAG Chat Interface...")
            
            # Database connections
            mongodb_url = os.getenv("MONGODB_URL", "mongodb://admin:password@localhost:27017")
            database_name = os.getenv("MONGODB_DATABASE", "text_to_sql_rag")
            
            self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_url)
            self.db = self.mongo_client[database_name]
            
            opensearch_host = os.getenv("OPENSEARCH_HOST", "localhost")
            opensearch_port = int(os.getenv("OPENSEARCH_PORT", "9200"))
            self.opensearch_client = AsyncOpenSearch(
                hosts=[{"host": opensearch_host, "port": opensearch_port}],
                http_auth=None, use_ssl=False, verify_certs=False
            )
            
            # Import and initialize services
            from text_to_sql_rag.services.view_service import ViewService
            from text_to_sql_rag.services.embedding_service import EmbeddingService, VectorService
            from text_to_sql_rag.services.hitl_service import HITLService
            from text_to_sql_rag.services.bedrock_endpoint_service import BedrockEndpointService
            
            # Get configuration
            bedrock_endpoint = os.getenv("BEDROCK_ENDPOINT_URL", "https://8v1n9dbomk.execute-api.us-east-1.amazonaws.com/testaccess")
            embedding_model = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
            llm_model = os.getenv("BEDROCK_LLM_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
            use_mock = os.getenv("USE_MOCK_EMBEDDINGS", "false").lower() == "true"
            
            # Initialize services
            self.services = {
                'view': ViewService(self.db),
                'embedding': EmbeddingService(
                    endpoint_url=bedrock_endpoint,
                    embedding_model=embedding_model,
                    use_mock=use_mock
                ),
                'vector': VectorService(self.opensearch_client, "view_metadata", "embedding"),
                'hitl': HITLService(self.db),
                'bedrock': BedrockEndpointService(bedrock_endpoint, embedding_model, llm_model)
            }
            
            print("All services initialized successfully!")
            print(f"Session ID: {self.current_session_id}")
            return True
            
        except Exception as e:
            print(f"Failed to initialize services: {e}")
            return False
    
    async def process_query(self, user_query: str):
        """Process a user query through the complete RAG pipeline."""
        try:
            print(f"\nProcessing: '{user_query}'")
            
            # Step 1: Generate embedding for the query
            print("Generating query embedding...")
            query_embedding = await self.services['embedding'].get_embedding(user_query)
            print(f"   Generated {len(query_embedding)}-dimensional embedding")
            
            # Step 2: Search for relevant views
            print("Searching for relevant database views...")
            similar_views = await self.services['vector'].search_similar_views(query_embedding, k=3)
            
            if not similar_views:
                print("No relevant views found. You may need to index more data.")
                return None
            
            print(f"   Found {len(similar_views)} relevant views:")
            context_views = []
            for view, score in similar_views:
                context_views.append(view)
                print(f"   - {view.view_name} (similarity: {score:.3f})")
            
            # Step 3: Build context
            print("Building database context...")
            context = "\n\n".join([view.generate_full_text() for view in context_views])
            print(f"   Context prepared ({len(context)} characters)")
            
            # Step 4: Generate SQL
            print("Generating SQL with Claude 3 Haiku...")
            sql_response = await self.services['bedrock'].generate_sql(user_query, context)
            
            print(f"\nGenerated SQL:")
            print(f"```sql")
            print(f"{sql_response['sql']}")
            print(f"```")
            print(f"\nExplanation: {sql_response['explanation']}")
            
            # Step 5: Create HITL approval request
            print(f"\nCreating approval request...")
            request_id = await self.services['hitl'].create_approval_request(
                session_id=self.current_session_id,
                user_query=user_query,
                generated_sql=sql_response['sql'],
                sql_explanation=sql_response['explanation'],
                selected_views=[view.view_name for view in context_views]
            )
            
            print(f"Approval request created: {request_id}")
            
            return {
                'request_id': request_id,
                'sql': sql_response['sql'],
                'explanation': sql_response['explanation'],
                'context_views': [view.view_name for view in context_views],
                'similarity_scores': [score for _, score in similar_views]
            }
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            return None
    
    async def handle_approval(self, request_id: str, action: str, notes: str = None):
        """Handle approval/rejection of a HITL request."""
        try:
            if action.lower() in ['approve', 'approved', 'yes', 'y']:
                result = await self.services['hitl'].approve_request(
                    request_id=request_id,
                    reviewer_notes=notes or "Approved via chat interface"
                )
                if result:
                    print("✅ [APPROVED] SQL query has been approved!")
                    print("🚀 In a production system, this SQL would now be executed.")
                    return True
                else:
                    print("❌ Failed to approve request")
                    return False
                    
            elif action.lower() in ['reject', 'rejected', 'no', 'n']:
                result = await self.services['hitl'].reject_request(
                    request_id=request_id,
                    reviewer_notes=notes or "Rejected via chat interface"
                )
                if result:
                    print("❌ [REJECTED] SQL query has been rejected.")
                    return True
                else:
                    print("❌ Failed to reject request")
                    return False
            else:
                print("⚠️  Invalid action. Use 'approve' or 'reject'")
                return False
                
        except Exception as e:
            print(f"❌ Error handling approval: {e}")
            return False
    
    def print_help(self):
        """Print available commands."""
        print("\n" + "="*50)
        print("AVAILABLE COMMANDS:")
        print("="*50)
        print("• Ask any natural language question about your data")
        print("• 'approve' - Approve the last generated SQL")
        print("• 'reject' - Reject the last generated SQL") 
        print("• 'help' - Show this help message")
        print("• 'quit' or 'exit' - Exit the chat")
        print("• 'session' - Show current session info")
        print("• 'stats' - Show system statistics")
        print("\nExample queries:")
        print("- 'Show me user engagement metrics for active users'")
        print("- 'What are the top performing investment tranches?'")
        print("- 'List all syndicate participation data'")
        print("="*50)
    
    async def show_stats(self):
        """Show system statistics."""
        try:
            print("\n📊 SYSTEM STATISTICS")
            print("="*40)
            
            # MongoDB stats
            db_stats = await self.db.command("dbstats")
            view_count = await self.db.view_metadata.count_documents({})
            print(f"📄 Documents in MongoDB: {view_count}")
            print(f"💾 Database size: {db_stats.get('dataSize', 0) / 1024:.1f} KB")
            
            # OpenSearch stats
            try:
                indices_info = await self.opensearch_client.cat.indices(format="json")
                for index in indices_info:
                    if index['index'] == 'view_metadata':
                        print(f"🔍 Documents in OpenSearch: {index['docs.count']}")
                        print(f"📊 Index size: {index['store.size']}")
                        break
            except:
                print("⚠️  Could not retrieve OpenSearch stats")
            
            # HITL stats
            hitl_stats = await self.services['hitl'].get_stats()
            print(f"👤 Total approval requests: {hitl_stats.get('total_requests', 0)}")
            print(f"⏳ Pending requests: {hitl_stats.get('pending_requests', 0)}")
            print(f"✅ Approved requests: {hitl_stats.get('approved_requests', 0)}")
            
            print("="*40)
            
        except Exception as e:
            print(f"❌ Error retrieving stats: {e}")
    
    async def run_interactive_chat(self):
        """Run the interactive chat loop."""
        if not await self.initialize():
            return
        
        print("\n🎉 Welcome to the Text-to-SQL RAG Chat Interface!")
        print("Type 'help' for available commands or start asking questions about your data.")
        print("="*60)
        
        last_request_id = None
        
        try:
            while True:
                try:
                    # Get user input
                    user_input = input(f"\n💬 You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        print("👋 Goodbye! Thanks for using the Text-to-SQL system.")
                        break
                    
                    elif user_input.lower() == 'help':
                        self.print_help()
                        continue
                    
                    elif user_input.lower() == 'session':
                        print(f"📱 Current session: {self.current_session_id}")
                        print(f"🕐 Started: {datetime.utcnow().strftime('%H:%M:%S UTC')}")
                        continue
                    
                    elif user_input.lower() == 'stats':
                        await self.show_stats()
                        continue
                    
                    elif user_input.lower() in ['approve', 'approved', 'yes', 'y']:
                        if last_request_id:
                            await self.handle_approval(last_request_id, 'approve')
                        else:
                            print("⚠️  No pending request to approve")
                        continue
                    
                    elif user_input.lower() in ['reject', 'rejected', 'no', 'n']:
                        if last_request_id:
                            await self.handle_approval(last_request_id, 'reject')
                        else:
                            print("⚠️  No pending request to reject")
                        continue
                    
                    # Process as a query
                    result = await self.process_query(user_input)
                    if result:
                        last_request_id = result['request_id']
                        print(f"\n👤 Type 'approve' to approve this SQL or 'reject' to reject it.")
                
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye! (Ctrl+C detected)")
                    break
                except EOFError:
                    print("\n\n👋 Goodbye! (EOF detected)")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
        
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.mongo_client:
                self.mongo_client.close()
            if self.opensearch_client:
                await self.opensearch_client.close()
            print("🔚 Cleaned up resources")
        except:
            pass

async def main():
    """Main entry point."""
    chat = TextToSQLChat()
    await chat.run_interactive_chat()

if __name__ == "__main__":
    asyncio.run(main())