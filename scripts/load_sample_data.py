#!/usr/bin/env python3
"""
Load sample metadata files into MongoDB and create vector embeddings in OpenSearch.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
import motor.motor_asyncio
from opensearchpy import AsyncOpenSearch

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Load sample data into the system."""
    
    def __init__(self):
        # Database connections
        self.mongodb_url = os.getenv("MONGODB_URL", "mongodb://admin:password@localhost:27017")
        self.database_name = os.getenv("MONGODB_DATABASE", "text_to_sql_rag")
        
        # OpenSearch configuration  
        self.opensearch_host = os.getenv("OPENSEARCH_HOST", "localhost")
        self.opensearch_port = int(os.getenv("OPENSEARCH_PORT", "9200"))
        
        # Bedrock configuration
        self.bedrock_endpoint = os.getenv("BEDROCK_ENDPOINT_URL", "https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com/prod")
        self.embedding_model = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
        self.use_mock = os.getenv("USE_MOCK_EMBEDDINGS", "false").lower() == "true"
        
        # Service instances
        self.mongo_client = None
        self.db = None
        self.opensearch_client = None
        self.embedding_service = None
        self.vector_service = None
        
        # Data paths
        self.meta_docs_path = Path(__file__).parent.parent / "meta_documents"
    
    async def initialize(self):
        """Initialize database connections and services."""
        try:
            # MongoDB connection
            self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(self.mongodb_url)
            self.db = self.mongo_client[self.database_name]
            await self.mongo_client.server_info()
            logger.info("✅ MongoDB connected")
            
            # OpenSearch connection
            self.opensearch_client = AsyncOpenSearch(
                hosts=[{"host": self.opensearch_host, "port": self.opensearch_port}],
                http_auth=None, use_ssl=False, verify_certs=False
            )
            cluster_info = await self.opensearch_client.info()
            logger.info(f"✅ OpenSearch connected: {cluster_info['version']['number']}")
            
            # Import and initialize services
            from text_to_sql_rag.services.embedding_service import EmbeddingService, VectorService
            
            self.embedding_service = EmbeddingService(
                endpoint_url=self.bedrock_endpoint,
                embedding_model=self.embedding_model,
                use_mock=self.use_mock
            )
            self.vector_service = VectorService(self.opensearch_client, "view_metadata", "embedding")
            
            logger.info("✅ All services initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
            return False
    
    async def ensure_opensearch_index(self):
        """Create OpenSearch index if it doesn't exist."""
        try:
            index_name = "view_metadata"
            
            # Check if index exists
            if await self.opensearch_client.indices.exists(index_name):
                logger.info(f"📊 Index '{index_name}' already exists")
                return True
            
            # Create index with proper mapping
            vector_size = 1536 if self.use_mock else 1024
            
            # First create index with k-NN settings
            index_settings = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                },
                "mappings": {
                    "properties": {
                        "view_name": {"type": "keyword"},
                        "schema": {"type": "keyword"}, 
                        "description": {"type": "text"},
                        "business_context": {"type": "text"},
                        "full_text": {"type": "text"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": vector_size,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil"
                            }
                        },
                        "_uploaded_at": {"type": "date"},
                        "_source_file": {"type": "keyword"}
                    }
                }
            }
            
            await self.opensearch_client.indices.create(index_name, body=index_settings)
            logger.info(f"✅ Created OpenSearch index '{index_name}' with {vector_size}-dimensional vectors")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create OpenSearch index: {e}")
            return False
    
    async def clear_existing_data(self):
        """Clear existing data from MongoDB and OpenSearch."""
        try:
            # Clear MongoDB collections
            collections = ['view_metadata', 'hitl_requests', 'session_states']
            for collection_name in collections:
                result = await self.db[collection_name].delete_many({})
                logger.info(f"🗑️  Cleared {result.deleted_count} documents from {collection_name}")
            
            # Clear OpenSearch index
            try:
                await self.opensearch_client.indices.delete("view_metadata")
                logger.info("🗑️  Deleted OpenSearch index")
            except:
                logger.info("📊 OpenSearch index didn't exist")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clear existing data: {e}")
            return False
    
    async def load_business_domains(self):
        """Load business domains from meta_documents/business_domains.json."""
        try:
            business_domains_file = self.meta_docs_path / "business_domains.json"
            
            if not business_domains_file.exists():
                logger.warning(f"Business domains file not found: {business_domains_file}")
                return 0
            
            with open(business_domains_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            domains = data.get('business_domains', [])
            if not domains:
                logger.warning("No business domains found in file")
                return 0
            
            # Store each domain as a separate document
            loaded_count = 0
            for domain in domains:
                domain['_uploaded_at'] = datetime.utcnow()
                domain['_source_file'] = 'business_domains.json'
                domain['_document_type'] = 'business_domain'
                
                result = await self.db.view_metadata.insert_one(domain)
                if result.inserted_id:
                    loaded_count += 1
                    logger.info(f"Loaded business domain: {domain.get('domain_name', 'Unknown')}")
            
            logger.info(f"Loaded {loaded_count} business domain documents")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to load business domains: {e}")
            return 0
    
    async def load_view_metadata(self):
        """Load view metadata files from meta_documents/views/."""
        return await self._load_documents_from_directory("views", "view_metadata")
    
    async def load_report_metadata(self):
        """Load report metadata files from meta_documents/reports/."""
        return await self._load_documents_from_directory("reports", "report_metadata")
    
    async def load_lookup_metadata(self):
        """Load lookup metadata files from meta_documents/lookups/."""
        return await self._load_documents_from_directory("lookups", "lookup_metadata")
    
    async def _load_documents_from_directory(self, directory_name: str, document_type: str):
        """Load documents from a specific directory."""
        try:
            dir_path = self.meta_docs_path / directory_name
            
            if not dir_path.exists():
                logger.warning(f"Directory not found: {dir_path}")
                return 0
            
            json_files = list(dir_path.glob("*.json"))
            if not json_files:
                logger.warning(f"No JSON files found in {dir_path}")
                return 0
            
            logger.info(f"Found {len(json_files)} {document_type} files")
            
            loaded_count = 0
            for json_file in json_files:
                try:
                    # Skip README files
                    if json_file.name.lower().startswith('readme'):
                        continue
                        
                    # Load JSON data
                    with open(json_file, 'r', encoding='utf-8') as f:
                        doc_data = json.load(f)
                    
                    # Add metadata
                    doc_data['_uploaded_at'] = datetime.utcnow()
                    doc_data['_source_file'] = json_file.name
                    doc_data['_document_type'] = document_type
                    
                    # Ensure identifier field based on document type
                    if document_type == 'view_metadata' and 'view_name' not in doc_data:
                        doc_data['view_name'] = json_file.stem.upper()
                    elif document_type == 'report_metadata' and 'report_name' not in doc_data:
                        doc_data['report_name'] = json_file.stem.replace('_', ' ').title()
                    elif document_type == 'lookup_metadata' and 'lookup_name' not in doc_data:
                        doc_data['lookup_name'] = doc_data.get('name', json_file.stem.replace('_', ' ').title())
                    
                    # Store in MongoDB
                    result = await self.db.view_metadata.insert_one(doc_data)
                    
                    if result.inserted_id:
                        loaded_count += 1
                        # Log appropriate identifier
                        identifier = (doc_data.get('view_name') or 
                                    doc_data.get('report_name') or 
                                    doc_data.get('lookup_name') or 
                                    doc_data.get('name', json_file.stem))
                        logger.info(f"Loaded: {identifier}")
                    
                except Exception as e:
                    logger.error(f"Error loading {json_file.name}: {e}")
            
            logger.info(f"Loaded {loaded_count} {document_type} documents")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to load {document_type}: {e}")
            return 0
    
    async def generate_embeddings_and_index(self):
        """Generate embeddings for all loaded documents and index in OpenSearch."""
        try:
            # Get all documents from MongoDB
            cursor = self.db.view_metadata.find({})
            documents = []
            async for doc in cursor:
                documents.append(doc)
            
            if not documents:
                logger.warning("⚠️  No documents found to embed")
                return 0
            
            logger.info(f"🔄 Generating embeddings for {len(documents)} documents...")
            
            embedded_count = 0
            for doc in documents:
                try:
                    # Import all metadata models
                    from text_to_sql_rag.models.view_models import ViewMetadata, ReportMetadata, LookupMetadata, LookupValue
                    
                    # Generate full text based on document type
                    document_type = doc.get('_document_type', 'view_metadata')
                    identifier = (doc.get('view_name') or doc.get('report_name') or 
                                doc.get('lookup_name') or doc.get('domain_name') or 
                                doc.get('name', 'unknown'))
                    logger.info(f"Processing {document_type}: {identifier}")
                    
                    if document_type == 'view_metadata':
                        # Convert columns to proper format
                        columns = []
                        for col in doc.get('columns', []):
                            if isinstance(col, dict):
                                columns.append({
                                    'name': col.get('name', ''),
                                    'type': col.get('type', ''),
                                    'notNull': col.get('notNull', False),
                                    'description': col.get('description', '')
                                })
                        
                        metadata_obj = ViewMetadata(
                            view_name=doc.get('view_name', ''),
                            view_type=doc.get('view_type', 'CORE'),
                            schema_name=doc.get('schema', 'default'),
                            description=doc.get('description', ''),
                            use_cases=doc.get('business_context', ''),
                            columns=columns,
                            sample_sql=doc.get('sample_queries', [{}])[0] if doc.get('sample_queries') else None
                        )
                    
                    elif document_type == 'report_metadata':
                        # Simplified approach - create text directly
                        report_name = doc.get('report_name', doc.get('view_name', ''))
                        view_name = doc.get('view_name', '')
                        report_desc = doc.get('report_description', doc.get('description', ''))
                        data_returned = doc.get('data_returned', '')
                        use_cases = doc.get('use_cases', '')
                        example_sql = doc.get('example_sql', '')
                        
                        # Create full text manually
                        parts = [
                            f"Report: {report_name}",
                            f"Type: STANDARD",
                        ]
                        
                        if view_name:
                            parts.append(f"View: {view_name}")
                        if report_desc:
                            parts.append(f"Description: {report_desc}")
                        if use_cases:
                            parts.append(f"Use Cases: {use_cases}")
                        if data_returned:
                            parts.append(f"Data Returned: {data_returned}")
                        if example_sql:
                            parts.append(f"Example SQL: {example_sql}")
                        
                        full_text = "\n".join(parts)
                        metadata_obj = None  # No metadata object for simplified approach
                    
                    elif document_type == 'lookup_metadata':
                        # Simplified approach - create text directly
                        lookup_name = doc.get('lookup_name', doc.get('name', ''))
                        lookup_desc = doc.get('description', '')
                        use_cases = doc.get('use_cases', '')
                        values = doc.get('values', [])
                        
                        # Create full text manually
                        parts = [
                            f"Lookup: {lookup_name}",
                            f"Type: REFERENCE",
                        ]
                        
                        if lookup_desc:
                            parts.append(f"Description: {lookup_desc}")
                        if use_cases:
                            parts.append(f"Use Cases: {use_cases}")
                        
                        # Add values summary
                        if values:
                            value_summary = []
                            for value in values[:10]:  # Limit to first 10 for space
                                if isinstance(value, dict):
                                    value_text = f"{value.get('code', '')} ({value.get('name', '')})"
                                    if value.get('description'):
                                        value_text += f": {value.get('description')}"
                                    value_summary.append(value_text)
                            
                            parts.append(f"Values: {'; '.join(value_summary)}")
                            
                            if len(values) > 10:
                                parts.append(f"... and {len(values) - 10} more values")
                        
                        full_text = "\n".join(parts)
                        metadata_obj = None  # No metadata object for simplified approach
                    
                    elif document_type == 'business_domain':
                        # Handle business domains
                        domain_name = doc.get('domain_name', '')
                        domain_desc = doc.get('description', '')
                        keywords = doc.get('keywords', [])
                        core_views = doc.get('core_views', [])
                        supporting_views = doc.get('supporting_views', [])
                        
                        # Create full text manually
                        parts = [
                            f"Business Domain: {domain_name}",
                            f"ID: {doc.get('domain_id', 'N/A')}",
                        ]
                        
                        if domain_desc:
                            parts.append(f"Description: {domain_desc}")
                        if keywords:
                            parts.append(f"Keywords: {', '.join(keywords)}")
                        if core_views:
                            parts.append(f"Core Views: {', '.join(core_views)}")
                        if supporting_views:
                            parts.append(f"Supporting Views: {', '.join(supporting_views)}")
                        
                        full_text = "\n".join(parts)
                        metadata_obj = None  # No metadata object for simplified approach
                    
                    else:
                        logger.warning(f"Unknown document type: {document_type}")
                        continue
                    
                    # Generate full text for embedding
                    if metadata_obj:
                        full_text = metadata_obj.generate_full_text()
                    # For simplified approach (reports/lookups), full_text is already created above
                    
                    # Generate embedding
                    embedding = await self.embedding_service.get_embedding(full_text)
                    
                    if not embedding:
                        logger.warning(f"Failed to generate embedding for {identifier}")
                        continue
                    
                    # Index in OpenSearch - unified structure for all document types
                    opensearch_doc = {
                        "document_type": document_type,
                        "identifier": identifier,
                        "full_text": full_text,
                        "embedding": embedding,
                        "_uploaded_at": doc.get('_uploaded_at'),
                        "_source_file": doc.get('_source_file')
                    }
                    
                    # Add type-specific fields
                    if document_type == 'view_metadata' and metadata_obj:
                        opensearch_doc.update({
                            "view_name": metadata_obj.view_name,
                            "schema": metadata_obj.schema_name,
                            "description": metadata_obj.description,
                            "use_cases": metadata_obj.use_cases
                        })
                    elif document_type == 'report_metadata':
                        opensearch_doc.update({
                            "report_name": doc.get('report_name', doc.get('view_name', '')),
                            "view_name": doc.get('view_name', ''),
                            "description": doc.get('report_description', doc.get('description', '')),
                            "use_cases": doc.get('use_cases', '')
                        })
                    elif document_type == 'lookup_metadata':
                        opensearch_doc.update({
                            "lookup_name": doc.get('lookup_name', doc.get('name', '')),
                            "description": doc.get('description', ''),
                            "use_cases": doc.get('use_cases', ''),
                            "values_count": len(doc.get('values', []))
                        })
                    elif document_type == 'business_domain':
                        opensearch_doc.update({
                            "domain_name": doc.get('domain_name', ''),
                            "domain_id": doc.get('domain_id'),
                            "description": doc.get('description', ''),
                            "keywords": doc.get('keywords', []),
                            "core_views": doc.get('core_views', []),
                            "supporting_views": doc.get('supporting_views', [])
                        })
                    
                    await self.opensearch_client.index(
                        index="view_metadata",
                        id=str(doc['_id']),
                        body=opensearch_doc
                    )
                    
                    embedded_count += 1
                    logger.info(f"Embedded and indexed: {identifier} ({len(embedding)}-dim)")
                    
                except Exception as e:
                    logger.error(f"Error embedding document {identifier}: {e}")
            
            # Refresh index
            await self.opensearch_client.indices.refresh("view_metadata")
            
            logger.info(f"✅ Generated and indexed {embedded_count} embeddings")
            return embedded_count
            
        except Exception as e:
            logger.error(f"❌ Failed to generate embeddings: {e}")
            return 0
    
    async def verify_data_load(self):
        """Verify that data was loaded correctly."""
        try:
            logger.info("🔍 Verifying data load...")
            
            # Check MongoDB
            mongo_count = await self.db.view_metadata.count_documents({})
            logger.info(f"📊 MongoDB documents: {mongo_count}")
            
            # Check OpenSearch
            try:
                opensearch_count = await self.opensearch_client.count(index="view_metadata")
                logger.info(f"🔍 OpenSearch documents: {opensearch_count['count']}")
                
                # Test a sample search using generic method
                if opensearch_count['count'] > 0:
                    test_embedding = await self.embedding_service.get_embedding("user metrics")
                    search_result = await self.vector_service.search_similar_documents(test_embedding, k=1)
                    
                    if search_result:
                        doc, score = search_result[0]
                        identifier = (doc.get('view_name') or doc.get('report_name') or 
                                    doc.get('lookup_name') or doc.get('domain_name') or 'Unknown')
                        logger.info(f"✅ Test search successful: {identifier} (similarity: {score:.3f})")
                    else:
                        logger.warning("⚠️  Test search returned no results")
                        
            except Exception as e:
                logger.error(f"❌ OpenSearch verification failed: {e}")
            
            return mongo_count > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to verify data load: {e}")
            return False
    
    async def cleanup(self):
        """Clean up connections."""
        try:
            if self.mongo_client:
                self.mongo_client.close()
            if self.opensearch_client:
                await self.opensearch_client.close()
            logger.info("🔚 Cleaned up connections")
        except:
            pass

async def main():
    """Main data loading process."""
    print("SAMPLE DATA LOADER")
    print("="*50)
    
    loader = DataLoader()
    
    try:
        # Initialize
        if not await loader.initialize():
            print("Failed to initialize loader")
            return False
        
        # Clear existing data (optional - comment out to append)
        print("\nClearing existing data...")
        await loader.clear_existing_data()
        
        # Ensure OpenSearch index exists
        print("\nSetting up OpenSearch index...")
        if not await loader.ensure_opensearch_index():
            print("Failed to setup OpenSearch index - you may need to create it manually")
            print("Try running: curl -X PUT 'localhost:9200/view_metadata' with proper mapping")
            print("Continuing anyway - index may already exist...")
            # Don't fail here, index might already exist
        
        # Load all metadata types (hierarchical order: domains first)
        print("\nLoading metadata files...")
        domain_count = await loader.load_business_domains()
        view_count = await loader.load_view_metadata()
        report_count = await loader.load_report_metadata()
        lookup_count = await loader.load_lookup_metadata()
        
        total_loaded = domain_count + view_count + report_count + lookup_count
        
        if total_loaded == 0:
            print("No data loaded")
            return False
        
        print(f"Total loaded: {total_loaded} documents (Domains: {domain_count}, Views: {view_count}, Reports: {report_count}, Lookups: {lookup_count})")
        
        # Generate embeddings and index
        print("\nGenerating embeddings and indexing...")
        embedded_count = await loader.generate_embeddings_and_index()
        
        if embedded_count == 0:
            print("No embeddings generated")
            return False
        
        # Verify data load
        print("\nVerifying data load...")
        if not await loader.verify_data_load():
            print("Data verification failed")
            return False
        
        print("\nDATA LOADING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Documents loaded: {total_loaded}")
        print(f"Embeddings created: {embedded_count}")
        print("\nYou can now:")
        print("• Run 'poetry run python chat_interface.py' for interactive chat")
        print("• Run 'poetry run python tests/test_rag_pipeline.py' to test the system")
        print("• Visit http://localhost:5601 to view data in OpenSearch Dashboard")
        
        return True
        
    finally:
        await loader.cleanup()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)