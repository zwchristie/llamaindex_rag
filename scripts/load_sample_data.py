#!/usr/bin/env python3
"""
Load sample metadata files into MongoDB and create vector embeddings in OpenSearch.
Uses the actual application services to ensure consistent configuration.
"""

import asyncio
import json
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """Load sample data using actual application services."""
    
    def __init__(self):
        # Import application services and settings
        from text_to_sql_rag.config.settings import settings
        from text_to_sql_rag.services.mongodb_service import MongoDBService
        from text_to_sql_rag.services.vector_service import LlamaIndexVectorService
        from text_to_sql_rag.services.enhanced_bedrock_service import EnhancedBedrockService
        
        self.settings = settings
        
        # Initialize actual application services
        self.mongodb_service = None
        self.vector_service = None
        self.bedrock_service = None
        
        # Data paths
        self.meta_docs_path = Path(__file__).parent.parent / "meta_documents"
        
        logger.info("Data loader initialized with actual application services")
    
    async def initialize(self):
        """Initialize services using actual application configuration."""
        try:
            # Debug: Print configuration values
            logger.info(f"DEBUG: BEDROCK_ENDPOINT_URL = '{self.settings.bedrock_endpoint.url}'")
            logger.info(f"DEBUG: BEDROCK_VERIFY_SSL = {self.settings.bedrock_endpoint.verify_ssl}")
            logger.info(f"DEBUG: OPENSEARCH_HOST = {self.settings.opensearch.host}")
            logger.info(f"DEBUG: OPENSEARCH_USE_SSL = {self.settings.opensearch.use_ssl}")
            logger.info(f"DEBUG: OPENSEARCH_VERIFY_CERTS = {self.settings.opensearch.verify_certs}")
            logger.info(f"DEBUG: OPENSEARCH_AUTH = {bool(self.settings.opensearch.get_http_auth())}")
            
            # Initialize MongoDB service
            from text_to_sql_rag.services.mongodb_service import MongoDBService
            self.mongodb_service = MongoDBService()
            
            if not self.mongodb_service.is_connected():
                logger.error("❌ MongoDB service not connected")
                return False
            
            logger.info("✅ MongoDB service connected")
            
            # Initialize Bedrock service with enhanced configuration
            if not self.settings.bedrock_endpoint.url:
                logger.error("❌ BEDROCK_ENDPOINT_URL not configured in .env file")
                logger.error("Please set BEDROCK_ENDPOINT_URL=https://your-endpoint-url.com in your .env file")
                return False
                
            from text_to_sql_rag.services.enhanced_bedrock_service import EnhancedBedrockService
            self.bedrock_service = EnhancedBedrockService(
                endpoint_url=self.settings.bedrock_endpoint.url,
                embedding_model=self.settings.aws.embedding_model,
                llm_model=self.settings.aws.llm_model,
                verify_ssl=self.settings.bedrock_endpoint.verify_ssl,
                ssl_cert_file=self.settings.bedrock_endpoint.ssl_cert_file,
                ssl_key_file=self.settings.bedrock_endpoint.ssl_key_file,
                ssl_ca_file=self.settings.bedrock_endpoint.ssl_ca_file,
                http_auth_username=self.settings.bedrock_endpoint.http_auth_username,
                http_auth_password=self.settings.bedrock_endpoint.http_auth_password
            )
            
            # Initialize Vector service (LlamaIndex with OpenSearch)
            from text_to_sql_rag.services.vector_service import LlamaIndexVectorService
            self.vector_service = LlamaIndexVectorService()
            
            if not self.vector_service.health_check():
                logger.error("❌ Vector service health check failed")
                return False
                
            logger.info("✅ Vector service initialized")
            
            logger.info("✅ All application services initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize services: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def load_business_domains(self) -> int:
        """Load business domains into MongoDB."""
        try:
            domains_file = self.meta_docs_path / "business_domains.json"
            if not domains_file.exists():
                logger.warning(f"Business domains file not found: {domains_file}")
                return 0
            
            with open(domains_file, 'r') as f:
                domains = json.load(f)
            
            # Clear existing domains
            self.mongodb_service.documents_collection.delete_many({"document_type": "business_domain"})
            
            inserted_count = 0
            for domain in domains:
                domain_doc = {
                    "document_type": "business_domain",
                    "domain_id": domain["domain_id"],
                    "name": domain["name"],
                    "description": domain["description"],
                    "keywords": domain["keywords"],
                    "related_views": domain.get("related_views", []),
                    "metadata": domain,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                
                result = self.mongodb_service.documents_collection.insert_one(domain_doc)
                if result.inserted_id:
                    inserted_count += 1
            
            logger.info(f"✅ Loaded {inserted_count} business domains")
            return inserted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to load business domains: {e}")
            return 0
    
    async def load_view_metadata(self) -> int:
        """Load view metadata into MongoDB."""
        try:
            views_dir = self.meta_docs_path / "views"
            if not views_dir.exists():
                logger.warning(f"Views directory not found: {views_dir}")
                return 0
            
            # Clear existing views
            self.mongodb_service.documents_collection.delete_many({"document_type": "view"})
            
            inserted_count = 0
            view_files = list(views_dir.glob("*.json"))
            
            for view_file in view_files:
                try:
                    with open(view_file, 'r') as f:
                        view_data = json.load(f)
                    
                    view_doc = {
                        "document_type": "view",
                        "view_name": view_data["view_name"],
                        "schema_name": view_data.get("schema_name", ""),
                        "description": view_data.get("description", ""),
                        "business_domains": view_data.get("business_domains", []),
                        "columns": view_data.get("columns", []),
                        "sample_queries": view_data.get("sample_queries", []),
                        "metadata": view_data,
                        "content": self._generate_view_content(view_data),
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                    
                    result = self.mongodb_service.documents_collection.insert_one(view_doc)
                    if result.inserted_id:
                        inserted_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ Failed to load view {view_file}: {e}")
            
            logger.info(f"✅ Loaded {inserted_count} view metadata documents")
            return inserted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to load view metadata: {e}")
            return 0
    
    async def load_reports(self) -> int:
        """Load report metadata into MongoDB."""
        try:
            reports_dir = self.meta_docs_path / "reports"
            if not reports_dir.exists():
                logger.warning(f"Reports directory not found: {reports_dir}")
                return 0
            
            # Clear existing reports
            self.mongodb_service.documents_collection.delete_many({"document_type": "report"})
            
            inserted_count = 0
            report_files = list(reports_dir.glob("*.json"))
            
            for report_file in report_files:
                try:
                    with open(report_file, 'r') as f:
                        report_data = json.load(f)
                    
                    report_doc = {
                        "document_type": "report",
                        "report_name": report_data["report_name"],
                        "description": report_data.get("description", ""),
                        "business_domains": report_data.get("business_domains", []),
                        "sql_examples": report_data.get("sql_examples", []),
                        "metadata": report_data,
                        "content": self._generate_report_content(report_data),
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                    
                    result = self.mongodb_service.documents_collection.insert_one(report_doc)
                    if result.inserted_id:
                        inserted_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ Failed to load report {report_file}: {e}")
            
            logger.info(f"✅ Loaded {inserted_count} report documents")
            return inserted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to load reports: {e}")
            return 0
    
    async def load_lookups(self) -> int:
        """Load lookup metadata into MongoDB."""
        try:
            lookups_dir = self.meta_docs_path / "lookups"
            if not lookups_dir.exists():
                logger.warning(f"Lookups directory not found: {lookups_dir}")
                return 0
            
            # Clear existing lookups
            self.mongodb_service.documents_collection.delete_many({"document_type": "lookup"})
            
            inserted_count = 0
            lookup_files = list(lookups_dir.glob("*.json"))
            
            for lookup_file in lookup_files:
                try:
                    with open(lookup_file, 'r') as f:
                        lookup_data = json.load(f)
                    
                    lookup_doc = {
                        "document_type": "lookup",
                        "lookup_name": lookup_data["lookup_name"],
                        "lookup_id": lookup_data.get("lookup_id"),
                        "description": lookup_data.get("description", ""),
                        "values": lookup_data.get("values", []),
                        "metadata": lookup_data,
                        "content": self._generate_lookup_content(lookup_data),
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                    
                    result = self.mongodb_service.documents_collection.insert_one(lookup_doc)
                    if result.inserted_id:
                        inserted_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ Failed to load lookup {lookup_file}: {e}")
            
            logger.info(f"✅ Loaded {inserted_count} lookup documents")
            return inserted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to load lookups: {e}")
            return 0
    
    async def create_embeddings(self) -> int:
        """Create embeddings for all documents using the vector service."""
        try:
            # Get all documents from MongoDB
            documents = list(self.mongodb_service.get_all_documents())
            
            if not documents:
                logger.warning("No documents found in MongoDB for embedding")
                return 0
            
            logger.info(f"Creating embeddings for {len(documents)} documents...")
            
            embedded_count = 0
            
            # Use the actual vector service to add documents
            for doc in documents:
                try:
                    document_id = doc.get("view_name") or doc.get("report_name") or doc.get("lookup_name") or str(doc["_id"])
                    content = doc.get("content", "")
                    
                    if not content:
                        logger.warning(f"No content for document {document_id}")
                        continue
                    
                    # Use the vector service to add the document
                    success = self.vector_service.add_document(
                        document_id=document_id,
                        content=content,
                        metadata=doc.get("metadata", {}),
                        document_type=doc.get("document_type", "unknown")
                    )
                    
                    if success:
                        embedded_count += 1
                        if embedded_count % 10 == 0:
                            logger.info(f"Embedded {embedded_count}/{len(documents)} documents...")
                    else:
                        logger.error(f"Failed to embed document: {document_id}")
                        
                except Exception as e:
                    logger.error(f"Error embedding document: {e}")
            
            logger.info(f"✅ Created embeddings for {embedded_count} documents")
            return embedded_count
            
        except Exception as e:
            logger.error(f"❌ Failed to create embeddings: {e}")
            return 0
    
    def _generate_view_content(self, view_data: Dict[str, Any]) -> str:
        """Generate searchable content from view metadata."""
        content_parts = []
        
        content_parts.append(f"View: {view_data.get('view_name', '')}")
        content_parts.append(f"Description: {view_data.get('description', '')}")
        
        if view_data.get('business_domains'):
            content_parts.append(f"Business Domains: {', '.join(map(str, view_data['business_domains']))}")
        
        if view_data.get('columns'):
            columns_text = []
            for col in view_data['columns']:
                col_desc = f"{col.get('name', '')} ({col.get('data_type', '')})"
                if col.get('description'):
                    col_desc += f" - {col['description']}"
                columns_text.append(col_desc)
            content_parts.append(f"Columns: {'; '.join(columns_text)}")
        
        if view_data.get('sample_queries'):
            content_parts.append("Sample Queries:")
            for query in view_data['sample_queries']:
                content_parts.append(f"- {query.get('description', '')}: {query.get('sql', '')}")
        
        return '\n'.join(content_parts)
    
    def _generate_report_content(self, report_data: Dict[str, Any]) -> str:
        """Generate searchable content from report metadata."""
        content_parts = []
        
        content_parts.append(f"Report: {report_data.get('report_name', '')}")
        content_parts.append(f"Description: {report_data.get('description', '')}")
        
        if report_data.get('business_domains'):
            content_parts.append(f"Business Domains: {', '.join(map(str, report_data['business_domains']))}")
        
        if report_data.get('sql_examples'):
            content_parts.append("SQL Examples:")
            for example in report_data['sql_examples']:
                content_parts.append(f"- {example.get('description', '')}: {example.get('sql', '')}")
        
        return '\n'.join(content_parts)
    
    def _generate_lookup_content(self, lookup_data: Dict[str, Any]) -> str:
        """Generate searchable content from lookup metadata."""
        content_parts = []
        
        content_parts.append(f"Lookup: {lookup_data.get('lookup_name', '')}")
        content_parts.append(f"Description: {lookup_data.get('description', '')}")
        
        if lookup_data.get('lookup_id'):
            content_parts.append(f"Lookup ID: {lookup_data['lookup_id']}")
        
        if lookup_data.get('values'):
            values_text = []
            for value in lookup_data['values']:
                if isinstance(value, dict):
                    value_desc = f"{value.get('key', '')}: {value.get('value', '')}"
                    if value.get('description'):
                        value_desc += f" - {value['description']}"
                    values_text.append(value_desc)
                else:
                    values_text.append(str(value))
            content_parts.append(f"Values: {'; '.join(values_text)}")
        
        return '\n'.join(content_parts)
    
    async def cleanup(self):
        """Cleanup connections."""
        try:
            if self.mongodb_service and hasattr(self.mongodb_service, 'client'):
                if self.mongodb_service.client:
                    self.mongodb_service.client.close()
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")


async def main():
    """Main function to load all sample data."""
    loader = DataLoader()
    
    try:
        # Initialize services
        logger.info("🔄 Initializing services...")
        if not await loader.initialize():
            logger.error("❌ Failed to initialize services")
            return False
        
        # Load data in sequence
        logger.info("🔄 Loading business domains...")
        domains_count = await loader.load_business_domains()
        
        logger.info("🔄 Loading view metadata...")
        views_count = await loader.load_view_metadata()
        
        logger.info("🔄 Loading reports...")
        reports_count = await loader.load_reports()
        
        logger.info("🔄 Loading lookups...")
        lookups_count = await loader.load_lookups()
        
        total_docs = domains_count + views_count + reports_count + lookups_count
        
        if total_docs > 0:
            logger.info("🔄 Creating embeddings...")
            embeddings_count = await loader.create_embeddings()
            
            logger.info("=" * 60)
            logger.info("🎉 DATA LOADING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"📊 SUMMARY:")
            logger.info(f"   • Business Domains: {domains_count}")
            logger.info(f"   • Views: {views_count}")
            logger.info(f"   • Reports: {reports_count}")
            logger.info(f"   • Lookups: {lookups_count}")
            logger.info(f"   • Total Documents: {total_docs}")
            logger.info(f"   • Embeddings Created: {embeddings_count}")
            logger.info("=" * 60)
        else:
            logger.error("❌ No documents were loaded!")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await loader.cleanup()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)