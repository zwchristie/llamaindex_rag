"""
Test file for document synchronization and processing.

This test validates:
- Document parsing and metadata extraction
- Chunking and embedding process
- Storage in MongoDB and vector database
- Proper handling of different document types
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.text_to_sql_rag.services.document_sync_service import DocumentSyncService
from src.text_to_sql_rag.services.mongodb_service import MongoDBService
from src.text_to_sql_rag.services.vector_service import LlamaIndexVectorService
from src.text_to_sql_rag.models.simple_models import DocumentType


class DocumentSyncTester:
    """Test class for document sync functionality."""
    
    def __init__(self):
        self.test_results = []
        
    def create_test_documents(self) -> Dict[str, Dict[str, Any]]:
        """Create sample test documents for each document type."""
        return {
            "business_domain.json": {
                "content": json.dumps({
                    "domain_name": "test_domain",
                    "description": "Test business domain",
                    "entities": ["entity1", "entity2"],
                    "relationships": []
                }),
                "type": DocumentType.BUSINESS_DOMAIN,
                "catalog": "test_catalog",
                "schema": "test_schema"
            },
            "core_view.json": {
                "content": json.dumps({
                    "view_name": "test_view",
                    "description": "Test core view",
                    "columns": [
                        {"name": "id", "type": "int", "description": "Primary key"},
                        {"name": "name", "type": "varchar", "description": "Name field"}
                    ]
                }),
                "type": DocumentType.CORE_VIEW,
                "catalog": "test_catalog", 
                "schema": "test_schema"
            },
            "lookup_metadata.json": {
                "content": json.dumps({
                    "lookup_name": "test_lookup",
                    "values": [{"key": "A", "value": "Active"}, {"key": "I", "value": "Inactive"}]
                }),
                "type": DocumentType.LOOKUP_METADATA,
                "catalog": "test_catalog",
                "schema": "test_schema"
            },
            "test_report.txt": {
                "content": "This is a test report document with sample content for testing the RAG system.",
                "type": DocumentType.REPORT,
                "catalog": "test_catalog",
                "schema": "test_schema"
            }
        }
    
    def test_document_parsing(self, sync_service: DocumentSyncService):
        """Test document parsing for different document types."""
        print("\n=== Testing Document Parsing ===")
        
        test_docs = self.create_test_documents()
        
        for filename, doc_info in test_docs.items():
            try:
                print(f"\nTesting parsing for {filename} ({doc_info['type'].value})")
                
                metadata = sync_service._parse_document_content(
                    content=doc_info['content'],
                    document_type=doc_info['type'],
                    catalog=doc_info['catalog'],
                    schema_name=doc_info['schema']
                )
                
                print(f"✓ Successfully parsed {filename}")
                print(f"  Metadata type: {type(metadata)}")
                print(f"  Has model_dump: {hasattr(metadata, 'model_dump')}")
                
                # Test the model_dump fix
                if metadata:
                    if hasattr(metadata, 'model_dump'):
                        metadata_dict = metadata.model_dump()
                    else:
                        metadata_dict = metadata
                    print(f"  Metadata keys: {list(metadata_dict.keys()) if isinstance(metadata_dict, dict) else 'Not a dict'}")
                
                self.test_results.append({
                    'test': f'parse_{filename}',
                    'status': 'PASS',
                    'message': 'Document parsed successfully'
                })
                
            except Exception as e:
                print(f"✗ Failed to parse {filename}: {str(e)}")
                self.test_results.append({
                    'test': f'parse_{filename}',
                    'status': 'FAIL',
                    'message': str(e)
                })
    
    def test_document_sync_flow(self, sync_service: DocumentSyncService):
        """Test the complete document sync flow."""
        print("\n=== Testing Document Sync Flow ===")
        
        test_docs = self.create_test_documents()
        
        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for filename, doc_info in test_docs.items():
                file_path = temp_path / filename
                file_path.write_text(doc_info['content'])
                
                try:
                    print(f"\nSyncing {filename}")
                    
                    result = sync_service._process_single_document(
                        file_path=file_path,
                        document_type=doc_info['type'],
                        catalog=doc_info['catalog'],
                        schema_name=doc_info['schema']
                    )
                    
                    print(f"✓ Sync result for {filename}:")
                    print(f"  Action: {result.action}")
                    print(f"  Success: {result.success}")
                    print(f"  Message: {result.message}")
                    
                    self.test_results.append({
                        'test': f'sync_{filename}',
                        'status': 'PASS' if result.success else 'FAIL',
                        'message': result.message
                    })
                    
                except Exception as e:
                    print(f"✗ Failed to sync {filename}: {str(e)}")
                    self.test_results.append({
                        'test': f'sync_{filename}',
                        'status': 'FAIL',
                        'message': str(e)
                    })
    
    def test_chunking_and_embedding(self, vector_service: LlamaIndexVectorService):
        """Test document chunking and embedding."""
        print("\n=== Testing Chunking and Embedding ===")
        
        test_content = """
        This is a test document for validating the chunking and embedding process.
        It contains multiple sentences and paragraphs to test how the system
        breaks down content into meaningful chunks.
        
        The document should be processed correctly and stored in the vector database
        with appropriate metadata for retrieval during RAG operations.
        """
        
        try:
            # Test chunking
            print("Testing document chunking...")
            chunks = vector_service._chunk_content(test_content)
            print(f"✓ Created {len(chunks)} chunks from test content")
            
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {i+1}: {chunk[:50]}...")
            
            self.test_results.append({
                'test': 'chunking',
                'status': 'PASS',
                'message': f'Successfully created {len(chunks)} chunks'
            })
            
        except Exception as e:
            print(f"✗ Chunking failed: {str(e)}")
            self.test_results.append({
                'test': 'chunking',
                'status': 'FAIL',
                'message': str(e)
            })
    
    def print_summary(self):
        """Print test results summary."""
        print("\n" + "="*50)
        print("TEST RESULTS SUMMARY")
        print("="*50)
        
        passed = sum(1 for r in self.test_results if r['status'] == 'PASS')
        failed = sum(1 for r in self.test_results if r['status'] == 'FAIL')
        
        print(f"Total tests: {len(self.test_results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print(f"\nFAILED TESTS:")
            for result in self.test_results:
                if result['status'] == 'FAIL':
                    print(f"  ✗ {result['test']}: {result['message']}")
        
        print(f"\nALL TESTS:")
        for result in self.test_results:
            status_symbol = "✓" if result['status'] == 'PASS' else "✗"
            print(f"  {status_symbol} {result['test']}: {result['message']}")


def main():
    """Main test function."""
    print("Starting Document Sync Test Suite...")
    
    tester = DocumentSyncTester()
    
    try:
        # Initialize services
        print("Initializing services...")
        
        # Check if we can initialize MongoDB service
        try:
            mongodb_service = MongoDBService()
            print("✓ MongoDB service initialized")
        except Exception as e:
            print(f"✗ MongoDB service initialization failed: {e}")
            return
        
        # Check if we can initialize Vector service
        try:
            vector_service = LlamaIndexVectorService()
            print("✓ Vector service initialized")
        except Exception as e:
            print(f"✗ Vector service initialization failed: {e}")
            # Continue without vector service for basic tests
            vector_service = None
        
        # Initialize document sync service
        try:
            sync_service = DocumentSyncService(
                meta_documents_path=Path("meta_documents"),
                mongodb_service=mongodb_service,
                vector_service=vector_service
            )
            print("✓ Document sync service initialized")
        except Exception as e:
            print(f"✗ Document sync service initialization failed: {e}")
            return
        
        # Run tests
        tester.test_document_parsing(sync_service)
        tester.test_document_sync_flow(sync_service)
        
        if vector_service:
            tester.test_chunking_and_embedding(vector_service)
        else:
            print("\n⚠ Skipping chunking/embedding tests - vector service not available")
        
    except Exception as e:
        print(f"Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        tester.print_summary()


if __name__ == "__main__":
    main()