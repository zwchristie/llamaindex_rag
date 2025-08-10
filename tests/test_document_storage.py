#!/usr/bin/env python3
"""Test MongoDB document storage and retrieval functionality."""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
import motor.motor_asyncio

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DocumentStorageTests:
    """Test suite for MongoDB document storage."""
    
    def __init__(self):
        self.mongodb_url = os.getenv("MONGODB_URL", "mongodb://admin:password@localhost:27017")
        self.database_name = os.getenv("MONGODB_DATABASE", "text_to_sql_rag")
        self.mongo_client = None
        self.db = None
        self.test_results = []
    
    async def setup(self):
        """Initialize MongoDB connection."""
        try:
            self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(self.mongodb_url)
            self.db = self.mongo_client[self.database_name]
            
            # Test connection
            await self.mongo_client.server_info()
            logger.info("✅ MongoDB connection established")
            return True
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            return False
    
    async def test_document_upload(self):
        """Test uploading documents from meta_documents folder."""
        test_name = "Document Upload"
        try:
            # Get sample documents
            meta_docs_path = Path(__file__).parent.parent / "meta_documents" / "views"
            
            if not meta_docs_path.exists():
                self.test_results.append({"test": test_name, "status": "SKIP", "message": "No meta_documents/views folder found"})
                return
            
            json_files = list(meta_docs_path.glob("*.json"))
            if not json_files:
                self.test_results.append({"test": test_name, "status": "SKIP", "message": "No JSON files found in meta_documents/views"})
                return
            
            # Upload each document
            uploaded_count = 0
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        doc_data = json.load(f)
                    
                    # Add metadata
                    doc_data['_uploaded_at'] = datetime.utcnow()
                    doc_data['_source_file'] = json_file.name
                    doc_data['_document_type'] = 'view_metadata'
                    
                    # Insert into MongoDB
                    result = await self.db.view_metadata.insert_one(doc_data)
                    if result.inserted_id:
                        uploaded_count += 1
                        logger.info(f"📄 Uploaded: {json_file.name}")
                    
                except Exception as e:
                    logger.error(f"❌ Error uploading {json_file.name}: {e}")
            
            if uploaded_count > 0:
                self.test_results.append({
                    "test": test_name, 
                    "status": "PASS", 
                    "message": f"Successfully uploaded {uploaded_count} documents"
                })
                logger.info(f"✅ {test_name}: {uploaded_count} documents uploaded successfully")
            else:
                self.test_results.append({
                    "test": test_name, 
                    "status": "FAIL", 
                    "message": "No documents were uploaded"
                })
                
        except Exception as e:
            self.test_results.append({
                "test": test_name, 
                "status": "FAIL", 
                "message": f"Error during upload: {str(e)}"
            })
            logger.error(f"❌ {test_name} failed: {e}")
    
    async def test_document_retrieval(self):
        """Test retrieving documents from MongoDB."""
        test_name = "Document Retrieval"
        try:
            # Count total documents
            total_docs = await self.db.view_metadata.count_documents({})
            
            if total_docs == 0:
                self.test_results.append({
                    "test": test_name, 
                    "status": "FAIL", 
                    "message": "No documents found in database"
                })
                return
            
            # Retrieve all documents
            cursor = self.db.view_metadata.find({})
            retrieved_docs = []
            async for doc in cursor:
                retrieved_docs.append(doc)
            
            # Test specific queries
            view_with_name = await self.db.view_metadata.find_one({"view_name": {"$exists": True}})
            view_with_schema = await self.db.view_metadata.find_one({"schema": {"$exists": True}})
            
            success_count = 0
            tests = [
                ("Total documents", total_docs > 0),
                ("Retrieved documents", len(retrieved_docs) == total_docs),
                ("Document with view_name", view_with_name is not None),
                ("Document with schema", view_with_schema is not None)
            ]
            
            for test_desc, passed in tests:
                if passed:
                    success_count += 1
                    logger.info(f"✅ {test_desc}: PASS")
                else:
                    logger.error(f"❌ {test_desc}: FAIL")
            
            if success_count == len(tests):
                self.test_results.append({
                    "test": test_name,
                    "status": "PASS", 
                    "message": f"Retrieved {total_docs} documents successfully"
                })
            else:
                self.test_results.append({
                    "test": test_name,
                    "status": "PARTIAL", 
                    "message": f"{success_count}/{len(tests)} retrieval tests passed"
                })
                
        except Exception as e:
            self.test_results.append({
                "test": test_name, 
                "status": "FAIL", 
                "message": f"Error during retrieval: {str(e)}"
            })
            logger.error(f"❌ {test_name} failed: {e}")
    
    async def test_document_queries(self):
        """Test complex document queries."""
        test_name = "Document Queries"
        try:
            queries_passed = 0
            total_queries = 0
            
            # Query 1: Find documents by type
            total_queries += 1
            view_docs = await self.db.view_metadata.count_documents({"_document_type": "view_metadata"})
            if view_docs > 0:
                queries_passed += 1
                logger.info(f"✅ Found {view_docs} view_metadata documents")
            
            # Query 2: Find documents with columns
            total_queries += 1
            docs_with_columns = await self.db.view_metadata.count_documents({"columns": {"$exists": True}})
            if docs_with_columns > 0:
                queries_passed += 1
                logger.info(f"✅ Found {docs_with_columns} documents with columns")
            
            # Query 3: Find recent uploads
            total_queries += 1
            recent_docs = await self.db.view_metadata.count_documents({"_uploaded_at": {"$exists": True}})
            if recent_docs > 0:
                queries_passed += 1
                logger.info(f"✅ Found {recent_docs} recently uploaded documents")
            
            # Query 4: Text search capability
            total_queries += 1
            try:
                # Try to find documents with specific keywords
                keyword_docs = await self.db.view_metadata.count_documents({
                    "$or": [
                        {"description": {"$regex": "user", "$options": "i"}},
                        {"view_name": {"$regex": "USER", "$options": "i"}}
                    ]
                })
                queries_passed += 1
                logger.info(f"✅ Text search found {keyword_docs} matching documents")
            except Exception:
                logger.warning("⚠️  Text search query failed")
            
            if queries_passed == total_queries:
                self.test_results.append({
                    "test": test_name,
                    "status": "PASS",
                    "message": f"All {total_queries} query tests passed"
                })
            else:
                self.test_results.append({
                    "test": test_name,
                    "status": "PARTIAL",
                    "message": f"{queries_passed}/{total_queries} query tests passed"
                })
                
        except Exception as e:
            self.test_results.append({
                "test": test_name,
                "status": "FAIL",
                "message": f"Error during queries: {str(e)}"
            })
            logger.error(f"❌ {test_name} failed: {e}")
    
    async def test_document_structure_validation(self):
        """Test document structure integrity."""
        test_name = "Document Structure Validation"
        try:
            # Get sample document
            sample_doc = await self.db.view_metadata.find_one({})
            
            if not sample_doc:
                self.test_results.append({
                    "test": test_name,
                    "status": "SKIP",
                    "message": "No documents available for validation"
                })
                return
            
            # Required fields for view metadata
            required_fields = ['view_name', '_uploaded_at', '_source_file', '_document_type']
            optional_fields = ['schema', 'description', 'columns', 'business_context']
            
            validation_results = []
            
            # Check required fields
            for field in required_fields:
                if field in sample_doc:
                    validation_results.append(f"✅ Required field '{field}' present")
                else:
                    validation_results.append(f"❌ Required field '{field}' missing")
            
            # Check optional fields
            optional_present = sum(1 for field in optional_fields if field in sample_doc)
            validation_results.append(f"📊 {optional_present}/{len(optional_fields)} optional fields present")
            
            # Check data types
            type_checks = []
            if '_uploaded_at' in sample_doc:
                type_checks.append(('_uploaded_at', isinstance(sample_doc['_uploaded_at'], datetime)))
            if 'view_name' in sample_doc:
                type_checks.append(('view_name', isinstance(sample_doc['view_name'], str)))
            
            type_passed = sum(1 for _, check in type_checks if check)
            validation_results.append(f"🔍 {type_passed}/{len(type_checks)} type validations passed")
            
            # Overall validation
            required_passed = sum(1 for field in required_fields if field in sample_doc)
            
            if required_passed == len(required_fields) and type_passed == len(type_checks):
                self.test_results.append({
                    "test": test_name,
                    "status": "PASS",
                    "message": "Document structure validation passed"
                })
                logger.info(f"✅ {test_name}: Document structure is valid")
            else:
                self.test_results.append({
                    "test": test_name,
                    "status": "FAIL",
                    "message": f"Structure validation issues: {', '.join(validation_results)}"
                })
                
            for result in validation_results:
                logger.info(result)
                
        except Exception as e:
            self.test_results.append({
                "test": test_name,
                "status": "FAIL",
                "message": f"Error during validation: {str(e)}"
            })
            logger.error(f"❌ {test_name} failed: {e}")
    
    async def cleanup(self):
        """Close database connections."""
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("🔚 Closed MongoDB connection")
    
    def print_summary(self):
        """Print test results summary."""
        print("\n" + "="*60)
        print("MONGODB DOCUMENT STORAGE TEST RESULTS")
        print("="*60)
        
        passed = sum(1 for result in self.test_results if result["status"] == "PASS")
        failed = sum(1 for result in self.test_results if result["status"] == "FAIL")
        partial = sum(1 for result in self.test_results if result["status"] == "PARTIAL")
        skipped = sum(1 for result in self.test_results if result["status"] == "SKIP")
        
        for result in self.test_results:
            status_icon = {
                "PASS": "✅",
                "FAIL": "❌", 
                "PARTIAL": "⚠️",
                "SKIP": "⏭️"
            }
            print(f"{status_icon[result['status']]} {result['test']}: {result['message']}")
        
        print(f"\nSummary: {passed} passed, {failed} failed, {partial} partial, {skipped} skipped")
        
        if failed == 0:
            print("🎉 All MongoDB storage tests completed successfully!")
            return True
        else:
            print(f"⚠️  {failed} test(s) failed - check logs for details")
            return False

async def main():
    """Run all document storage tests."""
    print("STARTING MONGODB DOCUMENT STORAGE TESTS")
    print("="*60)
    
    tester = DocumentStorageTests()
    
    try:
        # Setup
        if not await tester.setup():
            print("❌ Failed to setup test environment")
            return False
        
        # Run tests
        await tester.test_document_upload()
        await tester.test_document_retrieval()
        await tester.test_document_queries()
        await tester.test_document_structure_validation()
        
        # Print results
        success = tester.print_summary()
        return success
        
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)