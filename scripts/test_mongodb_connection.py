#!/usr/bin/env python3
"""
Test MongoDB connection to verify settings are correct.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from text_to_sql_rag.config.settings import settings
from pymongo import MongoClient

def test_mongodb_connection():
    """Test MongoDB connection."""
    print("🔍 Testing MongoDB Connection")
    print("=" * 40)
    
    # Print current settings
    print(f"MongoDB URL: {settings.mongodb.url}")
    print(f"MongoDB Database: {settings.mongodb.database}")
    print()
    
    try:
        print("Attempting to connect...")
        client = MongoClient(settings.mongodb.url)
        
        # Test connection
        print("Testing connection with ping...")
        client.admin.command('ismaster')
        print("✅ MongoDB connection successful!")
        
        # Test database access
        db = client[settings.mongodb.database]
        print(f"✅ Database '{settings.mongodb.database}' accessible")
        
        # List collections
        collections = db.list_collection_names()
        print(f"📋 Existing collections: {collections}")
        
        client.close()
        print("\n🎉 MongoDB connection test passed!")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("\nPossible solutions:")
        print("1. Check if MongoDB is running")
        print("2. Verify MONGODB_URL in your .env file")
        print("3. Check if the database name is correct")
        print("4. Ensure MongoDB is accessible from your network")
        return False

if __name__ == "__main__":
    test_mongodb_connection()