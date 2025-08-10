"""
Quick start script to get the system running.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and show progress."""
    print(f"\n🚀 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 40)
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, check=False)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed with exit code {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False

def main():
    print("🚀 QUICK START - Text-to-SQL RAG System")
    print("=" * 50)
    print("This script will set up and start your demo system")
    print()
    
    # Check if .env exists
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print("⚠️ Creating .env file from template...")
        try:
            env_example = Path(__file__).parent / ".env.example"
            if env_example.exists():
                import shutil
                shutil.copy(env_example, env_file)
                print("✅ .env file created from .env.example")
            else:
                print("❌ .env.example not found")
        except Exception as e:
            print(f"❌ Failed to create .env: {e}")
    
    # Step 1: Install dependencies
    print("\n📦 STEP 1: Installing dependencies")
    if not run_command(["poetry", "install"], "Installing Python dependencies"):
        print("\n❌ Failed to install dependencies. Please run 'poetry install' manually.")
        return False
    
    # Step 2: Start services
    print("\n🐳 STEP 2: Starting Docker services")
    if not run_command(["docker", "compose", "up", "-d"], "Starting MongoDB, OpenSearch, and Redis"):
        print("\n❌ Failed to start services. Please run 'make up' manually.")
        return False
    
    print("\n⏳ Waiting for services to start...")
    time.sleep(15)
    
    # Step 3: Seed data
    print("\n📊 STEP 3: Seeding mock data")
    if not run_command(["poetry", "run", "python", "scripts/seed_mock_data.py"], "Seeding mock metadata"):
        print("⚠️ Seeding failed - you can run 'make seed' manually later")
    
    # Step 4: Build search index
    print("\n🔍 STEP 4: Building search index")
    if not run_command(["poetry", "run", "python", "scripts/reindex_metadata.py"], "Building search index"):
        print("⚠️ Indexing failed - you can run 'make reindex' manually later")
    
    # Step 5: Validate system
    print("\n🧪 STEP 5: Validating system")
    run_command(["poetry", "run", "python", "validate_system.py"], "System validation")
    
    # Final instructions
    print("\n" + "=" * 50)
    print("🎉 QUICK START COMPLETE!")
    print("=" * 50)
    
    print("\n🚀 Next steps:")
    print("1. Start the API server:")
    print("   python src/text_to_sql_rag/api/new_main.py")
    print()
    print("2. Open your browser to:")
    print("   • API Docs: http://localhost:8000/docs")
    print("   • Health Check: http://localhost:8000/health")
    print("   • OpenSearch Dashboards: http://localhost:5601")
    print()
    print("3. Run comprehensive tests:")
    print("   python run_comprehensive_tests.py")
    print()
    print("🔧 Available make commands:")
    print("   make up          - Start services")
    print("   make down        - Stop services")
    print("   make seed        - Seed mock data")
    print("   make reindex     - Rebuild search index")
    print("   make test-system - Run system tests")
    print("   make dev-setup   - Complete development setup")
    print()
    print("📁 Key directories:")
    print("   • meta_documents/ - Sample metadata")
    print("   • tests/ - Test suites")
    print("   • scripts/ - Utility scripts")
    print()
    print("🎯 Your system is ready for the CTO demo!")

if __name__ == "__main__":
    main()