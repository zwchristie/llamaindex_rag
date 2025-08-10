"""
Test runner script for the text-to-SQL system.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=False)
    success = result.returncode == 0
    
    if success:
        print(f"✅ {description} - PASSED")
    else:
        print(f"❌ {description} - FAILED")
    
    return success


def main():
    """Run all test suites."""
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🧪 Text-to-SQL RAG Test Suite")
    print(f"📁 Working directory: {project_root}")
    
    all_passed = True
    
    # Unit tests
    unit_result = run_command(
        ["python", "-m", "pytest", "tests/unit/", "-v", "--tb=short"],
        "Unit Tests"
    )
    all_passed &= unit_result
    
    # Integration tests (skip if services not available)
    integration_result = run_command(
        ["python", "-m", "pytest", "tests/integration/", "-v", "--tb=short", "-m", "integration"],
        "Integration Tests"
    )
    # Don't fail overall if integration tests are skipped due to missing services
    
    # Test coverage report
    coverage_result = run_command(
        ["python", "-m", "pytest", "tests/unit/", "--cov=src/text_to_sql_rag", "--cov-report=term-missing", "--cov-report=html"],
        "Coverage Report"
    )
    
    # Linting
    lint_result = True
    try:
        lint_result = run_command(
            ["python", "-m", "flake8", "src/", "tests/", "--max-line-length=100", "--ignore=E203,W503"],
            "Code Linting (flake8)"
        )
    except FileNotFoundError:
        print("⚠️  flake8 not available - skipping linting")
    
    # Type checking
    type_result = True
    try:
        type_result = run_command(
            ["python", "-m", "mypy", "src/text_to_sql_rag/", "--ignore-missing-imports"],
            "Type Checking (mypy)"
        )
    except FileNotFoundError:
        print("⚠️  mypy not available - skipping type checking")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)
    print(f"Unit Tests: {'✅ PASSED' if unit_result else '❌ FAILED'}")
    print(f"Integration Tests: {'✅ PASSED' if integration_result else '⚠️  SKIPPED/FAILED'}")
    print(f"Coverage Report: {'✅ GENERATED' if coverage_result else '❌ FAILED'}")
    print(f"Code Linting: {'✅ PASSED' if lint_result else '❌ FAILED'}")
    print(f"Type Checking: {'✅ PASSED' if type_result else '❌ FAILED'}")
    
    if all_passed and lint_result and type_result:
        print("\n🎉 All critical tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed - please review the output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())