"""
Script to seed the database with mock view metadata for testing.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import motor.motor_asyncio
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from text_to_sql_rag.models.view_models import (
    ViewMetadata, ViewColumn, ViewJoin, 
    ReportMetadata, LookupMetadata
)
from text_to_sql_rag.services.view_service import ViewService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock view data with comprehensive metadata
MOCK_VIEWS = [
    {
        "view_name": "V_TRANCHE_SYNDICATES",
        "view_type": "CORE",
        "schema_name": "SYND",
        "description": "Core view providing tranche syndicate information including member details, participation amounts, and syndicate relationships",
        "use_cases": "Used for syndicate analysis, member participation reporting, and tranche composition queries",
        "columns": [
            {"name": "tranche_id", "type": "NUMBER", "notNull": True, "description": "Unique identifier for the tranche"},
            {"name": "syndicate_id", "type": "NUMBER", "notNull": True, "description": "Unique identifier for the syndicate"},
            {"name": "member_id", "type": "NUMBER", "notNull": True, "description": "Member identifier in the syndicate"},
            {"name": "participation_amount", "type": "NUMBER", "notNull": False, "description": "Amount of participation by this member"},
            {"name": "participation_percentage", "type": "NUMBER", "notNull": False, "description": "Percentage of participation (0-100)"},
            {"name": "member_name", "type": "VARCHAR2", "notNull": False, "description": "Name of the syndicate member"},
            {"name": "member_type", "type": "VARCHAR2", "notNull": False, "description": "Type of member (LEAD, PARTICIPANT, etc.)"},
            {"name": "join_date", "type": "DATE", "notNull": False, "description": "Date when member joined the syndicate"},
            {"name": "status", "type": "VARCHAR2", "notNull": False, "description": "Current status of the member"},
            {"name": "created_date", "type": "DATE", "notNull": True, "description": "Record creation date"}
        ],
        "joins": [
            {
                "table_name": "TRANCHES", 
                "join_type": "INNER", 
                "join_condition": "t.tranche_id = V_TRANCHE_SYNDICATES.tranche_id",
                "description": "Links to tranche master data"
            },
            {
                "table_name": "SYNDICATES", 
                "join_type": "INNER", 
                "join_condition": "s.syndicate_id = V_TRANCHE_SYNDICATES.syndicate_id",
                "description": "Links to syndicate master data"
            }
        ],
        "sample_sql": "SELECT tranche_id, syndicate_id, member_name, participation_amount FROM SYND.V_TRANCHE_SYNDICATES WHERE tranche_id = 12345",
        "example_query": "Show me all syndicate members for tranche 12345",
        "data_returned": "Tranche syndicate relationships with member participation details",
        "view_sql": "CREATE VIEW V_TRANCHE_SYNDICATES AS SELECT t.tranche_id, s.syndicate_id, sm.member_id, sm.participation_amount, sm.participation_percentage, m.member_name, m.member_type, sm.join_date, sm.status, sm.created_date FROM tranches t JOIN syndicate_members sm ON t.tranche_id = sm.tranche_id JOIN syndicates s ON sm.syndicate_id = s.syndicate_id JOIN members m ON sm.member_id = m.member_id"
    },
    {
        "view_name": "V_USER_METRICS",
        "view_type": "CORE",
        "schema_name": "ANALYTICS",
        "description": "Core analytics view providing comprehensive user engagement and activity metrics",
        "use_cases": "User behavior analysis, activity reporting, engagement tracking, and performance dashboards",
        "columns": [
            {"name": "user_id", "type": "NUMBER", "notNull": True, "description": "Unique user identifier"},
            {"name": "username", "type": "VARCHAR2", "notNull": False, "description": "User's login name"},
            {"name": "email", "type": "VARCHAR2", "notNull": False, "description": "User's email address"},
            {"name": "login_count", "type": "NUMBER", "notNull": False, "description": "Total number of logins"},
            {"name": "last_login_date", "type": "DATE", "notNull": False, "description": "Date of last login"},
            {"name": "total_session_time", "type": "NUMBER", "notNull": False, "description": "Total session time in minutes"},
            {"name": "page_views", "type": "NUMBER", "notNull": False, "description": "Total page views by user"},
            {"name": "documents_accessed", "type": "NUMBER", "notNull": False, "description": "Number of documents accessed"},
            {"name": "queries_executed", "type": "NUMBER", "notNull": False, "description": "Number of queries executed"},
            {"name": "avg_query_response_time", "type": "NUMBER", "notNull": False, "description": "Average query response time in seconds"},
            {"name": "registration_date", "type": "DATE", "notNull": True, "description": "User registration date"},
            {"name": "user_status", "type": "VARCHAR2", "notNull": True, "description": "Current user status (ACTIVE, INACTIVE, etc.)"}
        ],
        "joins": [
            {
                "table_name": "USERS", 
                "join_type": "INNER", 
                "join_condition": "u.user_id = V_USER_METRICS.user_id",
                "description": "Links to user master data"
            },
            {
                "table_name": "USER_SESSIONS", 
                "join_type": "LEFT", 
                "join_condition": "us.user_id = V_USER_METRICS.user_id",
                "description": "Aggregates session data"
            }
        ],
        "sample_sql": "SELECT user_id, username, login_count, last_login_date, total_session_time FROM ANALYTICS.V_USER_METRICS WHERE user_status = 'ACTIVE' ORDER BY login_count DESC",
        "example_query": "Show me the top 10 most active users this month",
        "data_returned": "User activity metrics and engagement statistics",
        "view_sql": "CREATE VIEW V_USER_METRICS AS SELECT u.user_id, u.username, u.email, COUNT(us.session_id) as login_count, MAX(us.login_date) as last_login_date, SUM(us.session_duration) as total_session_time, SUM(us.page_views) as page_views, COUNT(DISTINCT da.document_id) as documents_accessed, COUNT(qe.query_id) as queries_executed, AVG(qe.response_time) as avg_query_response_time, u.registration_date, u.status as user_status FROM users u LEFT JOIN user_sessions us ON u.user_id = us.user_id LEFT JOIN document_access da ON u.user_id = da.user_id LEFT JOIN query_executions qe ON u.user_id = qe.user_id GROUP BY u.user_id, u.username, u.email, u.registration_date, u.status"
    },
    {
        "view_name": "V_TRANSACTION_SUMMARY",
        "view_type": "CORE", 
        "schema_name": "FINANCE",
        "description": "Financial transaction summary view aggregating transaction data by various dimensions",
        "use_cases": "Financial reporting, transaction analysis, audit trails, and reconciliation processes",
        "columns": [
            {"name": "transaction_id", "type": "NUMBER", "notNull": True, "description": "Unique transaction identifier"},
            {"name": "transaction_date", "type": "DATE", "notNull": True, "description": "Date of the transaction"},
            {"name": "transaction_type", "type": "VARCHAR2", "notNull": True, "description": "Type of transaction (DEBIT, CREDIT, TRANSFER)"},
            {"name": "amount", "type": "NUMBER", "notNull": True, "description": "Transaction amount"},
            {"name": "currency", "type": "VARCHAR2", "notNull": True, "description": "Currency code (USD, EUR, GBP, etc.)"},
            {"name": "account_id", "type": "NUMBER", "notNull": True, "description": "Account identifier"},
            {"name": "account_name", "type": "VARCHAR2", "notNull": False, "description": "Account name or description"},
            {"name": "counterparty", "type": "VARCHAR2", "notNull": False, "description": "Transaction counterparty"},
            {"name": "description", "type": "VARCHAR2", "notNull": False, "description": "Transaction description"},
            {"name": "status", "type": "VARCHAR2", "notNull": True, "description": "Transaction status (PENDING, COMPLETED, FAILED)"},
            {"name": "created_by", "type": "VARCHAR2", "notNull": True, "description": "User who created the transaction"},
            {"name": "approved_by", "type": "VARCHAR2", "notNull": False, "description": "User who approved the transaction"}
        ],
        "joins": [
            {
                "table_name": "TRANSACTIONS", 
                "join_type": "INNER", 
                "join_condition": "t.transaction_id = V_TRANSACTION_SUMMARY.transaction_id",
                "description": "Core transaction data"
            },
            {
                "table_name": "ACCOUNTS", 
                "join_type": "INNER", 
                "join_condition": "a.account_id = V_TRANSACTION_SUMMARY.account_id",
                "description": "Account details"
            }
        ],
        "sample_sql": "SELECT transaction_date, transaction_type, amount, currency, account_name FROM FINANCE.V_TRANSACTION_SUMMARY WHERE transaction_date >= SYSDATE - 30 AND status = 'COMPLETED'",
        "example_query": "Show me all completed transactions in the last 30 days",
        "data_returned": "Transaction details with account and approval information",
        "view_sql": "CREATE VIEW V_TRANSACTION_SUMMARY AS SELECT t.transaction_id, t.transaction_date, t.transaction_type, t.amount, t.currency, t.account_id, a.account_name, t.counterparty, t.description, t.status, t.created_by, t.approved_by FROM transactions t INNER JOIN accounts a ON t.account_id = a.account_id"
    },
    {
        "view_name": "V_DOCUMENT_ACCESS_LOG",
        "view_type": "SUPPORTING",
        "schema_name": "AUDIT", 
        "description": "Audit view tracking document access patterns and user activity",
        "use_cases": "Security auditing, access pattern analysis, compliance reporting, and document usage tracking",
        "columns": [
            {"name": "access_id", "type": "NUMBER", "notNull": True, "description": "Unique access log identifier"},
            {"name": "document_id", "type": "NUMBER", "notNull": True, "description": "Document identifier"},
            {"name": "document_name", "type": "VARCHAR2", "notNull": False, "description": "Document name or title"},
            {"name": "user_id", "type": "NUMBER", "notNull": True, "description": "User who accessed the document"},
            {"name": "username", "type": "VARCHAR2", "notNull": False, "description": "Username of accessor"},
            {"name": "access_date", "type": "DATE", "notNull": True, "description": "Date and time of access"},
            {"name": "access_type", "type": "VARCHAR2", "notNull": True, "description": "Type of access (VIEW, DOWNLOAD, EDIT)"},
            {"name": "ip_address", "type": "VARCHAR2", "notNull": False, "description": "IP address of accessor"},
            {"name": "user_agent", "type": "VARCHAR2", "notNull": False, "description": "Browser/client user agent"},
            {"name": "session_id", "type": "VARCHAR2", "notNull": False, "description": "Session identifier"},
            {"name": "success", "type": "NUMBER", "notNull": True, "description": "Whether access was successful (1/0)"},
            {"name": "error_message", "type": "VARCHAR2", "notNull": False, "description": "Error message if access failed"}
        ],
        "joins": [
            {
                "table_name": "DOCUMENT_ACCESS", 
                "join_type": "INNER", 
                "join_condition": "da.access_id = V_DOCUMENT_ACCESS_LOG.access_id",
                "description": "Core access log data"
            },
            {
                "table_name": "DOCUMENTS", 
                "join_type": "LEFT", 
                "join_condition": "d.document_id = V_DOCUMENT_ACCESS_LOG.document_id",
                "description": "Document metadata"
            },
            {
                "table_name": "USERS", 
                "join_type": "LEFT", 
                "join_condition": "u.user_id = V_DOCUMENT_ACCESS_LOG.user_id",
                "description": "User information"
            }
        ],
        "sample_sql": "SELECT document_name, username, access_date, access_type FROM AUDIT.V_DOCUMENT_ACCESS_LOG WHERE access_date >= SYSDATE - 7 ORDER BY access_date DESC",
        "example_query": "Show me all document access in the last week",
        "data_returned": "Document access audit trail with user and session information",
        "view_sql": "CREATE VIEW V_DOCUMENT_ACCESS_LOG AS SELECT da.access_id, da.document_id, d.document_name, da.user_id, u.username, da.access_date, da.access_type, da.ip_address, da.user_agent, da.session_id, da.success, da.error_message FROM document_access da LEFT JOIN documents d ON da.document_id = d.document_id LEFT JOIN users u ON da.user_id = u.user_id"
    },
    {
        "view_name": "V_PORTFOLIO_PERFORMANCE", 
        "view_type": "CORE",
        "schema_name": "PORTFOLIO",
        "description": "Portfolio performance metrics view providing returns, risk metrics, and benchmark comparisons",
        "use_cases": "Investment performance analysis, risk reporting, portfolio optimization, and client reporting",
        "columns": [
            {"name": "portfolio_id", "type": "NUMBER", "notNull": True, "description": "Unique portfolio identifier"},
            {"name": "portfolio_name", "type": "VARCHAR2", "notNull": False, "description": "Portfolio name or description"},
            {"name": "as_of_date", "type": "DATE", "notNull": True, "description": "Performance calculation date"},
            {"name": "total_value", "type": "NUMBER", "notNull": False, "description": "Total portfolio value"},
            {"name": "daily_return", "type": "NUMBER", "notNull": False, "description": "Daily return percentage"},
            {"name": "mtd_return", "type": "NUMBER", "notNull": False, "description": "Month-to-date return percentage"},
            {"name": "ytd_return", "type": "NUMBER", "notNull": False, "description": "Year-to-date return percentage"},
            {"name": "annual_return", "type": "NUMBER", "notNull": False, "description": "Annualized return percentage"},
            {"name": "volatility", "type": "NUMBER", "notNull": False, "description": "Portfolio volatility (standard deviation)"},
            {"name": "sharpe_ratio", "type": "NUMBER", "notNull": False, "description": "Risk-adjusted Sharpe ratio"},
            {"name": "benchmark_return", "type": "NUMBER", "notNull": False, "description": "Benchmark return for comparison"},
            {"name": "alpha", "type": "NUMBER", "notNull": False, "description": "Alpha vs benchmark"},
            {"name": "beta", "type": "NUMBER", "notNull": False, "description": "Beta vs benchmark"}
        ],
        "joins": [
            {
                "table_name": "PORTFOLIOS", 
                "join_type": "INNER", 
                "join_condition": "p.portfolio_id = V_PORTFOLIO_PERFORMANCE.portfolio_id",
                "description": "Portfolio master data"
            },
            {
                "table_name": "PORTFOLIO_RETURNS", 
                "join_type": "INNER", 
                "join_condition": "pr.portfolio_id = V_PORTFOLIO_PERFORMANCE.portfolio_id",
                "description": "Historical return data"
            }
        ],
        "sample_sql": "SELECT portfolio_name, total_value, ytd_return, volatility, sharpe_ratio FROM PORTFOLIO.V_PORTFOLIO_PERFORMANCE WHERE as_of_date = (SELECT MAX(as_of_date) FROM PORTFOLIO.V_PORTFOLIO_PERFORMANCE)",
        "example_query": "Show me the latest performance metrics for all portfolios",
        "data_returned": "Portfolio performance metrics including returns and risk measures",
        "view_sql": "CREATE VIEW V_PORTFOLIO_PERFORMANCE AS SELECT p.portfolio_id, p.portfolio_name, pr.as_of_date, pr.total_value, pr.daily_return, pr.mtd_return, pr.ytd_return, pr.annual_return, pr.volatility, pr.sharpe_ratio, pr.benchmark_return, pr.alpha, pr.beta FROM portfolios p INNER JOIN portfolio_returns pr ON p.portfolio_id = pr.portfolio_id"
    }
]

# Mock reports data
MOCK_REPORTS = [
    {
        "report_name": "syndicate_participation_report",
        "description": "Report showing syndicate member participation across tranches",
        "sample_queries": [
            "Show me syndicate participation for all tranches this month",
            "Which syndicate members have the highest participation rates?",
            "Generate a syndicate composition report for tranche 12345"
        ],
        "expected_columns": ["tranche_id", "syndicate_id", "member_name", "participation_amount", "participation_percentage"],
        "use_cases": ["Syndicate analysis", "Member performance tracking", "Risk distribution analysis"]
    },
    {
        "report_name": "user_activity_dashboard",
        "description": "Dashboard report for user engagement and activity metrics", 
        "sample_queries": [
            "Show me the most active users this week",
            "Generate a user engagement report for the last quarter",
            "Which users have not logged in recently?"
        ],
        "expected_columns": ["username", "login_count", "total_session_time", "last_login_date", "queries_executed"],
        "use_cases": ["User engagement analysis", "System usage tracking", "Inactive user identification"]
    },
    {
        "report_name": "transaction_reconciliation_report",
        "description": "Financial reconciliation report for transaction processing",
        "sample_queries": [
            "Show me all failed transactions from yesterday", 
            "Generate a daily transaction summary report",
            "Which transactions are still pending approval?"
        ],
        "expected_columns": ["transaction_date", "transaction_type", "amount", "status", "created_by", "approved_by"],
        "use_cases": ["Financial reconciliation", "Audit compliance", "Transaction monitoring"]
    }
]

# Mock lookups data
MOCK_LOOKUPS = [
    {
        "lookup_name": "transaction_statuses",
        "description": "Valid transaction status codes and descriptions",
        "values": [
            {"code": "PENDING", "description": "Transaction awaiting approval"},
            {"code": "APPROVED", "description": "Transaction approved for processing"},
            {"code": "COMPLETED", "description": "Transaction successfully completed"},
            {"code": "FAILED", "description": "Transaction failed to process"},
            {"code": "CANCELLED", "description": "Transaction cancelled by user"},
            {"code": "REJECTED", "description": "Transaction rejected by approver"}
        ],
        "context": "Use when filtering transactions by status or explaining status codes"
    },
    {
        "lookup_name": "user_statuses", 
        "description": "Valid user status codes and descriptions",
        "values": [
            {"code": "ACTIVE", "description": "Active user account"},
            {"code": "INACTIVE", "description": "Temporarily inactive user account"},
            {"code": "SUSPENDED", "description": "Account suspended due to violations"},
            {"code": "LOCKED", "description": "Account locked due to security reasons"},
            {"code": "EXPIRED", "description": "Account expired and needs renewal"},
            {"code": "PENDING", "description": "New account pending activation"}
        ],
        "context": "Use when filtering users by status or explaining account states"
    },
    {
        "lookup_name": "currency_codes",
        "description": "Supported currency codes and names",
        "values": [
            {"code": "USD", "description": "United States Dollar"},
            {"code": "EUR", "description": "Euro"},
            {"code": "GBP", "description": "British Pound Sterling"},
            {"code": "JPY", "description": "Japanese Yen"},
            {"code": "CHF", "description": "Swiss Franc"},
            {"code": "CAD", "description": "Canadian Dollar"},
            {"code": "AUD", "description": "Australian Dollar"}
        ],
        "context": "Use when working with multi-currency transactions or conversions"
    }
]


async def seed_database():
    """Seed the database with mock data."""
    try:
        # Get MongoDB connection string from environment or use default
        mongodb_url = os.getenv("MONGODB_URL", "mongodb://admin:password@localhost:27017")
        database_name = os.getenv("MONGODB_DATABASE", "text_to_sql_rag")
        
        # Connect to MongoDB
        client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_url)
        db = client[database_name]
        
        logger.info(f"Connected to MongoDB: {database_name}")
        
        # Initialize services
        view_service = ViewService(db)
        
        # Create indexes
        await view_service.ensure_indexes()
        
        # Clear existing data
        logger.info("Clearing existing data...")
        await db.view_metadata.delete_many({})
        await db.report_metadata.delete_many({})
        await db.lookup_metadata.delete_many({})
        
        # Seed views
        logger.info(f"Seeding {len(MOCK_VIEWS)} view metadata documents...")
        for view_data in MOCK_VIEWS:
            # Convert to ViewMetadata model
            columns = [ViewColumn(**col) for col in view_data["columns"]]
            joins = [ViewJoin(**join) for join in view_data.get("joins", [])]
            
            view_metadata = ViewMetadata(
                view_name=view_data["view_name"],
                view_type=view_data["view_type"],
                schema_name=view_data.get("schema_name"),
                description=view_data["description"],
                use_cases=view_data["use_cases"],
                columns=columns,
                joins=joins,
                view_sql=view_data.get("view_sql"),
                sample_sql=view_data.get("sample_sql"),
                example_query=view_data.get("example_query"),
                data_returned=view_data.get("data_returned")
            )
            
            await view_service.create_view(view_metadata)
            logger.info(f"Seeded view: {view_metadata.view_name}")
        
        # Seed reports
        logger.info(f"Seeding {len(MOCK_REPORTS)} report metadata documents...")
        for report_data in MOCK_REPORTS:
            report = ReportMetadata(**report_data)
            await db.report_metadata.insert_one(report.dict())
            logger.info(f"Seeded report: {report.report_name}")
        
        # Seed lookups
        logger.info(f"Seeding {len(MOCK_LOOKUPS)} lookup metadata documents...")
        for lookup_data in MOCK_LOOKUPS:
            lookup = LookupMetadata(**lookup_data)
            await db.lookup_metadata.insert_one(lookup.dict())
            logger.info(f"Seeded lookup: {lookup.lookup_name}")
        
        # Print summary
        stats = await view_service.get_stats()
        logger.info("Database seeding completed!")
        logger.info(f"Total views: {stats.get('total_views', 0)}")
        logger.info(f"Core views: {stats.get('core_views', 0)}")
        logger.info(f"Supporting views: {stats.get('supporting_views', 0)}")
        
        # Close connection
        client.close()
        
    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(seed_database())