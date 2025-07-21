"""Shared database configuration and base classes."""

from sqlalchemy.ext.declarative import declarative_base

# Shared SQLAlchemy base for all database models
Base = declarative_base()