# Makefile for Text-to-SQL RAG Development

.PHONY: help up down clean seed test lint format docs

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

up: ## Start infrastructure services (MongoDB, OpenSearch, Redis)
	docker compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo "Infrastructure services started. Run application with 'python src/text_to_sql_rag/api/new_main.py'"

up-full: ## Start all services including the application
	docker compose -f docker-compose.full.yml up -d --build
	@echo "Waiting for services to be ready..."
	@sleep 15
	@echo "Full application stack started. API available at http://localhost:8000"

down: ## Stop all services
	docker compose down
	docker compose -f docker-compose.full.yml down

clean: ## Stop services and remove volumes
	docker compose down -v
	docker compose -f docker-compose.full.yml down -v
	docker system prune -f

seed: ## Seed the database with mock data
	@echo "Seeding database with mock metadata..."
	poetry run python scripts/seed_mock_data.py
	@echo "Mock data seeded successfully!"

reindex: ## Rebuild OpenSearch index from MongoDB
	@echo "Rebuilding OpenSearch index..."
	poetry run python scripts/reindex_metadata.py
	@echo "Index rebuilt successfully!"

test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-system: ## Run comprehensive system tests
	@echo "Running comprehensive system tests..."
	python tests/run_system_tests.py

test-setup: ## Setup and verify system for testing
	@echo "Setting up system for testing..."
	python scripts/setup_for_testing.py

test-all: test-unit test-integration test-system ## Run all test suites

lint: ## Run code linting
	flake8 src/ tests/
	mypy src/

format: ## Format code with black and isort
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

install: ## Install dependencies with poetry
	poetry install

dev-setup: up seed reindex ## Complete development setup
	@echo "Development environment ready!"
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env and configure"
	@echo "  2. Run: python src/text_to_sql_rag/api/new_main.py"
	@echo "  3. API will be at: http://localhost:8000"
	@echo "Services:"
	@echo "  - OpenSearch: http://localhost:9200"
	@echo "  - OpenSearch Dashboards: http://localhost:5601"
	@echo "  - MongoDB: localhost:27017"

docs: ## Generate API documentation
	@echo "API documentation available at: http://localhost:8000/docs"

logs: ## Show logs for all services
	docker compose logs -f

status: ## Show status of all services
	docker compose ps