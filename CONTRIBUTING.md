# Contributing to Text-to-SQL RAG Application

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python 3.9+
- Poetry for dependency management
- Git
- Access to AWS Bedrock (for testing with actual services)
- Qdrant instance (local or cloud)

### Development Setup

1. **Fork and clone the repository:**
```bash
git clone https://github.com/yourusername/text-to-sql-rag.git
cd text-to-sql-rag
```

2. **Install dependencies:**
```bash
poetry install
```

3. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your actual values
```

4. **Run tests:**
```bash
poetry run pytest
```

5. **Start the development server:**
```bash
poetry run uvicorn src.text_to_sql_rag.api.main:app --reload --port 8000
```

## Development Guidelines

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks:
```bash
poetry run black src/ tests/
poetry run isort src/ tests/
poetry run flake8 src/ tests/
poetry run mypy src/
```

### Commit Messages

Use conventional commits format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(api): add query validation endpoint
fix(rag): improve error handling in SQL generation
docs(readme): update installation instructions
```

### Branch Naming

- Feature branches: `feature/description`
- Bug fixes: `fix/description`
- Documentation: `docs/description`
- Refactoring: `refactor/description`

### Pull Request Process

1. **Create a feature branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes with tests**

3. **Run quality checks:**
```bash
poetry run pytest
poetry run black src/ tests/
poetry run isort src/ tests/
poetry run flake8 src/ tests/
poetry run mypy src/
```

4. **Commit your changes:**
```bash
git add .
git commit -m "feat(scope): your description"
```

5. **Push and create PR:**
```bash
git push origin feature/your-feature-name
```

## Project Structure

```
src/text_to_sql_rag/
├── api/                    # FastAPI endpoints
├── core/                   # Core business logic
├── services/              # External service integrations
├── models/                # Data models
├── config/                # Configuration management
└── utils/                 # Utility functions

tests/                     # Test files
├── unit/                  # Unit tests
├── integration/           # Integration tests
└── fixtures/              # Test fixtures

docs/                      # Documentation
data/                      # Data directories
```

## Types of Contributions

### Bug Reports

When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Error messages and stack traces

### Feature Requests

For feature requests, please provide:
- Clear description of the feature
- Use case and motivation
- Proposed implementation approach
- Any breaking changes

### Code Contributions

Areas where contributions are welcome:
- New retrieval strategies
- Additional LLM providers
- Improved query validation
- Better error handling
- Performance optimizations
- Documentation improvements
- Test coverage

### Documentation

- API documentation improvements
- Usage examples
- Tutorial content
- Architecture explanations

## Testing

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src

# Run specific test file
poetry run pytest tests/unit/test_rag_pipeline.py

# Run tests with verbose output
poetry run pytest -v
```

### Writing Tests

- Write unit tests for all new functions
- Include integration tests for API endpoints
- Use meaningful test names
- Include both positive and negative test cases
- Mock external services in unit tests

Example test structure:
```python
def test_generate_sql_query_success():
    """Test successful SQL query generation."""
    # Arrange
    pipeline = create_test_pipeline()
    query = "Show me all customers"
    
    # Act
    result = pipeline.generate_sql_query(query)
    
    # Assert
    assert result["sql_query"]
    assert result["confidence"] > 0.5
```

## Documentation

### API Documentation

- All endpoints must have docstrings
- Include request/response examples
- Document all parameters and return values

### Code Documentation

- Use type hints for all functions
- Write clear docstrings for classes and methods
- Include examples for complex functions

## Performance Guidelines

- Consider memory usage for large documents
- Optimize vector search queries
- Use async/await for I/O operations
- Monitor API response times

## Security Considerations

- Never commit secrets or API keys
- Validate all user inputs
- Use secure defaults
- Follow OWASP guidelines

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. Tag release after merge
5. Create GitHub release

## Questions and Support

- Open an issue for bugs or feature requests
- Start a discussion for general questions
- Join our community chat (if available)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

Thank you for contributing! 🎉