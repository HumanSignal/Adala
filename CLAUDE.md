# Adala Development Guidelines

## Build & Test Commands
- Install dependencies: `poetry install --with dev`
- Enter environment: `poetry shell`
- Run all tests: `pytest`
- Run specific test: `pytest tests/test_file.py::test_function_name`
- Run tests with recording: `pytest --record_mode=once --block-network`
- Run tests with network: `pytest -m "use_openai or use_azure"`
- Build docs: `mkdocs serve -f ./docs/mkdocs.yml`

## Code Style
- Use Python type hints throughout the codebase
- Follow PEP 8 naming conventions: snake_case for variables/functions, CamelCase for classes
- Prefer composition over inheritance when extending framework components
- When defining new skills, inherit from appropriate base classes
- Use f-strings for string formatting
- Docstrings should follow Google style format
- Exception handling should use specific exception types from utils.exceptions
- Test coverage required for all new features

## Architecture
- Agent: Main entry point - connects Skills, Environments and Runtimes
- Skills: Core capabilities (classification, extraction, etc.)
- Environments: Data sources and interaction points
- Runtimes: LLM backends (OpenAI, etc.)
- Utils: Shared functionality across components