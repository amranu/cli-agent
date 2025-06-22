# Testing Guide

This directory contains the test suite for the cli-agent project.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_base_agent.py  # Core agent functionality
│   ├── test_input_handler.py # Terminal input handling
│   ├── test_slash_commands.py # Command system
│   └── test_builtin_tools.py # Tool definitions
├── integration/             # Integration tests for workflows
│   └── test_agent_workflows.py # End-to-end workflows
├── fixtures/                # Test data and mocks (future)
├── conftest.py             # Shared test fixtures
└── README.md               # This file
```

## Running Tests

### Install Test Dependencies

```bash
# Install test dependencies
pip install -e ".[test]"

# Or install from requirements file
pip install -r requirements-test.txt
```

### Run All Tests

```bash
# Using pytest directly
pytest tests/

# Using the test runner script
python run_tests.py all
```

### Run Specific Test Types

```bash
# Unit tests only
python run_tests.py unit
pytest tests/unit/ -m unit

# Integration tests only  
python run_tests.py integration
pytest tests/integration/ -m integration

# Specific test file
pytest tests/unit/test_base_agent.py

# Specific test method
pytest tests/unit/test_base_agent.py::TestBaseMCPAgent::test_init_main_agent
```

### Run Tests with Coverage

```bash
# With coverage report
pytest tests/ --cov=cli_agent --cov=. --cov-report=term-missing --cov-report=html

# Using test runner (coverage enabled by default)
python run_tests.py all
```

### Run Tests in Parallel

```bash
# Use pytest-xdist for parallel execution
pytest tests/ -n auto
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Test individual components in isolation
- Mock external dependencies
- Fast execution
- High coverage target (90%+)

### Integration Tests (`@pytest.mark.integration`)
- Test complete workflows end-to-end
- May use real file system operations
- Moderate execution time
- Focus on critical user journeys

### Slow Tests (`@pytest.mark.slow`)
- Performance tests, long-running operations
- Can be skipped in quick test runs
- Run with: `pytest -m "not slow"` to skip

## Test Configuration

### pytest.ini
- Test discovery settings
- Coverage configuration
- Marker definitions
- Asyncio mode settings

### conftest.py
- Shared fixtures for all tests
- Mock objects and test data
- Test environment setup

## Fixtures Available

- `temp_dir`: Temporary directory for file operations
- `sample_host_config`: Test configuration object
- `mock_tools`: Mock tool definitions
- `mock_base_agent`: Mock agent instance
- `sample_messages`: Sample conversation data
- `mock_input_handler`: Mock input handler
- `mock_subagent_manager`: Mock subagent manager

## Writing New Tests

### Unit Test Example

```python
import pytest
from cli_agent.core.base_agent import BaseMCPAgent

@pytest.mark.unit
class TestMyComponent:
    def test_basic_functionality(self, mock_base_agent):
        result = mock_base_agent.some_method()
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_async_functionality(self, mock_base_agent):
        result = await mock_base_agent.async_method()
        assert result == "expected"
```

### Integration Test Example

```python
import pytest

@pytest.mark.integration
class TestWorkflow:
    @pytest.mark.asyncio
    async def test_complete_workflow(self, mock_base_agent, temp_dir):
        # Test complete user workflow
        result = await mock_base_agent.process_user_request("test")
        assert "success" in result
```

## Coverage Targets

- **Overall**: 85% minimum
- **Core modules**: 90% minimum  
- **Critical paths**: 95% minimum

## Debugging Tests

```bash
# Run with verbose output
pytest tests/ -v

# Run with debugging output
pytest tests/ -s

# Run specific test with debugging
pytest tests/unit/test_base_agent.py::test_method -v -s

# Run with pdb on failure
pytest tests/ --pdb
```

## CI/CD Integration

Tests run automatically on:
- Pull requests
- Pushes to main/master
- Multiple Python versions (3.8-3.12)
- Multiple operating systems (Ubuntu, macOS, Windows)

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `cli_agent` package is installed in development mode
2. **Async test failures**: Make sure to use `@pytest.mark.asyncio` for async tests
3. **File permission errors**: Use `temp_dir` fixture for file operations
4. **Mock issues**: Check that mocks are properly configured in conftest.py

### Getting Help

- Check existing test patterns in the codebase
- Review fixture definitions in `conftest.py`
- Run tests with `-v` for verbose output
- Use `--pdb` to debug failing tests