# Contributing to NeuralLive (adaptiveengineer)

Thank you for your interest in contributing to NeuralLive! This document provides guidelines and instructions for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Basic understanding of artificial life systems (helpful but not required)

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/V1B3hR/adaptiveengineer.git
   cd adaptiveengineer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/
   ```

## Development Workflow

### Code Quality Standards

This project uses automated code quality tools configured via pre-commit hooks:

- **black**: Code formatting (line length: 79)
- **isort**: Import sorting (black profile)
- **flake8**: Linting with docstrings and bugbear
- **bandit**: Security scanning
- **Pre-commit hooks**: Whitespace, file fixes, yaml/json validation

### Before Committing

Pre-commit hooks will run automatically on `git commit`. To run them manually:

```bash
pre-commit run --all-files
```

### Code Style

- Follow PEP 8 guidelines (enforced by flake8)
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions small and focused (cyclomatic complexity < 10)
- Use descriptive variable names

### Testing

- Write tests for all new functionality
- Maintain or improve test coverage (target: 90%)
- Run tests before submitting PR: `pytest tests/`
- Test files should be named `test_*.py`

### Documentation

- Update relevant documentation for code changes
- Add docstrings following Google style
- Update CHANGELOG.md for significant changes
- Create/update README files for new modules

## Project Structure

```
adaptiveengineer/
├── adaptiveengineer.py      # Main monolithic file (being refactored)
├── neuralive/               # New modular structure
│   ├── core/                # Core functionality
│   ├── utils/               # Utility functions
│   └── __init__.py          # Package initialization
├── core/                    # Existing core modules
├── plugins/                 # Plugin system
├── simulation/              # Simulation components
├── tests/                   # Test suite
├── docs/                    # Documentation
└── example/                 # Example scripts
```

## Contributing Process

### 1. Create an Issue

Before starting work, create or comment on an issue to:
- Describe the problem or feature
- Discuss approach with maintainers
- Avoid duplicate work

### 2. Fork and Branch

```bash
# Fork the repository on GitHub, then:
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 3. Make Changes

- Follow code quality standards
- Write/update tests
- Update documentation
- Keep commits atomic and well-described

### 4. Run Quality Checks

```bash
# Run tests
pytest tests/

# Run pre-commit hooks
pre-commit run --all-files

# Check security
bandit -r . -ll
```

### 5. Submit Pull Request

- Push to your fork
- Create PR against main branch
- Fill out PR template
- Link related issues
- Wait for review

### Pull Request Guidelines

- **Title**: Clear, descriptive summary
- **Description**: What, why, and how
- **Tests**: Include test results
- **Breaking Changes**: Clearly marked
- **Documentation**: Updated as needed

## Code Review Process

1. Automated checks must pass (CI/CD)
2. Maintainer review
3. Address feedback
4. Final approval
5. Merge (squash and merge for clean history)

## Refactoring Guidelines

The project is undergoing Phase 1 refactoring. When extracting modules:

1. **Extract incrementally**: One module at a time
2. **Maintain compatibility**: Existing imports must work
3. **Test after extraction**: Run full test suite
4. **Document changes**: Update relevant documentation
5. **Follow pattern**: See `neuralive/utils/memory_utils.py` as example

### Example Extraction Process

```python
# 1. Create new module in neuralive/
# neuralive/utils/new_module.py

def extracted_function(arg: str) -> int:
    """Docstring here."""
    pass

# 2. Update neuralive/utils/__init__.py
from .new_module import extracted_function
__all__ = [..., "extracted_function"]

# 3. Create tests
# tests/test_new_module.py

def test_extracted_function():
    assert extracted_function("test") == expected

# 4. Optionally update adaptiveengineer.py
# to use the new module (maintaining backward compatibility)
```

## Phase 1 Continuation Priorities

If contributing to Phase 1 refactoring, prioritize:

1. **High complexity functions** (complexity > 10)
2. **Utility functions** (low coupling, high cohesion)
3. **Well-tested code** (easier to verify correctness)
4. **Independent modules** (fewer dependencies)

## Communication

- **Issues**: Bug reports, feature requests
- **Pull Requests**: Code contributions
- **Discussions**: Questions, ideas, architecture discussions

## License

By contributing, you agree that your contributions will be licensed under the GNU General Public License v3.0.

## Questions?

If you have questions:
1. Check existing issues and discussions
2. Review documentation
3. Create a new issue with the "question" label

## Recognition

Contributors will be recognized in:
- CHANGELOG.md
- GitHub contributors page
- Release notes

Thank you for contributing to NeuralLive! 🌟

---

**Note**: This is an active research project. Expect architecture to evolve as we implement advanced features (consciousness, swarm intelligence, self-healing, etc.). Your contributions help build the future of artificial life systems!
