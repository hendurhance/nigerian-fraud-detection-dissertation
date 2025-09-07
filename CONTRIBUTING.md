# Contributing to NIBSS Fraud Detection System

Thank you for your interest in contributing to the NIBSS Fraud Detection System! This document provides guidelines for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Guidelines](#development-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to abide by respectful and professional standards.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of machine learning and fraud detection concepts
- Familiarity with Nigerian banking systems (helpful but not required)

### Setting Up Development Environment

1. **Fork the repository**
   ```bash
   git clone https://github.com/hendurhance/nigerian-fraud-detection-dissertation.git
   cd nigerian-fraud-detection-dissertation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Run tests to ensure setup works**
   ```bash
   pytest tests/
   ```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new functionality
- **Code Contributions**: Implement new features or fix bugs
- **Documentation**: Improve or add documentation
- **Data Science**: Enhance algorithms or add new analysis methods
- **Testing**: Add or improve test coverage
- **Performance**: Optimize existing code

### Areas of Interest

- **Algorithm Improvements**: Enhance fraud detection models
- **Feature Engineering**: Add new behavioral or temporal features
- **Business Logic**: Improve cost analysis for Nigerian banking
- **Data Generation**: Enhance the synthetic data generator
- **Visualization**: Create better charts and analysis visualizations
- **Performance**: Optimize code for larger datasets
- **Documentation**: Improve clarity and completeness

## Development Guidelines

### Branching Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Individual feature branches
- `bugfix/*`: Bug fix branches
- `hotfix/*`: Critical fixes for production

### Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   pytest tests/
   python -m pytest tests/ --cov=src/
   ```

4. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: add new fraud detection feature"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Guidelines

Use conventional commit format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Examples:
```
feat: add SHAP waterfall plots for model interpretability
fix: correct monthly fraud distribution calculation
docs: update dataset generator algorithm documentation
test: add unit tests for preprocessing pipeline
```

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages follow guidelines
- [ ] Branch is up to date with main

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (specify)

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Documentation
- [ ] Code is documented
- [ ] README updated if needed
- [ ] API docs updated if needed

## Additional Notes
Any additional context or considerations
```

### Review Process

1. **Automated Checks**: CI/CD will run tests and linting
2. **Code Review**: Maintainers will review for:
   - Code quality and style
   - Test coverage
   - Documentation completeness
   - Algorithm correctness
   - Performance considerations
3. **Feedback**: Address any requested changes
4. **Approval**: After approval, maintainers will merge

## Issue Guidelines

### Bug Reports

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Error messages or logs
- Minimal reproducible example

### Feature Requests

Include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation approach
- Any relevant examples or references

### Templates

Use issue templates when available for consistency.

## Style Guidelines

### Code Style

- **Formatter**: Black (line length: 88)
- **Import sorting**: isort
- **Linting**: flake8
- **Type hints**: Use where appropriate

### Python Guidelines

```python
# Good
def calculate_fraud_score(
    transaction: Dict[str, Any], 
    customer_history: pd.DataFrame
) -> float:
    """Calculate fraud risk score for a transaction.
    
    Args:
        transaction: Transaction details
        customer_history: Customer transaction history
        
    Returns:
        Fraud risk score between 0 and 1
    """
    # Implementation here
    pass

# Bad
def calc_score(tx, hist):
    # No docstring, unclear parameters
    pass
```

### Documentation Style

- **Docstrings**: Google style
- **Comments**: Clear and concise
- **Type hints**: Include for public APIs
- **Examples**: Include usage examples

### Data Science Guidelines

- **Reproducibility**: Use random seeds
- **Performance**: Profile code for large datasets
- **Visualization**: Clear, publication-ready plots
- **Statistical Rigor**: Include confidence intervals where appropriate

## Testing

### Test Structure

```
tests/
├── unit/                 # Unit tests
├── integration/         # Integration tests
├── data/               # Test data files
└── conftest.py         # Pytest configuration
```

### Writing Tests

```python
import pytest
import pandas as pd
from src.nibss_fraud_dataset_generator import NIBSSFraudDatasetGenerator

def test_fraud_rate_accuracy():
    """Test that generated dataset has correct fraud rate."""
    generator = NIBSSFraudDatasetGenerator(dataset_size=10000, seed=42)
    df = generator.generate_dataset()
    
    actual_fraud_rate = df['is_fraud'].mean()
    expected_fraud_rate = 0.003
    
    assert abs(actual_fraud_rate - expected_fraud_rate) < 0.001

def test_channel_distribution():
    """Test that channel distribution matches NIBSS patterns."""
    # Implementation here
    pass
```

### Test Coverage

- Aim for >80% test coverage
- Focus on critical paths and edge cases
- Include both positive and negative test cases

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/

# Run specific test file
pytest tests/test_generator.py

# Run with verbose output
pytest -v
```

## Performance Considerations

### Dataset Generation
- Optimize for memory usage with large datasets
- Use vectorized operations where possible
- Profile memory and CPU usage

### Machine Learning
- Consider computational complexity
- Test with various dataset sizes
- Optimize hyperparameter search

## Documentation Standards

### Code Documentation
- All public functions must have docstrings
- Include parameter types and descriptions
- Provide usage examples
- Document any assumptions or limitations

### Algorithm Documentation
- Explain the mathematical basis
- Include references to papers or sources
- Provide implementation rationale
- Document calibration methodology

## Questions or Need Help?

- **Documentation**: Check existing docs first
- **Issues**: Search existing issues
- **Discussion**: Use GitHub Discussions for questions
- **Contact**: Reach out to maintainers

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes for significant contributions
- Academic papers (if applicable)

Thank you for contributing to the NIBSS Fraud Detection System!