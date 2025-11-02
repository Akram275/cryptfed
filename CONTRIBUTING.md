# Contributing to CrypTFed

We love contributions! CrypTFed is an open-source project and we welcome contributions from the community.

## **How to Contribute**

### 🐛 **Reporting Bugs**
- Use the [GitHub Issues](https://github.com/cryptfed/cryptfed/issues) tracker
- Include a clear title and description
- Provide steps to reproduce the issue
- Include your environment details (Python version, OS, etc.)

### **Suggesting Features**
- Open an issue with the "enhancement" label
- Describe the feature and its benefits
- Explain why this feature would be useful

### **Contributing Code**

#### Development Setup
```bash
# Fork the repository on GitHub
git clone https://github.com/your-username/cryptfed.git
cd cryptfed

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]
```

#### Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Format code with `black`: `black cryptfed/`
- Check with `flake8`: `flake8 cryptfed/`

#### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cryptfed

# Run specific test file
pytest tests/test_aggregators.py
```

#### Pull Request Process
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### 📚 **Contributing Documentation**
- Documentation is in the `docs/` directory
- Use clear, concise language
- Include code examples where appropriate
- Test all code examples

### **Contributing Examples**
- Add examples to the appropriate level directory:
  - `examples/level_1_basic/` - Simple, educational examples
  - `examples/level_2_intermediate/` - Feature demonstrations
  - `examples/level_3_advanced/` - Research and complex scenarios
- Include clear comments and documentation
- Test examples on clean environments

## **Development Guidelines**

### Code Organization
```
cryptfed/
├── core/           # Core federated learning components
├── aggregators/    # Aggregation algorithms
├── fhe/           # Homomorphic encryption managers
└── __init__.py    # Main API exports
```

### Adding New Aggregators
1. Inherit from `BaseAggregator`
2. Implement required methods:
   - `requires_plaintext_updates` property
   - `aggregate()` method
3. Add to `aggregator_registry` in `aggregators/__init__.py`
4. Write comprehensive tests
5. Add example usage

### Adding New FHE Schemes
1. Inherit from `BaseFHEManager`
2. Implement required methods:
   - `generate_crypto_context_and_keys()`
   - `encrypt()` and `decrypt()`
3. Add to `fhe_manager_registry` in `fhe/__init__.py`
4. Write comprehensive tests
5. Add benchmarking support

### Commit Message Format
```
type(scope): short description

Longer description if needed.

- List any breaking changes
- Reference issues: Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## **Testing Guidelines**

### Test Categories
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Example Tests**: Verify all examples run successfully
- **Performance Tests**: Benchmark critical operations

### Writing Good Tests
```python
def test_aggregator_basic_functionality():
    """Test that aggregator works with simple inputs."""
    # Arrange
    updates = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    weights = [0.5, 0.5]
    aggregator = PlaintextFedAvg()
    
    # Act
    result = aggregator.aggregate(updates, weights)
    
    # Assert
    expected = np.array([2.5, 3.5, 4.5])
    np.testing.assert_array_almost_equal(result, expected)
```

## 🔒 **Security Considerations**

When contributing to cryptographic components:
- **Never** hardcode keys or sensitive parameters
- **Always** use secure random number generation
- **Test** edge cases and error conditions
- **Document** security assumptions and limitations
- **Review** cryptographic implementations carefully

## **Performance Considerations**

- **Profile** your code for performance bottlenecks
- **Optimize** critical paths (encryption, aggregation)
- **Add** benchmarking for new features
- **Document** time and space complexity

## 🌍 **Community**

- Be respectful and inclusive
- Help others learn and contribute
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md)
- Join discussions in GitHub Discussions

## **Good First Issues**

Look for issues labeled `good-first-issue`:
- Documentation improvements
- Simple bug fixes
- Adding tests
- Code cleanup and refactoring

## ❓ **Questions?**

- **Documentation**: Check the [docs](docs/) directory
- **Examples**: Browse [examples](examples/) for usage patterns
- **Issues**: Search existing [GitHub Issues](https://github.com/cryptfed/cryptfed/issues)
- **Discussions**: Start a [GitHub Discussion](https://github.com/cryptfed/cryptfed/discussions)

Thank you for contributing to CrypTFed!