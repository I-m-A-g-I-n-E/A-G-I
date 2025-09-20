# Contributing to A-G-I

Thank you for your interest in contributing to the A-G-I project! We welcome contributions that help advance the 48-manifold Universal Semantic Kernel (USK) and Harmonic Integrity Protocol research.

## 🚀 Quick Start for Contributors

1. **Read the Contributor License Agreement (CLA)**: All contributors must accept our [CLA](CLA.md) before contributions can be merged
2. **Understand the 48-Manifold Principles**: Review our [custom Copilot instructions](.github/copilot-instructions.md) and core documentation
3. **Follow the "Transfer, Not Transform" Philosophy**: Maintain reversible, on-grid operations
4. **Ensure 48-Alignment**: All dimensions, tiles, and windows must be multiples of 48

## 📋 Contribution Process

### Before You Start

1. **Check Existing Issues**: Look for existing issues or discussions related to your intended contribution
2. **Create an Issue**: For significant changes, create an issue first to discuss the approach
3. **Fork the Repository**: Create your own fork to work in
4. **Review Documentation**: Familiarize yourself with:
   - [README.md](README.md) - Project overview and examples
   - [bio/HANDOFF.md](bio/HANDOFF.md) - Technical principles and core requirements
   - [docs/](docs/) - Additional research documentation

### Making Your Contribution

1. **Create a Feature Branch**: Use a descriptive name like `feature/your-feature-name`
2. **Follow Code Standards**: 
   - Maintain 48-alignment in all operations
   - Use reversible operations only (no lossy transformations)
   - Include comprehensive tests for new functionality
   - Follow existing naming conventions and patterns
3. **Write Tests**: All new code should include appropriate tests
4. **Update Documentation**: Update relevant documentation for your changes
5. **Run the Test Suite**: Ensure all tests pass before submitting

```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest

# Run the full test suite
pytest -v

# Run specific test categories
pytest tests/test_fractal48.py -v  # Core 48-manifold tests
pytest tests/test_immunity_unit.py -v  # Immune system tests
```

### Submitting Your Contribution

1. **Create a Pull Request**: Submit a PR with a clear title and description
2. **CLA Signature**: Our CLA bot will prompt you to sign the [Contributor License Agreement](CLA.md)
3. **Address Review Feedback**: Work with maintainers to address any feedback
4. **Final Approval**: Once approved, your contribution will be merged

## 🔍 CLA (Contributor License Agreement) Process

### Why We Require a CLA

Our CLA ensures:
- Legal clarity for incorporating contributions
- Ability to relicense the project as needed (including commercial licensing)
- Protection for both contributors and the project
- Compliance with our source-available protective licensing model

### How to Sign the CLA

1. **Automatic Process**: When you submit your first PR, our CLA bot will automatically comment
2. **Electronic Signature**: Follow the bot's instructions to electronically sign the CLA
3. **One-Time Process**: Once signed, you won't need to sign again for future contributions
4. **Corporate Contributors**: Contact maintainers for a Corporate CLA if contributing on behalf of a company

### CLA Bot Commands

The CLA bot responds to these commands in PR comments:
- `@cla-bot check` - Check current CLA status
- `@cla-bot recheck` - Re-verify CLA signature

## 🧪 Types of Contributions We Welcome

### High-Priority Areas

- **48-Manifold Optimizations**: Improvements to core permutation and transformation algorithms
- **Biological Applications**: Enhancements to protein folding, immune system modeling
- **Musical Theory Integration**: Expanding harmonic composition capabilities
- **Nuclear Fusion Applications**: Applying 48-manifold to fusion plasma control
- **Performance Optimizations**: Maintaining exactness while improving speed
- **Test Coverage**: Expanding test coverage for edge cases and new features

### Code Quality Standards

- **Reversibility**: All operations must be exactly reversible (no lossy transforms)
- **48-Alignment**: Dimensions must be multiples of 48
- **Parity Preservation**: Maintain keven/kodd separation
- **Provenance Tracking**: Log all transformations for audit trails
- **Error Handling**: Robust error handling with clear messages

### Documentation Contributions

- **Code Examples**: Clear, working examples of API usage
- **Research Documentation**: Academic-quality documentation of algorithms and theory
- **User Guides**: Step-by-step guides for common use cases
- **API Documentation**: Comprehensive docstring coverage

## 🛠️ Development Environment Setup

### System Requirements

- Python 3.11+ (3.12 recommended)
- PyTorch with MPS/CUDA support (optional but recommended)
- At least 8GB RAM for complex protein folding tasks

### Installation

```bash
# Clone your fork
git clone https://github.com/your-username/A-G-I.git
cd A-G-I

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest  # For testing

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
pytest --version
```

### Running Examples

```bash
# Test core 48-manifold operations
python main.py

# Test biological pipeline
python agi.py compose --sequence "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG" --samples 4

# Run fractal visualization
python -m fractal_48.cli render --width 1920 --height 1152
```

## 🧮 Code Standards and Best Practices

### Core Principles

1. **Transfer, Not Transform**: Preserve information through reversible operations
2. **48-Manifold Integrity**: All operations respect the 48-basis manifold
3. **Measurement-First**: Validate through metrics, avoid unsupported claims
4. **Separation of Laws vs Policy**: Distinguish immutable laws from tunable parameters

### Coding Guidelines

```python
# ✅ Good: 48-aligned operations
def tile_48(image):
    assert image.shape[0] % 48 == 0, f"Height {image.shape[0]} must be 48-aligned"
    assert image.shape[1] % 48 == 0, f"Width {image.shape[1]} must be 48-aligned"
    return image.reshape(H//48, 48, W//48, 48)

# ❌ Bad: Lossy operations
def downsample(image):
    return F.max_pool2d(image, 2)  # Destroys information!

# ✅ Good: Reversible permutation
def space_to_depth_2(x):
    # Exact permutation that can be perfectly reversed
    return rearrange(x, 'b c (h h2) (w w2) -> b (c h2 w2) h w', h2=2, w2=2)

# ✅ Good: Provenance tracking
def apply_with_provenance(data, operation):
    result = operation(data)
    provenance = {"operation": operation.__name__, "timestamp": time.time()}
    return result, provenance
```

### Testing Requirements

All new functionality must include:
- **Unit tests** for individual functions
- **Integration tests** for end-to-end workflows
- **Property tests** for mathematical invariants
- **Round-trip tests** for reversible operations

```python
def test_reversible_operation():
    """Test that operation is perfectly reversible."""
    input_data = create_test_data()
    forward_result = forward_operation(input_data)
    reconstructed = inverse_operation(forward_result)
    assert np.allclose(input_data, reconstructed, rtol=1e-12)
```

## 🏛️ Governance and Review Process

### Maintainer Responsibilities

- **Code Review**: All contributions are reviewed by maintainers
- **Testing**: Automated tests must pass before merge
- **Documentation**: Documentation updates are reviewed for accuracy
- **Licensing**: Ensure all contributions comply with licensing requirements

### Review Criteria

- **Technical Correctness**: Code works as intended and follows best practices
- **48-Manifold Compliance**: Adheres to core mathematical principles
- **Test Coverage**: Adequate test coverage for new functionality
- **Documentation**: Clear documentation for public APIs
- **Backwards Compatibility**: Changes don't break existing functionality

### Merge Process

1. **Automated Checks**: CI/CD runs tests and checks
2. **CLA Verification**: CLA bot confirms contributor has signed CLA
3. **Code Review**: At least one maintainer review required
4. **Final Approval**: Maintainer approval before merge

## 🔒 Security and Responsible Disclosure

If you discover security vulnerabilities:

1. **Do NOT** create a public issue
2. **Contact** maintainers through private channels (see [SECURITY.md](SECURITY.md))
3. **Provide** detailed information about the vulnerability
4. **Wait** for coordination before public disclosure

## 📞 Getting Help

### Community Support

- **GitHub Issues**: For bugs, feature requests, and general questions
- **GitHub Discussions**: For broader discussions about the project
- **Documentation**: Check existing documentation first

### Contacting Maintainers

- **General Questions**: Create a GitHub issue with the "question" label
- **CLA Issues**: Use the "CLA" label on GitHub issues
- **Security Issues**: Follow the process in [SECURITY.md](SECURITY.md)
- **Commercial Licensing**: See contact information in [LICENSE](LICENSE)

## 📝 License and Legal

### Code License

All contributions are made under the [A-G-I Source-Available Protective License (SAPL)](LICENSE), which:
- Allows research and evaluation use
- Restricts commercial and competitive use
- Requires separate licensing for commercial applications
- Maintains strong intellectual property protection

### Contributor Rights

By contributing, you:
- Retain copyright ownership of your contributions
- Grant broad licensing rights to the project (via CLA)
- Allow relicensing under various terms as needed
- Enable commercial licensing and monetization

### Dependencies

All dependencies must be compatible with our licensing model. When adding new dependencies:
1. Check license compatibility
2. Update the [LICENSES/](LICENSES/) directory
3. Document any license obligations

---

## 🙏 Thank You

Your contributions help advance cutting-edge research in reversible computation, harmonic composition, and biological modeling. Every contribution, whether code, documentation, testing, or feedback, moves the project forward.

**Happy Contributing!** 🚀

---

For questions about this contribution guide, please create an issue with the "documentation" and "question" labels.