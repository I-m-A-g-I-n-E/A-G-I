## Description

Brief description of changes and motivation.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)  
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Test improvements

## 48-Manifold Compliance

- [ ] All operations maintain 48-alignment
- [ ] Operations are reversible (no lossy transformations)
- [ ] Parity (keven/kodd) is preserved where applicable
- [ ] Mathematical integrity verified through tests

## Testing

- [ ] New tests added for new functionality
- [ ] All existing tests pass
- [ ] Round-trip reversibility verified (if applicable)
- [ ] Performance regression tests pass

```bash
# Test commands used
pytest tests/ -v
python -m fractal_48.tests  # If touching fractal_48 module
```

## Documentation

- [ ] Code includes appropriate docstrings
- [ ] Public APIs documented
- [ ] README updated (if needed)
- [ ] Examples provided for new features

## Legal and Licensing

- [ ] **CLA signed** (required for all contributors)
- [ ] No proprietary or copyrighted code included
- [ ] New dependencies (if any) are license-compatible
- [ ] Attribution preserved for any third-party code

## Code Quality

- [ ] Code follows project style guidelines
- [ ] No hardcoded constants (use Laws.* or named constants)
- [ ] Proper error handling with descriptive messages
- [ ] No TODO comments in production code

## Backwards Compatibility

- [ ] Changes are backwards compatible
- [ ] Existing APIs preserved (or deprecation planned)
- [ ] Migration guide provided (if breaking changes)

## Additional Context

Add any other context about the pull request here.

## Checklist for Maintainers

- [ ] CLA verification passed
- [ ] Code review completed  
- [ ] Tests pass in CI
- [ ] Documentation reviewed
- [ ] Security implications considered
- [ ] Performance impact evaluated

---

By submitting this pull request, I confirm that:
- I have read and agree to the [Contributor License Agreement (CLA)](https://github.com/I-m-A-g-I-n-E/A-G-I/blob/main/CLA.md)
- My contribution follows the [contribution guidelines](https://github.com/I-m-A-g-I-n-E/A-G-I/blob/main/CONTRIBUTING.md)
- I understand this project is licensed under the [A-G-I SAPL](https://github.com/I-m-A-g-I-n-E/A-G-I/blob/main/LICENSE)