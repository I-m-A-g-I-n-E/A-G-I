# Third-Party Licenses

This directory contains license information for all third-party dependencies used in the A-G-I project.

## License Compatibility

All dependencies have been reviewed for compatibility with the A-G-I Source-Available Protective License (SAPL). The following licenses are used by our dependencies:

### Compatible Licenses

- **BSD 3-Clause License**: NumPy, PyTorch, Matplotlib, SciPy, Click, JupyterLab
- **BSD 2-Clause License**: Numba
- **Apache License 2.0**: tmtools
- **Python Software Foundation License**: Python Standard Library components

## Dependency Overview

| Package | Version | License | License File |
|---------|---------|---------|--------------|
| numpy | Latest | BSD-3-Clause | numpy-LICENSE.txt |
| torch | Latest | BSD-3-Clause | pytorch-LICENSE.txt |
| matplotlib | Latest | PSF-based | matplotlib-LICENSE.txt |
| scipy | Latest | BSD-3-Clause | scipy-LICENSE.txt |
| click | Latest | BSD-3-Clause | click-LICENSE.txt |
| jupyterlab | Latest | BSD-3-Clause | jupyterlab-LICENSE.txt |
| tmtools | Latest | Apache-2.0 | tmtools-LICENSE.txt |
| numba | Latest | BSD-2-Clause | numba-LICENSE.txt |

## License Compliance

### Attribution Requirements

Some dependencies require attribution in distributions:
- All BSD-licensed packages require copyright notice preservation
- Apache-licensed packages require notice file inclusion
- PSF-licensed components require Python license acknowledgment

### Distribution Requirements

When distributing the A-G-I project:
1. Include this LICENSES directory in distributions
2. Preserve all copyright notices and license texts
3. Include attribution requirements in NOTICES file
4. Comply with any additional license requirements

### Commercial Use Implications

For commercial licensing of A-G-I:
- Dependency licenses remain in effect
- Commercial licensees must comply with dependency license terms
- Some dependencies may require additional compliance measures
- Consult individual license files for specific requirements

## Updating Dependencies

When adding or updating dependencies:
1. Review license compatibility with SAPL
2. Add license files to this directory
3. Update this README with dependency information
4. Update NOTICES file with attribution requirements
5. Verify compliance with dependency license terms

## License Files

Individual license files for each dependency are stored in this directory using the naming convention: `[package-name]-LICENSE.txt`

For the most current license information, always refer to the official package repositories and documentation.

---

**Last Updated**: [Date]  
**Maintained By**: A-G-I Project Maintainers

For questions about license compliance, please create an issue with the "license" label.