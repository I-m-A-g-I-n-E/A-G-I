# Ensure project root is importable in tests
import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def sequence() -> str:
    """Provide a default protein sequence for tests.

    Many tests assume a valid amino-acid sequence string. We return a
    48-residue sequence (matches default system size) by repeating a
    standard 20-AA alphabet.
    """
    aa20 = "ACDEFGHIKLMNPQRSTVWY"
    # Repeat and trim to 48
    s = (aa20 * 3)[:48]
    return s
