"""
Entry point for running fractal_48 as a module.

Usage:
    python -m fractal_48.cli [command] [options]
"""

from .cli import cli

if __name__ == '__main__':
    cli()