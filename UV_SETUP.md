# UV Setup Guide

This document explains how to set up and use UV for faster dependency management.

## Installing UV

UV is a fast Python package installer and resolver written in Rust, providing 10-100x faster installation than pip.

### Linux/macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Alternative: Using pip

```bash
pip install uv
```

## Generating UV Lock File

Once UV is installed, generate the lock file for reproducible builds:

```bash
# Navigate to project root
cd aml-miner-template

# Generate lock file from pyproject.toml
uv lock
```

This creates `uv.lock` with pinned versions of all dependencies.

## Using UV for Development

### Install Dependencies

```bash
# Install project dependencies (creates/uses virtual environment automatically)
uv sync

# Install with development dependencies
uv sync --extra dev
```

### Install Package in Development Mode

```bash
# Install the package in editable mode
uv pip install -e .
```

### Add New Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name
```

## Benefits of UV

- **Speed**: 10-100x faster than pip
- **Reproducible**: Lock file ensures exact versions
- **Compatible**: Works with existing pyproject.toml and requirements.txt
- **Automatic venv**: Creates and manages virtual environments automatically
- **Better resolver**: More reliable dependency resolution

## Fallback to pip

UV is optional. The project still works with pip:

```bash
pip install -r requirements.txt
```

## Notes

- The `uv.lock` file should be committed to version control for reproducible builds
- UV reads `.python-version` file to use the correct Python version (3.13.5)
- UV is fully compatible with the existing pyproject.toml configuration