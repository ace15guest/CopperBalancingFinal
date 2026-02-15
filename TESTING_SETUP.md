# Testing Setup Guide

This guide will help you set up your environment to run the blur filter tests.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Step 1: Create a Virtual Environment

A virtual environment isolates your project dependencies from your system Python installation.

### On Windows (PowerShell):
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get an execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### On Windows (Command Prompt):
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat
```

### On macOS/Linux:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

## Step 2: Install the Package and Dependencies

Once your virtual environment is activated (you should see `(venv)` in your prompt):

```bash
# Install the package in development mode with test dependencies
pip install -e ".[dev]"

# Or install just the test dependencies
pip install pytest pytest-cov pytest-xdist
```

## Step 3: Verify Installation

```bash
# Check that pytest is installed
pytest --version

# Check that the package is installed
python -c "from lib.array_operations.blur_filters import blur_call; print('Success!')"
```

## Step 4: Run the Tests

### Option 1: Using pytest directly
```bash
# Run all blur filter tests
pytest tests/test_array_operations/test_blur_filters.py -v

# Run tests for a specific quadrant
pytest tests/test_array_operations/test_blur_filters.py -v -k "Q1"

# Run with coverage report
pytest tests/test_array_operations/test_blur_filters.py --cov=lib.array_operations.blur_filters --cov-report=html
```

### Option 2: Using the interactive test runner
```bash
python tests/run_blur_tests.py
```

## Step 5: Deactivate Virtual Environment

When you're done testing:

```bash
deactivate
```

## Quick Start Script

For convenience, here's a complete setup script:

### Windows (PowerShell) - save as `setup_tests.ps1`:
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/test_array_operations/test_blur_filters.py -v

Write-Host "Setup complete! Virtual environment is active."
Write-Host "To deactivate, run: deactivate"
```

### macOS/Linux - save as `setup_tests.sh`:
```bash
#!/bin/bash

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/test_array_operations/test_blur_filters.py -v

echo "Setup complete! Virtual environment is active."
echo "To deactivate, run: deactivate"
```

## Troubleshooting

### Issue: "pytest: command not found"
**Solution:** Make sure your virtual environment is activated and pytest is installed:
```bash
pip install pytest
```

### Issue: "ModuleNotFoundError: No module named 'lib'"
**Solution:** Install the package in development mode:
```bash
pip install -e .
```

### Issue: "No module named 'scipy'" or similar
**Solution:** Install all dependencies:
```bash
pip install -e ".[dev]"
```

### Issue: Tests are skipped with "No valid array found"
**Solution:** Make sure you have the NPZ data files in `Assets/processed_pngs/`. The tests expect files like:
- `Assets/processed_pngs/Q1_dpi_300/Q1_dpi_300.npz`
- `Assets/processed_pngs/Q2_dpi_300/Q2_dpi_300.npz`
- etc.

## What Gets Installed

When you run `pip install -e ".[dev]"`, the following packages are installed:

**Core dependencies:**
- numpy (array operations)
- scipy (blur filters)
- Pillow (image loading)

**Development dependencies:**
- pytest (test framework)
- pytest-cov (coverage reporting)
- black (code formatting)
- flake8 (linting)
- mypy (type checking)

## Next Steps

After setup, see `tests/test_array_operations/README.md` for detailed information about the test suite and available test options.