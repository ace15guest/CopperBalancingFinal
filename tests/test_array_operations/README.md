# Array Operations Tests

This directory contains tests for the array operations library, specifically focusing on blur filter functions.

## Test Files

### test_blur_filters.py

Comprehensive tests for blur functions (`blur_call`, `box_blur`, `blur_2d_gaussian`) that test against real data from Q1, Q2, Q3, Q4, and Global quadrant folders.

#### Test Coverage

The test suite covers:

1. **blur_call function**
   - Box blur on all quadrants (Q1, Q2, Q3, Q4, Global)
   - Gaussian blur on all quadrants
   - Invalid blur type handling
   - Invalid dimension handling

2. **box_blur function**
   - Default parameters on all quadrants
   - Various window sizes (3, 5, 7, 11)
   - Different boundary modes (nearest, reflect, mirror, wrap)
   - Invalid window size handling
   - Integer dtype preservation

3. **blur_2d_gaussian function**
   - Default parameters on all quadrants
   - Various sigma values (0.5, 1.0, 2.0, 3.0, 5.0)
   - Anisotropic blur (different sigma per axis)
   - Different boundary modes
   - Invalid sigma handling
   - Integer dtype preservation

4. **Comparison tests**
   - Consistency across quadrants
   - Difference between box and Gaussian blur

5. **Edge cases**
   - Uniform arrays
   - Small arrays
   - Large window sizes

## Running the Tests

### Run all blur filter tests:
```bash
pytest tests/test_array_operations/test_blur_filters.py -v
```

### Run specific test class:
```bash
pytest tests/test_array_operations/test_blur_filters.py::TestBlurCall -v
```

### Run specific test:
```bash
pytest tests/test_array_operations/test_blur_filters.py::TestBlurCall::test_blur_call_box_blur -v
```

### Run tests for a specific quadrant (using keyword filter):
```bash
pytest tests/test_array_operations/test_blur_filters.py -v -k "Q1"
```

### Run with coverage:
```bash
pytest tests/test_array_operations/test_blur_filters.py --cov=lib.array_operations.blur_filters --cov-report=html
```

### Run in parallel (faster):
```bash
pytest tests/test_array_operations/test_blur_filters.py -v -n auto
```

## Test Data

Tests use NPZ files from the following directories:
- `Assets/processed_pngs/Q1_dpi_*/*.npz`
- `Assets/processed_pngs/Q2_dpi_*/*.npz`
- `Assets/processed_pngs/Q3_dpi_*/*.npz`
- `Assets/processed_pngs/Q4_dpi_*/*.npz`
- `Assets/processed_pngs/Global_dpi_*/*.npz`

### Available DPI Settings:
- **Q1**: 50, 100, 200, 250, 300, 500, 700
- **Q2**: 50, 100, 200, 250, 300, 500, 700
- **Q3**: 50, 100, 200, 250, 300, 500, 700
- **Q4**: 100, 200, 250
- **Global**: 100, 200, 250, 300

## Requirements

Install test dependencies:
```bash
pip install pytest pytest-cov pytest-xdist
```

## Test Results

The tests verify that:
- Blur functions work correctly on all quadrants
- Output shapes match input shapes
- Output dtypes are preserved
- Blur actually modifies the data (not a no-op)
- Edge cases are handled properly
- Invalid inputs raise appropriate errors

## Notes

- Tests automatically skip if NPZ files are not found
- Type checking warnings from basedpyright are expected and don't affect test functionality
- Tests are parameterized to run on all available quadrant/DPI combinations
- Some tests use subsets of data to reduce execution time while maintaining coverage