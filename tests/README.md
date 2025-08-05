# Tests for inf-bp-texts

This directory contains comprehensive test suites for the `inf_bp_texts.inference` module.

## Test Organization

### Core Test Files

- **`test_birth_death_process.py`** - Tests for basic `BirthDeathProcess` class functionality
  - Initialization and parameter setting
  - Manuscript addition and age calculation
  - Process summaries and basic methods
  - Type consistency and attribute management

- **`test_log_likelihood.py`** - Tests for log-likelihood calculations
  - Self-attribute usage (`self.ages`, `self.birth_rate`, `self.death_rate`)
  - Parameter validation and edge cases
  - Numerical stability and critical cases
  - Negative log-likelihood for optimization

- **`test_fitting.py`** - Tests for parameter estimation and model fitting
  - Internal data fitting (`fit_from_self_data()`)
  - External DataFrame fitting (`fit()`)
  - Optimization methods and robustness
  - Fit result structure and validation

- **`test_simulation.py`** - Tests for birth-death process simulation
  - Simulation reproducibility and randomness
  - DataFrame structure and data validation
  - Age consistency and state management
  - Parameter effects on simulation outcomes

- **`test_utils.py`** - Tests for utility functions and edge cases
  - Data preparation (`prepare_manuscript_data()`)
  - Model comparison (`compare_models()`)
  - Prediction methods
  - Edge cases and error handling

- **`test_integration.py`** - Integration tests for complete workflows
  - End-to-end simulation and fitting
  - Real-world data processing workflows
  - Model comparison workflows
  - Numerical robustness testing

## Running Tests

### Run All Tests
```bash
poetry run python -m pytest tests/ -v
```

### Run Specific Test Files
```bash
# Basic functionality
poetry run python -m pytest tests/test_birth_death_process.py -v

# Log-likelihood calculations
poetry run python -m pytest tests/test_log_likelihood.py -v

# Fitting functionality
poetry run python -m pytest tests/test_fitting.py -v

# Simulation functionality
poetry run python -m pytest tests/test_simulation.py -v

# Utility functions
poetry run python -m pytest tests/test_utils.py -v

# Integration tests
poetry run python -m pytest tests/test_integration.py -v
```

### Run Tests with Coverage
```bash
poetry run python -m pytest tests/ --cov=inf_bp_texts --cov-report=html
```

### Run Tests in Parallel
```bash
poetry run python -m pytest tests/ -n auto
```

## Test Coverage

The test suite covers:

### ✅ Core Functionality
- Class initialization and basic methods
- Manuscript addition and age calculation
- Process state management
- Parameter validation

### ✅ Mathematical Operations
- Log-likelihood calculations with self attributes
- Parameter separation (birth_rate, death_rate)
- Count inference from ages
- Numerical stability and edge cases

### ✅ Parameter Estimation
- Internal data fitting (using `self.ages`)
- External DataFrame fitting
- Multiple optimization methods
- Fit result validation and structure

### ✅ Simulation
- Birth-death process simulation
- Data structure validation
- Reproducibility and randomness
- State consistency

### ✅ Utility Functions
- Data preparation for analysis
- Model comparison workflows
- Prediction methods
- Helper function validation

### ✅ Integration Workflows
- Complete analysis pipelines
- Real-world data processing
- Model selection and comparison
- Robustness testing

## Expected Warnings

Some tests may produce expected warnings:

- **Runtime Warnings** from scipy optimization (numerical precision issues)
- **Optimization Warnings** for unknown solver options (harmless)
- **User Warnings** for using unfitted models in predictions

These warnings are normal and indicate the code is handling edge cases appropriately.

## Test Philosophy

### Robustness Over Strict Success
Many tests check for valid result structures rather than strict success, since:
- Optimization can legitimately fail with certain data
- Numerical methods have inherent limitations
- Real-world data often presents challenging cases

### Comprehensive Edge Case Testing
Tests include scenarios like:
- Empty data sets
- Extreme parameter values
- Numerical boundary conditions
- Invalid inputs and error handling

### Integration Testing
Tests verify that components work together correctly in realistic workflows, not just in isolation.

## Adding New Tests

When adding new functionality to `inference.py`, please add corresponding tests:

1. **Unit tests** for individual methods
2. **Integration tests** for workflows
3. **Edge case tests** for robustness
4. **Performance tests** for large datasets (if applicable)

Follow the existing test patterns and naming conventions for consistency.

## Debugging Test Failures

If tests fail unexpectedly:

1. **Check warnings** - they often indicate the source of issues
2. **Run individual tests** - isolate the problem
3. **Add debug prints** - understand the data flow
4. **Check numerical stability** - extreme values can cause legitimate failures

The test suite is designed to be informative about what's working and what's not, rather than being overly strict about success in cases where failure is mathematically reasonable.
