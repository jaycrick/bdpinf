# Parameter Constraints in Birth-Death Process Fitting

The fitting module now supports flexible parameter constraints, allowing you to:
1.
**Fix parameters** to specific values (they won't be optimized)
2.
**Create equality groups** where multiple parameters must be equal

## Quick Start Examples

### Example 1: Fixed Parameters

Fix the death rate at epoch 0 to 0.1 while optimizing all other parameters:

```python
import jax.numpy as jnp

from bdpinf.inference.fitting import (
    ParameterConstraints,
    fit_birth_death_processes,
)

# Create constraints: fix death rate at index 2 (if n_epochs=2, indices 0-1 are birth, 2-3 are death)
constraints = ParameterConstraints(
    fixed_params={2: 0.1}  # death_rates[0] = 0.1
)

ages = jnp.array([10.0, 20.0, 30.0])
times = jnp.array([100.0])

result = fit_birth_death_processes(
    list_of_ages=[ages], times=times, param_constraints=constraints
)
```

### Example 2: Equality Constraints

Make the birth rates equal across the first two epochs:

```python
constraints = ParameterConstraints(
    equality_groups=[[0, 1]]  # birth_rates[0] == birth_rates[1]
)

result = fit_birth_death_processes(
    list_of_ages=[ages], times=times, param_constraints=constraints
)
```

### Example 3: Combined Constraints

Fix one parameter and create equality groups for others:

```python
# Assuming n_epochs = 3:
# Indices 0-2: birth_rates[0], birth_rates[1], birth_rates[2]
# Indices 3-5: death_rates[0], death_rates[1], death_rates[2]

constraints = ParameterConstraints(
    fixed_params={
        5: 0.05,  # death_rates[2] = 0.05
    },
    equality_groups=[
        [0, 1],  # birth_rates[0] == birth_rates[1]
        [3, 4],  # death_rates[0] == death_rates[1]
    ],
)

result = fit_birth_death_processes(
    list_of_ages=[epochs_data], times=times, param_constraints=constraints
)
```

## Parameter Indexing

When specifying constraints, parameter indices work as follows:

- **Birth rates**: indices `0` to `n_epochs - 1`
- **Death rates**: indices `n_epochs` to `2*n_epochs - 1`

For example, with 3 epochs:
```
Index 0: birth_rates[0]
Index 1: birth_rates[1]
Index 2: birth_rates[2]
Index 3: death_rates[0]  <- corresponds to death_rates at first epoch
Index 4: death_rates[1]
Index 5: death_rates[2]
```

## Use Cases

### Scenario 1: Time-Invariant Rates

If you believe birth and death rates are constant across time periods:

```python
n_epochs = 4
constraints = ParameterConstraints(
    equality_groups=[
        list(range(n_epochs)),  # all birth rates equal
        list(range(n_epochs, 2 * n_epochs)),  # all death rates equal
    ]
)
```

This effectively reduces a 4-epoch problem to 2 parameters (one birth, one death).

### Scenario 2: Partial Pooling

Group similar time periods but allow differences:

```python
# Period 1-2 have same rates, Period 3-4 have same rates
constraints = ParameterConstraints(
    equality_groups=[
        [0, 1],  # epochs 0-1 birth rates
        [2, 3],  # epochs 2-3 birth rates
        [4, 5],  # epochs 0-1 death rates
        [6, 7],  # epochs 2-3 death rates
    ]
)
```

### Scenario 3: Expert Prior Information

Incorporate domain knowledge by fixing rates based on external information:

```python
constraints = ParameterConstraints(
    fixed_params={
        0: 0.05,  # Expert says birth rate at epoch 0 is ~0.05
        3: 0.02,  # Expert says death rate at epoch 0 is ~0.02
    }
)
```

## Validation

The `ParameterConstraints` class validates your specifications:

```python
constraints = ParameterConstraints(...)
try:
    constraints.validate(n_total_params=6)  # n_epochs = 3
except ValueError as e:
    print(f"Invalid constraints: {e}")
```

Invalid constraints include:
- Index out of bounds
- Overlapping fixed and equality groups
- Equality groups with fewer than 2 elements

## Integration with Other Methods

Constraints work seamlessly with other `fit_birth_death_processes` parameters:

```python
result = fit_birth_death_processes(
    list_of_ages=[ages],
    times=times,
    initial_birth_rates=initial_br,
    initial_death_rates=initial_dr,
    optimizers=["LBFGS", "Adam"],
    param_constraints=constraints,
    random_seed=42,
)
```

## Notes

- Constraints reduce the effective parameter space during optimization, potentially improving convergence
- Fixed parameters are not optimized but are included in likelihood calculations
- Equal parameters share a single fitted value in the reduced optimization space
- The final result always returns parameters in the full parameter space
