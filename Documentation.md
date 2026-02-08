# asdm

## Documentation
`Version 0.7.2 — February 2026`

**Note:** We recommend referring to the source code at [src/asdm/asdm.py](src/asdm/asdm.py) for exact usage details. Please open an `issue` or contact the maintainer if you find a bug. Pull requests with fixes, patches, or new features are welcome.

---

## Table of Contents
- [Simulation Specifications](#simulation-specifications)
- [Creation of SD Models](#creation-of-sd-models)
- [Model Building Methods](#model-building-methods)
- [Model Modification Methods](#model-modification-methods)
- [Variable Documentation and Tags](#variable-documentation-and-tags)
- [Simulation Methods](#simulation-methods)
- [Result Management Methods](#result-management-methods)
- [Model Inspection Methods](#model-inspection-methods)
- [XMILE Export](#xmile-export)
- [Built-in Functions Reference](#built-in-functions-reference)
- [Arrays and Subscripts](#arrays-and-subscripts)
- [Conveyors](#conveyors)
- [Data Import](#data-import)

---

## Simulation Specifications

Every model has a `sim_specs` dictionary that controls simulation timing. When loading from XMILE, these are parsed automatically. When building a model from code, set them directly:

```python
model.sim_specs['initial_time'] = 0        # Start time
model.sim_specs['simulation_time'] = 100    # Duration (not end time)
model.sim_specs['dt'] = 0.25               # Time step
model.sim_specs['time_units'] = 'Weeks'    # Time unit label
```

**Default values** (when no XMILE file is loaded):
| Key | Default | Description |
|---|---|---|
| `initial_time` | `0` | Simulation start time |
| `current_time` | `0` | Current time (managed by the engine) |
| `dt` | `0.25` | Integration time step |
| `simulation_time` | `13` | Total simulation duration |
| `time_units` | `'Weeks'` | Label for time units |

**Note:** `simulation_time` is the *duration*, not the end time. The simulation runs from `initial_time` to `initial_time + simulation_time`.

---

## Creation of SD Models
```
def __init__(self, from_xmile=None):
    """
    Initialises the sdmodel instance, optionally loading a model from an XMILE file.

    Parameters:
    - from_xmile (str, optional): The file path to an XMILE model file (.stmx or .xmile).
      If provided, the model's structure, equations, simulation specs, and data imports
      are parsed from the file.

    Example:
        model = sdmodel()                              # Empty model
        model = sdmodel(from_xmile='my_model.stmx')   # From XMILE file
    """
```

---

## Model Building Methods
```
def add_stock(self, name, equation, non_negative=True, is_conveyor=False, in_flows=[], out_flows=[]):
    """
    Adds a stock variable to the model.

    Parameters:
    - name (str): The name of the stock variable.
    - equation: The initial value equation (str, int, or float).
    - non_negative (bool, optional): Ensures the stock value cannot go negative.
      Defaults to True.
    - is_conveyor (bool, optional): If True, the stock acts as a conveyor with transit
      time. See the Conveyors section. Defaults to False.
    - in_flows (list, optional): Names of inflow variables to this stock.
    - out_flows (list, optional): Names of outflow variables from this stock.
    """
```
```
def add_flow(self, name, equation, leak=None, non_negative=False):
    """
    Adds a flow variable to the model.

    Parameters:
    - name (str): The name of the flow variable.
    - equation: The equation defining the flow's rate (str, int, or float).
    - leak (optional): If specified, this flow acts as a leakage flow from a conveyor.
      Defaults to None.
    - non_negative (bool, optional): Ensures the flow value cannot be negative.
      Defaults to False.
    """
```
```
def add_aux(self, name, equation):
    """
    Adds an auxiliary variable to the model.

    Parameters:
    - name (str): The name of the auxiliary variable.
    - equation: The equation defining the auxiliary's behaviour (str, int, or float).
    """
```
```
def add_delayed_aux(self, name, equation):
    """
    Adds a delayed auxiliary variable to the model. A delayed auxiliary is internally
    backed by an implicit stock but appears as an auxiliary in the model interface.

    Parameters:
    - name (str): The name of the delayed auxiliary variable.
    - equation: The equation defining its behaviour (str, int, or float).
    """
```

---

## Model Modification Methods
```
def replace_element_equation(self, name, new_equation, track_modification=True):
    """
    Replaces the equation of a specified model element (stock, flow, auxiliary, or
    delayed auxiliary) with a new equation.

    Parameters:
    - name (str): The name of the model element whose equation is to be replaced.
    - new_equation: The new equation. Can be a str, int, float, numpy numeric type,
      or a DataFeeder object.
    - track_modification (bool, optional): If True, the modification is tracked so
      that save_xmile() knows to update this variable. Defaults to True.

    Raises:
    - Exception: If the element name does not exist in the model.
    """
```
```
def overwrite_graph_function_points(self, name, new_xpts=None, new_xscale=None, new_ypts=None):
    """
    Overwrites the points or scale of a graph function (lookup table) associated
    with a model element.

    Parameters:
    - name (str): The name of the model element with a graph function.
    - new_xpts (list of float, optional): New x-axis data points.
    - new_xscale (tuple of float, optional): New x-axis scale as (min, max).
    - new_ypts (list of float, optional): New y-axis data points.

    Raises:
    - Exception: If all input parameters are None.
    """
```

---

## Variable Documentation and Tags

Variables can have attached documentation text and tags, which are preserved in XMILE files.

```
def get_variable_doc(self, var_name):
    """
    Get the documentation text for a variable.

    Parameters:
    - var_name (str): Variable name (Python format, underscores for spaces).

    Returns:
    - Documentation text string, or None if no documentation exists.
    """
```
```
def set_variable_doc(self, var_name, doc_text):
    """
    Set the documentation text for a variable.

    Parameters:
    - var_name (str): Variable name (Python format, underscores for spaces).
    - doc_text (str): Documentation text (plain text or HTML).

    Raises:
    - ValueError: If the variable does not exist in the model.
    """
```
```
def get_variable_tags(self, var_name):
    """
    Get the tags for a variable.

    Parameters:
    - var_name (str): Variable name (Python format, underscores for spaces).

    Returns:
    - List of tag strings, or empty list if no tags exist.
    """
```
```
def set_variable_tags(self, var_name, tags):
    """
    Set the tags for a variable.

    Parameters:
    - var_name (str): Variable name (Python format, underscores for spaces).
    - tags (list of str): Tag strings (e.g., ['data', 'need references']).

    Raises:
    - ValueError: If the variable does not exist in the model.
    """
```

---

## Simulation Methods
```
def simulate(self, time=None, dt=None, pause=False):
    """
    Runs the simulation of the model.

    Parameters:
    - time (float, optional): The simulation duration. If None, uses the duration
      from sim_specs['simulation_time'].
    - dt (float, optional): The time step. If None, uses sim_specs['dt'].
    - pause (bool, optional): If True, the simulation pauses after the specified time
      (or after one iteration if time is not specified), allowing parameter changes
      before resuming. Defaults to False.

    The simulation can be run multiple times to continue from where it left off,
    enabling interactive parameter changes mid-simulation:

        model.simulate(time=10, pause=True)
        model.replace_element_equation('birth_rate', '0.05')
        model.simulate(time=10)  # Continues from t=10 to t=20
    """
```
```
def clear_last_run(self):
    """
    Resets the simulation state, allowing the model to be re-run from the beginning.
    Clears all simulation results, resets current_time to initial_time, and returns
    the model to 'loaded' state.

    Use this when you want to start a fresh simulation after modifying the model.

        model.simulate()
        results_1 = model.export_simulation_result()
        model.clear_last_run()
        model.replace_element_equation('growth_rate', '0.1')
        model.simulate()
        results_2 = model.export_simulation_result()
    """
```
```
def initialize(self):
    """
    Initialises the model for simulation. This parses all equations (if not already
    parsed), builds dependency graphs, and calculates initial values for all variables.

    Called automatically by simulate(). You generally do not need to call this directly
    unless you want to inspect initial values before running a simulation.
    """
```

---

## Result Management Methods
```
def export_simulation_result(self, flatten=False, format='dict', to_csv=False):
    """
    Exports the results of the simulation in the specified format.

    Parameters:
    - flatten (bool, optional): Only relevant for models with arrays. Normally,
      arrayed variable results are stored as dictionaries {dimension: value}. If True,
      flattens the result so each dimension becomes a separate variable named
      'variable_dimension'. Defaults to False.
    - format (str, optional): Output format — 'dict' for a Python dictionary,
      'df' for a pandas DataFrame. Defaults to 'dict'.
    - to_csv (bool or str, optional): If True, exports results to 'asdm.csv'.
      If a string, uses it as the CSV filename. Defaults to False.

    Returns:
    - The simulation results in the specified format.
    """
```
```
def get_element_simulation_result(self, name, subscript=None):
    """
    Gets the simulation result time series for a specific variable.

    Parameters:
    - name (str): The variable name.
    - subscript (tuple, optional): For arrayed variables, the specific subscript
      to retrieve (e.g., ('young',) or ('region_1', 'age_group_2')).

    Returns:
    - A list of values over time for the specified variable (and subscript).
      For arrayed variables without a subscript, returns a dict of
      {subscript_tuple: [values]}.
    """
```
```
def display_results(self, variables=None):
    """
    Displays simulation results as a line plot using Matplotlib.

    Parameters:
    - variables (list or str, optional): Variable name(s) to display. If None or
      an empty list, displays results for all variables.
    """
```

---

## Model Inspection Methods
```
def summary(self):
    """
    Prints a summary of the model, including:
    - Simulation specs (initial_time, simulation_time, dt, time_units)
    - Runtime state (current_time, simulation progress)
    - Model state ('created', 'loaded', 'parsed', 'initialized', 'simulated', etc.)
    """
```
```
def is_dependent(self, var1, var2):
    """
    Checks if var2 depends directly on var1 — i.e., var1 appears in var2's equation.

    Parameters:
    - var1 (str): The potential dependency variable.
    - var2 (str): The variable to check.

    Returns:
    - True if var2 depends on var1, False otherwise.
    """
```
```
def generate_cld(self, vars=None, show=False, loop=True):
    """
    Generates a Causal Loop Diagram (CLD) from the model's dependency structure
    using NetworkX.

    Parameters:
    - vars (list, optional): Specific variable names to include in the CLD.
      If None, includes all variables.
    - show (bool, optional): If True, displays the CLD plot. Defaults to False.
    - loop (bool, optional): If True, includes feedback loops. Defaults to True.

    Returns:
    - A NetworkX DiGraph representing the causal loop structure.
    """
```
```
def generate_full_dependent_graph(self, show=False):
    """
    Generates the full dependency graphs used internally for simulation ordering.
    Produces two graphs: one for initialization and one for iteration.

    Parameters:
    - show (bool, optional): If True, displays the dependency graphs. Defaults to False.
    """
```

---

## XMILE Export
```
def save_xmile(self, filepath=None, _force_update_all=False):
    """
    Saves the model to XMILE format. Updates an existing XMILE file with any
    modifications made to the model, preserving the original structure including
    views and layout. Only modified variables are updated.

    The model must have been originally loaded from an XMILE file.

    Parameters:
    - filepath (str, optional): Path to save the file. If None, saves to the
      original file path with an '_asdm' suffix (e.g., 'model_asdm.stmx').

    Returns:
    - Path to the saved file.

    Raises:
    - RuntimeError: If the model was not loaded from an XMILE file.

    Example:
        model = sdmodel(from_xmile='my_model.stmx')
        model.replace_element_equation('growth_rate', '0.1')
        saved_path = model.save_xmile('modified_model.stmx')
    """
```

---

## Built-in Functions Reference

These functions can be used in equation strings passed to `add_stock`, `add_flow`, `add_aux`, and `replace_element_equation`. Function names are **case-insensitive** in equations.

### Time Functions
| Function | Syntax | Description |
|---|---|---|
| `INIT` | `INIT(variable)` | Returns the initial value of a variable |
| `STEP` | `STEP(height, start_time)` | Returns 0 before `start_time`, then `height` |
| `PULSE` | `PULSE(volume, first_pulse, interval)` | Generates pulses at specified intervals |
| `DELAY` | `DELAY(input, delay_time, initial)` | Material delay (pipeline delay) |
| `DELAY1` | `DELAY1(input, delay_time, initial)` | First-order exponential delay |
| `DELAY3` | `DELAY3(input, delay_time, initial)` | Third-order exponential delay |
| `SMTH1` | `SMTH1(input, averaging_time, initial)` | First-order exponential smooth |
| `SMTH3` | `SMTH3(input, averaging_time, initial)` | Third-order exponential smooth |
| `HISTORY` | `HISTORY(variable, time)` | Returns the value of a variable at a past time |

### Math Functions
| Function | Syntax | Description |
|---|---|---|
| `MIN` | `MIN(a, b)` | Minimum of two values |
| `MAX` | `MAX(a, b)` | Maximum of two values |
| `INT` | `INT(x)` | Integer part (truncation) |
| `EXP` | `EXP(x)` | Exponential (e^x) |
| `LOG10` | `LOG10(x)` | Base-10 logarithm |
| `MOD` | `MOD(a, b)` | Modulo (remainder) |
| `SAFEDIV` | `SAFEDIV(a, b)` | Division that returns 0 when dividing by 0 |

### Logical and Comparison Functions
| Function | Syntax | Description |
|---|---|---|
| `AND` | `a AND b` | Logical AND |
| `OR` | `a OR b` | Logical OR |
| `NOT` | `NOT(a)` | Logical NOT |
| `GT` | `GT(a, b)` | Greater than (a > b) |
| `LT` | `LT(a, b)` | Less than (a < b) |
| `NGT` | `NGT(a, b)` | No greater than (a <= b) |
| `NLT` | `NLT(a, b)` | No less than (a >= b) |
| `EQS` | `EQS(a, b)` | Equals (a == b) |

### Stochastic Functions
| Function | Syntax | Description |
|---|---|---|
| `NORMAL` | `NORMAL(mean, std_dev)` | Random draw from normal distribution |
| `BINOMIAL` | `BINOMIAL(n, p)` | Random draw from binomial distribution |

### Special Functions
| Function | Syntax | Description |
|---|---|---|
| `LOOKUP` | `LOOKUP(graph_func, x)` | Evaluates a graph function (lookup table) at x |
| `SUM` | `SUM(arrayed_variable)` | Sums all elements of an arrayed variable |
| `LOGISTICBOUND` | `LOGISTICBOUND(yfrom, yto, x, xmiddle, speed)` | S-curve transition between two values |
| `EXPBOUND` | `EXPBOUND(yfrom, yto, x, exponent, xstart, xfinish)` | Exponential transition between two values |

### Operators
| Operator | Syntax | Description |
|---|---|---|
| `+` | `a + b` | Addition |
| `-` | `a - b` | Subtraction |
| `*` | `a * b` | Multiplication |
| `/` | `a / b` | Division |
| `^` | `a ^ b` | Exponentiation |

---

## Arrays and Subscripts

ASDM supports multi-dimensional arrayed variables, allowing a single variable name to hold values across multiple dimensions (e.g., age groups, regions).

### From XMILE
When loading a model with arrays from XMILE, dimensions and subscripts are parsed automatically. Each arrayed variable's equation can be:
- **Parallel** — all elements share the same equation.
- **Element-by-element** — each element has its own equation.

### Accessing Arrayed Results
```python
# Export with arrays as nested dicts (default)
result = model.export_simulation_result(format='dict')
# result['Population'] -> {('young',): [...], ('old',): [...]}

# Export with arrays flattened to separate columns
result = model.export_simulation_result(format='df', flatten=True)
# DataFrame columns: 'Population_young', 'Population_old', ...
```

### Referencing Subscripts in Equations
In equation strings, subscripts are written in square brackets:
```
Population[young]
Population[region_1, age_group_2]
```

The `SUM` function aggregates across dimensions:
```
SUM(Population)
```

---

## Conveyors

Conveyors are a special type of stock that models material moving through a pipeline with a fixed transit time. Items enter the conveyor and exit after the specified delay.

### Creating Conveyors
```python
model.add_stock('pipeline', equation='100', is_conveyor=True, in_flows=['input'], out_flows=['output'])
```

### Leak Flows
Conveyors support leak flows — material that "leaks out" during transit (e.g., spoilage, attrition):
```python
model.add_flow('leakage', equation='0.05', leak=True)
```

When loading from XMILE, conveyor attributes (transit time, leak fractions) are parsed automatically.

---

## Data Import

ASDM supports importing data from CSV files into model variables. This is configured in XMILE files via `<data><import>` elements. Two import modes are available:

### Parameter Import
Overwrites variable initial values or constants with values from a CSV file. Useful for setting scenario-specific parameters.

### Time-Varying Import
Replaces a variable's equation with a `DataFeeder` that returns the appropriate value for each time step, with linear interpolation between data points.

### DataFeeder (Programmatic)
You can also create a `DataFeeder` directly in code:
```python
from asdm.asdm import DataFeeder

data_values = [10, 20, 30, 40, 50]
feeder = DataFeeder(data=data_values, from_time=0, data_dt=1, interpolate=True)
model.replace_element_equation('external_input', feeder)
```

**DataFeeder Parameters:**
| Parameter | Type | Description |
|---|---|---|
| `data` | list | A list of numeric values, one per time step |
| `from_time` | float | The start time for the data series (default: 0) |
| `data_dt` | float | The time step between data points (default: 1) |
| `interpolate` | bool | If True, linearly interpolates between data points (default: False) |

---
