# asdm 

## Documentation
`Version 25 Feb 2024`

**Note:** The documentation shown below heavily relied on OpenAI's `ChatGPT`, and may contain undiscovered errors. We recommend that you also refer to the source code of [asdm](asdm/asdm.py) for the exact useage of the functionalities. Please open an `issue` or contact me if you believe there is a bug. You are also welcomed to open a `pull request` if you come up with a fix, a patch, or a new feature.

### Creation of SD Models
```
def __init__(self, from_xmile=None):
    """
    Initializes the sdmodel instance, optionally loading a model from an XMILE file.

    Parameters:
    - from_xmile (str, optional): The file path to an XMILE model file. If provided,
      the class will attempt to load and parse the XMILE file to set up the model's
      initial configuration.

    Attributes initialized here include debug settings, simulation specifications,
    model components (stocks, flows, auxiliaries, etc.), and the simulation environment.
    Additionally, parsers and solvers for the model equations are set up.
    """
```
### Model Building Methods
```
def add_stock(self, name, equation, non_negative=True, is_conveyor=False, in_flows=[], out_flows=[]):
    """
    Adds a stock variable to the model.

    Parameters:
    - name (str): The name of the stock variable.
    - equation: The equation defining the stock's behavior.
    - non_negative (bool, optional): Ensures that the stock value cannot be negative.
      Defaults to True.
    - is_conveyor (bool, optional): Specifies if the stock acts as a conveyor.
      Defaults to False.
    - in_flows (list, optional): A list of inflow variable names to the stock.
    - out_flows (list, optional): A list of outflow variable names from the stock.

    This method does not return any value.
    """
```
```
def add_flow(self, name, equation, leak=None, non_negative=False):
    """
    Adds a flow variable to the model.

    Parameters:
    - name (str): The name of the flow variable.
    - equation: The equation defining the flow's behavior.
    - leak (optional): Specifies if the flow acts as a leakage. Defaults to None.
    - non_negative (bool, optional): Ensures that the flow value cannot be negative.
      Defaults to False.

    This method does not return any value.
    """
```
```
def add_aux(self, name, equation):
    """
    Adds an auxiliary variable to the model.

    Parameters:
    - name (str): The name of the auxiliary variable.
    - equation: The equation defining the auxiliary variable's behavior.

    This method does not return any value.
    """
```
### Model Modification Methods
```
def replace_element_equation(self, name, new_equation):
    """
    Replaces the equation of a specified model element (stock, flow, or auxiliary variable)
    with a new equation.

    Parameters:
    - name (str): The name of the model element (stock, flow, or auxiliary variable) whose
      equation is to be replaced.
    - new_equation: The new equation to replace the existing one. This can be a string
      representation of the equation or a numerical value. The type of `new_equation`
      must be either `str`, `int`, `float`, or a compatible numpy numeric type.

    This method updates the model's internal representation of the specified element's
    equation to the new equation provided.

    Raises:
    - Exception: If the new equation's type is unsupported or if the specified element
      name does not exist within the current model.

    This method does not return any value.
    """
```
```
def overwrite_graph_function_points(self, name, new_xpts=None, new_xscale=None, new_ypts=None):
    """
    Overwrites the points or scale of a graph function associated with a model element
    (stock, flow, or auxiliary variable).

    This method is specifically useful for dynamically modifying the behavior of elements
    that are defined using graph functions within the simulation model.

    Parameters:
    - name (str): The name of the model element (stock, flow, or auxiliary variable) associated
      with the graph function to be modified.
    - new_xpts (list of float, optional): A new list of x points for the graph function. If None,
      the x points are not modified.
    - new_xscale (tuple of float, optional): A new x scale (min, max) for the graph function. If None,
      the x scale is not modified.
    - new_ypts (list of float, optional): A new list of y points for the graph function. If None,
      the y points are not modified.

    This method allows for dynamic adjustments to the graph functions used in the model,
    enabling scenarios such as sensitivity analysis or scenario testing.

    Raises:
    - Exception: If all input parameters (`new_xpts`, `new_xscale`, `new_ypts`) are None,
      indicating that there are no modifications to make.

    This method does not return any value but updates the graph function of the specified
    element with the new points or scale provided.
    """
```
### Simulation Methods
```
def simulate(self, time=None, dt=None, dynamic=True, verbose=False, debug_against=None):
    """
    Runs the simulation of the model over the specified time.

    Parameters:
    - time (optional): The total time to simulate. If None, uses the simulation time
      specified in sim_specs.
    - dt (optional): The time step to use for the simulation. If None, uses the dt
      specified in sim_specs.
    - dynamic (bool, optional): If True, allows dynamic adjustment of simulation
      parameters. Defaults to True.
    - verbose (bool, optional): If True, prints detailed logs of the simulation process.
      Defaults to False.
    - debug_against (optional): Specifies a file or a flag for debugging purposes.

    This method updates the model's state based on the simulation results.
    """
```
### Result Management Methods
```
def export_simulation_result(self, flatten=False, format='dict', to_csv=False, dt=False):
    """
    Exports the results of the simulation in the specified format.

    Parameters:
    - flatten (bool, optional): Only useful when the model uses arrays. Normally results of arrayed variables are stored as dictionaries like {dimension: value}. If True, flattens the result structure. Flattened result treats each dimension as a separate variable with name 'variable_dimension'. Defaults to False.
    - format (str, optional): The format of the output ('dict' or 'df' for DataFrame).
      Defaults to 'dict'.
    - to_csv (bool or str, optional): If True or a file path is provided, exports the
      results to a CSV file. Defaults to False.
    - dt (bool, optional): If True, includes the simulation time in the results.
      Defaults to False.

    Returns:
    - The simulation results in the specified format.
    """
```
```
def display_results(self, variables=None):
    """
    Displays the simulation results for the specified variables using a line plot.

    Parameters:
    - variables (list or str, optional): The names of the variables to display. If None,
      displays results for all variables.

    This method does not return any value but shows a plot of the selected variables'
    values over time.
    """
```