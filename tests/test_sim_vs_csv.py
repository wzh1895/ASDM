# File: tests/test_simulation_vs_csv.py

import pytest
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from asdm import sdmodel

# test paths
test_batch_1 = Path('resources/basic_test_models')
test_batch_2 = Path('resources/comprehensive_test_models')

params = []

# For every file that ends with .stmx under test_1, add it and the .csv with same stem name to params
# for file in test_batch_1.glob('*.stmx'):
#     param = (file, file.with_suffix('.csv'))
#     print('collecting', param, '...')
#     params.append(param)

# Manually add more pairs from test_1
params.extend([
    (test_batch_1 / 'Non-negative_stocks_with_flows.stmx', test_batch_1 / 'Non-negative_stocks_with_flows.csv'),
])

# Add more pairs from test_2
# params.extend([
#     (test_batch_2 / 'IntegratedNewModel20250107.stmx', test_batch_2 / 'IntegratedNewModel20250107.csv'),
#     # (test_batch_2 / 'World3.stmx', test_batch_2 / 'World3.csv'),
#     (test_batch_2 / 'pop_coc_cvd.stmx', test_batch_2 / 'pop_coc_cvd.csv'),
# ])

def id_func(param):
    """Generate a more descriptive ID for each (model_path, csv_path) tuple."""
    model_path, csv_path = param
    # Option 1: Show entire path
    return f"{model_path.stem}"

def find_root_cause_variables(model, differing_vars, max_root_causes=10):
    """
    Find root cause variables using dependency graph analysis.
    
    Args:
        model: The asdm model instance
        differing_vars: List of variable names that have differences
        max_root_causes: Maximum number of root causes to return
    
    Returns:
        tuple: (root_causes, dependency_info)
    """
    # Build dependency graph for all differing variables
    combined_graph = nx.DiGraph()
    
    # Parse model if needed
    if model.state == 'loaded':
        model.parse()
    
    for var in differing_vars:
        try:
            # Extract base variable name without subscripts for dependency graph
            # Convert "variable_name[subscript1, subscript2]" to "variable_name"
            base_var_name = var.split('[')[0] if '[' in var else var
            
            # Create dependency graph for the base variable
            var_graph = model.create_variable_dependency_graph(base_var_name, mode='iter')
            combined_graph = nx.compose(combined_graph, var_graph)
        except Exception as e:
            # Handle other potential errors gracefully
            print(f"Warning: Could not create dependency graph for {var}: {e}")
            continue
    
    # Create mapping from base variable names to full variable names (with subscripts)
    base_to_full = {}
    for var in differing_vars:
        base_var = var.split('[')[0] if '[' in var else var
        if base_var not in base_to_full:
            base_to_full[base_var] = []
        base_to_full[base_var].append(var)
    
    # Find base variables that have differences and are in the dependency graph
    graph_differing_bases = [base_var for base_var in base_to_full.keys() if base_var in combined_graph.nodes()]
    
    if not graph_differing_bases:
        return [], "No differing variables found in dependency graph"
    
    # Find root causes: base variables with differences that don't depend on other base variables with differences
    root_cause_bases = []
    dependency_info = {}
    
    for base_var in graph_differing_bases:
        # Get all variables this base variable depends on (predecessors in the dependency graph)
        dependencies = set(nx.ancestors(combined_graph, base_var))
        
        # Check if any of its dependencies also have differences (among base variables)
        differing_dependencies = dependencies.intersection(set(graph_differing_bases))
        
        dependency_info[base_var] = {
            'total_dependencies': len(dependencies),
            'differing_dependencies': differing_dependencies,
            'is_root_cause': len(differing_dependencies) == 0,
            'full_variable_names': base_to_full[base_var]  # Keep track of all subscripted versions
        }
        
        if len(differing_dependencies) == 0:
            root_cause_bases.append(base_var)
    
    # Convert back to full variable names for the root causes (prefer the first subscripted version if multiple)
    root_causes = []
    for base_var in root_cause_bases:
        full_vars = base_to_full[base_var]
        root_causes.append(full_vars[0])  # Use the first subscripted version as representative
    
    # Sort root causes by number of variables they affect (descendants in the graph)
    def get_impact_score(var):
        base_var = var.split('[')[0] if '[' in var else var
        if base_var in combined_graph:
            return len(nx.descendants(combined_graph, base_var))
        return 0
    
    root_causes.sort(key=get_impact_score, reverse=True)
    
    return root_causes[:max_root_causes], dependency_info

ids = [id_func(param) for param in params]

@pytest.mark.parametrize(
    "model_path, csv_path",
    params,
    ids=ids
)
def test_simulation_output_vs_csv(model_path, csv_path):
    # 1. Run the model to get the simulated DataFrame
    model = sdmodel(
        from_xmile=model_path,
        parser_debug_level='info',
        solver_debug_level='info',
        simulator_debug_level='info',
    )
    model.simulate()
    df_model = model.export_simulation_result(format='df')
    
    # 2. Load the CSV file into a DataFrame
    df_reference = pd.read_csv(csv_path)
    # replace " " with "_" in column names
    df_reference.columns = df_reference.columns.str.replace(' ', '_')
    # sort columns
    df_reference = df_reference.reindex(sorted(df_reference.columns), axis=1)

    # 3. Compare the two DataFrames to identify the first row they differ
    difference_mask = np.abs(df_model - df_reference) > 1e-5 # there seems to be a precision cut off in the reference tool (e.g., Simulated=13865782.79459000, Reference=13865782.79460000, Diff= -0.00001000)
    
    # Export the numeric differences for debugging
    diff = df_model - df_reference
    diff = diff[difference_mask]
    diff.to_csv(f'resources/diff/asdm_diff_{model_path.stem}.csv', index=False)

    # If any difference is found, fail with info on the first differing row
    if difference_mask.any().any():
        first_diff_row_index = difference_mask.any(axis=1).idxmax()
        
        # Find which specific variables differ in this row
        row_diff_mask = difference_mask.loc[first_diff_row_index]
        differing_vars = row_diff_mask[row_diff_mask].index.tolist()
        
        # Create readable comparison for differing variables only
        comparison_details = []
        for var in differing_vars:
            sim_val = df_model.loc[first_diff_row_index, var]
            ref_val = df_reference.loc[first_diff_row_index, var]
            diff_val = sim_val - ref_val
            comparison_details.append(
                f"  {var:50s}: Simulated={sim_val:12.8f}, Reference={ref_val:12.8f}, Diff={diff_val:+12.8f}"
            )
        
        # Count total differences across all rows
        total_diffs = difference_mask.sum().sum()
        rows_with_diffs = difference_mask.any(axis=1).sum()
        
        # Perform root cause analysis
        try:
            root_causes, dependency_info = find_root_cause_variables(model, differing_vars)
            
            if root_causes:
                root_cause_text = "\n ROOT CAUSE ANALYSIS:\n"
                root_cause_text += f"   Found {len(root_causes)} potential root cause variable(s):\n"
                
                for i, root_var in enumerate(root_causes[:5], 1):  # Show top 5 root causes
                    sim_val = df_model.loc[first_diff_row_index, root_var] if root_var in df_model.columns else "N/A"
                    ref_val = df_reference.loc[first_diff_row_index, root_var] if root_var in df_reference.columns else "N/A"
                    
                    base_root_var = root_var.split('[')[0] if '[' in root_var else root_var
                    if base_root_var in dependency_info:
                        impact = len([v for v in dependency_info if base_root_var in dependency_info[v]['differing_dependencies']])
                        root_cause_text += f"   {i}. {root_var}\n"
                        root_cause_text += f"      Simulated: {sim_val}, Reference: {ref_val}\n"
                        root_cause_text += f"      Impact: Affects {impact} other variables\n"
                
                if len(root_causes) > 5:
                    root_cause_text += f"   ... and {len(root_causes) - 5} more\n"
                    
                root_cause_text += f"\n DEBUG TIP: Focus on these root causes first!\n"
                root_cause_text += f"   Use: variable_filter={root_causes[:3]} in your debug script\n"
            else:
                root_cause_text = "\n  ROOT CAUSE ANALYSIS: Could not identify clear root causes.\n"
                root_cause_text += "   All differing variables may be interdependent.\n"
                
        except Exception as e:
            root_cause_text = f"\n  ROOT CAUSE ANALYSIS FAILED: {e}\n"
        
        comparison_text = "\n".join(comparison_details)
        
        pytest.fail(
            f"For simulation_input={model_path}, csv_path={csv_path}, "
            f"DataFrames differ starting at row index {first_diff_row_index}.\n"
            f"Variables with differences in this row ({len(differing_vars)} variables):\n"
            f"{comparison_text}\n"
            f"Total differences found: {total_diffs} values across {rows_with_diffs} rows.\n"
            f"{root_cause_text}"
        )
    # Otherwise, the test succeeds (no explicit assertion needed).
