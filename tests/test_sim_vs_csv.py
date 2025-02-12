# File: tests/test_simulation_vs_csv.py

import pytest
import pandas as pd
import numpy as np

import importlib.util
import sys
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

spec = importlib.util.spec_from_file_location("asdm", Path('src/asdm/asdm.py'))
asdm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asdm)
sys.modules['asdm'] = asdm

# Import the module
from asdm import sdmodel

# test paths
test_batch_1 = Path('resources/basic_test_models')
test_batch_2 = Path('resources/comprehensive_test_models')

params = []

# For every file that ends with .stmx under test_1, add it and the .csv with same stem name to params
for file in test_batch_1.glob('*.stmx'):
    param = (file, file.with_suffix('.csv'))
    print('collecting', param, '...')
    params.append(param)

# params.extend([
#     (test_1 / 'Conveyor.stmx', test_1 / 'Conveyor.csv')
# ])

# Add more pairs from test_2
params.extend([
    (test_batch_2 / 'IntegratedNewModel20250107.stmx', test_batch_2 / 'IntegratedNewModel20250107.csv'),
    (test_batch_2 / 'World3.stmx', test_batch_2 / 'World3.csv'),
])

def id_func(param):
    """Generate a more descriptive ID for each (model_path, csv_path) tuple."""
    model_path, csv_path = param
    # Option 1: Show entire path
    return f"{model_path.stem}"

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
    # sort columns
    df_model = df_model.reindex(sorted(df_model.columns), axis=1)
    
    # 2. Load the CSV file into a DataFrame
    df_reference = pd.read_csv(csv_path)
    # sort columns
    df_reference = df_reference.reindex(sorted(df_reference.columns), axis=1)
    
    # 3. Compare the two DataFrames to identify the first row they differ (tolerance=1e-8)
    difference_mask = np.abs(df_model - df_reference) > 1e-8
    
    # Export the numeric differences for debugging
    diff = df_model - df_reference
    diff = diff[difference_mask]
    # diff.to_csv('asdm_diff.csv', index=False)

    # If any difference is found, fail with info on the first differing row
    if difference_mask.any().any():
        first_diff_row_index = difference_mask.any(axis=1).idxmax()

        pytest.fail(
            f"For simulation_input={model_path}, csv_path={csv_path}, "
            f"DataFrames differ starting at row index {first_diff_row_index}.\n"
            f"Simulated row:  {df_model.loc[first_diff_row_index].to_dict()}\n"
            f"Reference row:  {df_reference.loc[first_diff_row_index].to_dict()}"
        )
    # Otherwise, the test succeeds (no explicit assertion needed).
