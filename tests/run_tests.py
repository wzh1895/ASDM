import sys
import traceback
import logging
sys.path.append('ASDM')
import pathlib
import pprint
import pandas as pd
import numpy as np
# from ASDM.EngineStatic import Structure
from ASDM.Engine import Structure

### Load the tests and outcomes ###

path_tests = pathlib.Path('./BuiltinTestModels')
path_test_results = pathlib.Path('./BuiltinTestModels/Stella_outcomes')
tests = list()
outcomes = list()
for t in path_tests.iterdir():
    if t.suffix == '.stmx':
        tests.append(t)
        o = path_test_results/(t.stem+'.csv')
        if o.is_file():
            outcomes.append(o)
        else:
            raise Exception("File not found: {}".format(o))

# pprint.pprint(tests)
# print(outcomes)

### Test definition

def test(test_path, outcome_path):
    # Generate ASDM outcome
    model = Structure(from_xmile=test_path.resolve())
    
    model.simulate(dynamic=False)

    df_asdm = model.export_simulation_result(format='df')

    # Process ASDM outcome for comparison
    columns = list(df_asdm.columns)
    new_columns = [n.replace('_', ' ') for n in columns]
    column_name_map = dict(zip(df_asdm.columns, new_columns))

    asdm_outcome = df_asdm.rename(columns=column_name_map)
    # asdm_outcome.drop(['TIME'], axis=1, inplace=True)

    asdm_outcome = asdm_outcome[sorted(asdm_outcome.columns)]

    # Load Stella outcome
    df_stella = pd.read_csv(outcome_path)
    columns = list(df_stella.columns)
    new_columns = [column.strip('\"') for column in columns]
    column_name_map = dict(zip(columns, new_columns))
    df_stella.drop([columns[0]], axis=1, inplace=True)

    stella_outcome = df_stella.rename(columns=column_name_map)
    stella_outcome = stella_outcome[sorted(stella_outcome.columns)] 

    # Comparison by subtracting
    df_comparison = asdm_outcome.subtract(stella_outcome, axis=1)

    # Comparison with non_zeros
    df_comparison_non_zeros = pd.DataFrame()
    for column in df_comparison.columns:
        if not (-0.00001 < np.max(df_comparison[column].tolist()) < 0.00001):
            if column not in ['TIME', 'Unnamed: 0']:
                df_comparison_non_zeros[column] = df_comparison[column]

    if len(df_comparison_non_zeros.columns) == 0:
        return True
    else:
        return False

### Start the tests ###

passed = list()
failed = list()

for i in range(len(tests)):
    test_path = tests[i]
    outcome_path = outcomes[i]
    # print('Running test no. {:0>2}, {}'.format(i, test_path.stem))
    try:
        if_passed = test(test_path=test_path, outcome_path=outcome_path)
        if if_passed:
            passed.append((i, test_path.stem))
        else:
            failed.append((i, test_path.stem))
    except Exception as e:
        print('Test no. {:0>2}, {}'.format(i, test_path.stem))
        failed.append((i, test_path.stem))
        logging.error(traceback.format_exc())

print('\n-------Summary-------')
print('Total tests: {}'.format(len(tests)))
print('Passed     : {}'.format(len(passed)))
print('Failed     : {}'.format(len(failed)))
print()
for f in failed:
    print('             {:0>2} {}'.format(f[0], f[1]))
