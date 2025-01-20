import importlib.util
import sys
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

spec = importlib.util.spec_from_file_location("asdm", Path("src/asdm/asdm.py"))
asdm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asdm)
sys.modules['asdm'] = asdm

from asdm import sdmodel

model_path = Path('resources/basic_test_models') / 'Conveyor.stmx'

model = sdmodel(
        from_xmile=model_path,
        parser_debug_level='info',
        solver_debug_level='info',
        simulator_debug_level='debug',
    )

dg = model.generate_dependent_graph(show='init')
dg = model.generate_dependent_graph(show='iter')

model.simulate(time=1)
result = model.export_simulation_result(format='df')
result.to_csv('resources/debug.csv')