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

model_path = Path('resources') / 'IntegratedNewModel20250107.stmx'
# model_path = Path('resources') / 'SimpleStockFlow.stmx'

model = sdmodel(
        from_xmile=model_path,
        parser_debug_level='info',
        solver_debug_level='info',
        # simulator_debug_level='info',
        simulator_debug_level='debug',
    )

# dg = model.generate_dependent_graph(show='init')
# dg = model.generate_dependent_graph(show='iter')

model.simulate(time=1)
a = model.export_simulation_result(format='df')
print(a)
print()

model.replace_element_equation('NewOrderRate', 3)
model.simulate(time=1)
a = model.export_simulation_result(format='df')
print(a)
print()

model.replace_element_equation('NewOrderRate', 0)
model.simulate(time=1)
a = model.export_simulation_result(format='df')
print(a)
print()

model.simulate(time=3)
a = model.export_simulation_result(format='df')
print(a)
print()