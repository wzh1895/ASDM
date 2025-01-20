import importlib.util
import sys
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# comment out the following lines to use the installed version of asdm
spec = importlib.util.spec_from_file_location("asdm", Path("src/asdm/asdm.py"))
asdm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asdm)
sys.modules['asdm'] = asdm

from asdm import sdmodel

model_path = Path('resources') / 'test_model.stmx'

model = sdmodel(
        from_xmile=model_path,
        # parser_debug_level='info',
        # solver_debug_level='info',
        # simulator_debug_level='info',
        # simulator_debug_level='debug',
    )

# dg = model.generate_dependent_graph(show='init')
# dg = model.generate_dependent_graph(show='iter')

import time
import tqdm
start = time.time()

for i in tqdm.tqdm(range(100)):
    model.simulate(time=100)
    model.clear_last_run()

end = time.time()

print(f"Elapsed time: {end - start}")