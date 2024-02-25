from asdm.asdm import sdmodel
model = sdmodel(from_xmile='TestModels/simple care home model.stmx')
model.simulate(debug_against=False, verbose=False)
result = model.export_simulation_result(format='df', to_csv=True)