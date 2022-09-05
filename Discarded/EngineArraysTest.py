from Discarded.EngineArrays import Structure

# model = Structure(from_xmile='BuiltInTestModels/Goal_gap.stmx')
# model = Structure(from_xmile='BuiltInTestModels/Goal_gap_array.stmx')
# model = Structure(from_xmile='BuiltInTestModels/Built_in_vars.stmx')
# model = Structure(from_xmile='BuiltInTestModels/IF_THEN_ELSE.stmx')
# model = Structure(from_xmile='BuiltInTestModels/Graph_function.stmx ')
# model = Structure(from_xmile='BuiltInTestModels/Array_cross_reference.stmx ')
# model = Structure(from_xmile='BuiltInTestModels/Delays.stmx ')
# model = Structure(from_xmile='BuiltInTestModels/Logic.stmx ')
# model = Structure(from_xmile='BuiltInTestModels/Conveyor.stmx ')
# model = Structure(from_xmile='BuiltInTestModels/Conveyor_leakage.stmx ')

# model = Structure(from_xmile='TestModels\Elective Recovery Model flattened.stmx')
model = Structure(from_xmile='TestModels\Elective Recovery Model.stmx')


model.clear_last_run()
model.simulate(simulation_time=10)

print('\nResults')
result = model.export_simulation_result().transpose()
print(result)
print(result.columns)
print('\nPASS\n')