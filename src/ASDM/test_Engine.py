from ASDM import Structure


if __name__ == '__main__':

    #### Test Models ###

    # model = Structure(from_xmile='BuiltinTestModels/Basic_math.stmx')
    # model = Structure(from_xmile='BuiltinTestModels/MOD.stmx')
    # model = Structure(from_xmile='BuiltinTestModels/MOD_arrayed.stmx')
    # model = Structure(from_xmile='BuiltinTestModels/Min_Max.stmx')
    # model = Structure(from_xmile='BuiltinTestModels/Non-negative_stocks.stmx')
    # model = Structure(from_xmile='BuiltinTestModels/Non-negative_stocks_with_flows.stmx')
    # model = Structure(from_xmile='BuiltinTestModels/Isolated_var.stmx')

    # model = Structure(from_xmile='BuiltinTestModels/Goal_gap.stmx')
    # model = Structure(from_xmile='BuiltinTestModels/Time_unit.stmx')

    # model = Structure(from_xmile='BuiltinTestModels/Goal_gap_array.stmx')
    # model = Structure(from_xmile='BuiltinTestModels/Array_parallel_reference.stmx')
    # model = Structure(from_xmile='BuiltinTestModels/Array_cross_reference.stmx')
    # model = Structure(from_xmile='BuiltinTestModels/Array_cross_reference_inference.stmx')
    
    # model = Structure(from_xmile='BuiltInTestModels/Built_in_vars.stmx')
    
    # model = Structure(from_xmile='BuiltInTestModels/Logic.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/IF_THEN_ELSE.stmx')
    
    # model = Structure(from_xmile='BuiltInTestModels/Graph_function.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/LOOKUP.stmx')

    # model = Structure(from_xmile='BuiltInTestModels/INIT.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/Delays.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/Delays2.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/History.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/Smooth.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/Time_related_functions.stmx')
    
    # model = Structure(from_xmile='BuiltInTestModels/Conveyor.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/Conveyor_leakage.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/Conveyor_leakage1.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/Conveyor_leakage2.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/Conveyor_initialisation.stmx')

    ### Production Models ###

    model = Structure(from_xmile='TestModels/Elective Recovery Model.stmx')
    # model=Structure(from_xmile='TestModels/2022_07_14 no milk without meat.stmx')
    # model=Structure(from_xmile='TestModels/TempTest1.stmx')

    ### Controls ###

    # Dynamic simulation
    model.simulate(debug_against=True, verbose=False)
    model.summary()

    ### Simulation inspections ###

    model.export_simulation_result(to_csv=True)
    # r = model.export_simulation_result(format='df', to_csv=False)
    # print(r)

    # vars_to_view = [
    #     '13wk_wait_for_urgent_treatment', 
    # #     'Negative_test_results', 
    # #     'COVID_modified_percent_urgent', 
    # #     'Undergoing_diagnostic_tests',
    # #     'Positive_test_results_urgent',
    # #     'Less_than_6mth_to_urgent',
    # #     'Between_6_to_12mth_wait_to_urgent',
    # #     'Between_12_to_24mth_wait_to_urgent',
    # #     'Urgent_treatment',
    # #     'Total_treatment_capacity',
    # #     'Routine_treatment',
    # #     'Net_COVID_induced_changes_in_underlying_health_needs?'
    #     ]
    
    # vars_to_view = list(model.name_space.keys())
    # vars_to_view.remove('TIME')
    # model.display_results(variables=vars_to_view)