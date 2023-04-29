from ASDM import Parser

if __name__ == '__main__':

    string_0a = 'a'
    string_1a = 'a+b-c'
    string_1b = 'a+b-2'
    string_2a = 'a*b'
    string_2b = 'a/2'
    string_2c = 'a*b*c'
    string_3a = 'INIT(a)'
    string_3b = 'DELAY(a, 1)'
    string_3c = 'DELAY(a+1, 2, 3)'
    string_4a = 'a*((b+c)-d)'
    string_4b = '4*(1-(2*3))'
    string_4c = '10-4-3-2-1'
    string_5a = 'a > b'
    string_5b = 'a < 2'
    string_5c = 'a >= b'
    string_5d = 'a <= 2'
    string_6a = 'IF a THEN b ELSE c'
    string_6b = 'IF a THEN IF ba THEN bb ELSE bc ELSE c'
    string_7a = 'a[b]'
    string_7b = 'a[b,c]'
    string_7c = 'a[1,b]'
    string_full = 'IF (a + h[ele1]) > (c * 10) THEN INIT(d / e) ELSE f - g'

    string_a = '(Pre_COVID_capacity_for_diagnostics*(1-(COVID_period/100)*(Reduced_diagnostic_capacity_during_COVID/100)*COVID_switch))+((((Percent_increase_in_diagnostic_capacity_post_COVID/100)*COVID_switch)*Pre_COVID_capacity_for_diagnostics)*(Timing_of_new_diagnostic_capacity/100))'
    string_b = '(a*(1-(b/100)*(c/100)*d))+((((e/100)*d)*a)*(e/100))' # equivalent to string_a
    string_c = 'a*b*c*d'
    string_d = '(a*(1-(b/100)*(c/100)*d))'
    string_e = 'DELAY(Waiting_more_than_12mths, 52)-Between_12_to_24mth_wait_to_urgent-Waiting_more_than_12mths-Routine_treatment_from_12_to_24mth_wait'
    string_f = 'DELAY(a, 52)-b-a-c'
    string_g = '( IF TIME < DEMAND_CHANGE_START_YEAR THEN BASELINE_PC_CONSUMPTION_OF_BOVINE_MEAT ELSE (1-SWITCH_CONSUMPTION_RECOMMENDATIONS_0_off_1_on)*BASELINE_PC_CONSUMPTION_OF_BOVINE_MEAT+recommended_pc_consumption_of_bovine_meat*SWITCH_CONSUMPTION_RECOMMENDATIONS_0_off_1_on )'
    string_h = '( IF a < b THEN c ELSE (1-d)*c+e*d )' # equivalent to string_g
    string_i = '(Waiting_6mths_for_treatment*(percent_becoming_urgent_by_waiting_time_pa[Less_than_6mths]/100))/52'
    string_j = 'HISTORY(a+1,  1)'
    string_k = 'Expected_population_rate_of_incidence_pw+((Underlying_trend_in_health_needs-1)*Underlying_trend_in_health_needs*Switch_for_demographic_increase)'
    string_l = '(LeakingFlow_1+0.0001)/(OutFlow_1+LeakingFlow_1+0.0001)'
    string_m = 'DELAY(Waiting_more_than_6mths, 26)-Between_6_to_12mth_wait_to_urgent-Routine_treatment_from_6_to_12th_waits'
    string_n = 'IF TIME <= 35 THEN Actual_Flexi_list_per_month ELSE Expected_FS_per_month'

    parser = Parser()
    graph = parser.parse(string_n, verbose=True)
    parser.plot_ast(graph=graph)
