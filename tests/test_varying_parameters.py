"""
Test suite for varying parameters during simulation runs.

This tests the ability to:
1. Simulate for a short period (e.g., 1 time unit)
2. Change parameter values using replace_element_equation()
3. Continue simulation from where it left off
4. Repeat as needed (interactive simulation pattern)
"""

import pytest
from asdm import sdmodel
import numpy as np
from pathlib import Path


class GoalGap(sdmodel):
    """Simple Goal Gap model for testing."""
    def __init__(self):
        super(GoalGap, self).__init__()
        self.add_stock("Stock", 100, in_flows=['Flow'])
        self.add_aux("Goal", 20)
        self.add_aux("Adjustment_time", 5)
        self.add_aux("Gap", "Goal-Stock")
        self.add_flow("Flow", "Gap/Adjustment_time")
        # Match stmx file specs to ensure consistency in tests
        self.sim_specs['dt'] = 0.5
        self.sim_specs['simulation_time'] = 8.0
        self.sim_specs['time_units'] = 'Months'


@pytest.fixture
def goal_gap_model():
    """Create a fresh Goal Gap model for each test."""
    return GoalGap()


def test_continuous_simulation_matches_single_run(goal_gap_model):
    """
    Test that running simulate() multiple times in sequence (without parameter changes)
    produces the same results as a single simulate() call.
    
    This verifies that the simulation truly continues from where it left off.
    """
    # Model 1: Run simulation all at once
    model1 = GoalGap()
    model1.simulate(time=10, dt=0.5)
    
    # Model 2: Run simulation in steps
    model2 = GoalGap()
    model2.simulate(time=2, dt=0.5)
    model2.simulate(time=2, dt=0.5)
    model2.simulate(time=2, dt=0.5)
    model2.simulate(time=2, dt=0.5)
    model2.simulate(time=2, dt=0.5)
    
    # Compare final Stock values
    stock1_final = model1.name_space['Stock']
    stock2_final = model2.name_space['Stock']
    
    print(f"Single run final Stock: {stock1_final}")
    print(f"Multi-step run final Stock: {stock2_final}")
    
    # They should be identical (or very close due to floating point)
    assert np.isclose(stock1_final, stock2_final, rtol=1e-10), \
        f"Stock values don't match: {stock1_final} vs {stock2_final}"
    
    # Compare time slices
    assert len(model1.time_slice) == len(model2.time_slice), \
        f"Different number of time slices: {len(model1.time_slice)} vs {len(model2.time_slice)}"
    
    # Verify time slice values match
    for time_point in model1.time_slice.keys():
        stock1_at_t = model1.time_slice[time_point]['Stock']
        stock2_at_t = model2.time_slice[time_point]['Stock']
        assert np.isclose(stock1_at_t, stock2_at_t, rtol=1e-10), \
            f"Stock at time {time_point} doesn't match: {stock1_at_t} vs {stock2_at_t}"


def test_time_slice_accumulation():
    """
    Test that time_slice accumulates properly across multiple simulate() calls.
    History should never be lost or overwritten.
    """
    model = GoalGap()
    
    # First simulation period
    model.simulate(time=2, dt=0.5)
    time_points_after_first = list(model.time_slice.keys())
    print(f"Time points after first simulation: {sorted(time_points_after_first)}")
    
    # Second simulation period
    model.simulate(time=2, dt=0.5)
    time_points_after_second = list(model.time_slice.keys())
    print(f"Time points after second simulation: {sorted(time_points_after_second)}")
    
    # All previous time points should still be present
    for tp in time_points_after_first:
        assert tp in time_points_after_second, \
            f"Time point {tp} was lost after second simulation"
    
    # Should have more time points now
    assert len(time_points_after_second) > len(time_points_after_first), \
        "Time slice didn't grow with second simulation"


def test_stmx_continuous_simulation():
    """
    Test continuous simulation with a model loaded from stmx file.
    Verifies that the interactive simulation feature works with loaded models,
    not just programmatically created ones.
    """
    stmx_path = Path(__file__).parent.parent / 'resources' / 'basic_test_models' / 'Goal_gap.stmx'
    
    # Model 1: Run simulation all at once
    model1 = sdmodel(from_xmile=str(stmx_path))
    model1.simulate(time=10, dt=0.5)
    
    # Model 2: Run simulation in steps
    model2 = sdmodel(from_xmile=str(stmx_path))
    model2.simulate(time=2, dt=0.5)
    model2.simulate(time=2, dt=0.5)
    model2.simulate(time=2, dt=0.5)
    model2.simulate(time=2, dt=0.5)
    model2.simulate(time=2, dt=0.5)
    
    # Compare final Stock values
    stock1_final = model1.name_space['Stock']
    stock2_final = model2.name_space['Stock']
    
    print(f"STMX single run final Stock: {stock1_final}")
    print(f"STMX multi-step run final Stock: {stock2_final}")
    
    assert np.isclose(stock1_final, stock2_final, rtol=1e-10), \
        f"Stock values don't match: {stock1_final} vs {stock2_final}"
    
    # Verify time slice values match
    for time_point in model1.time_slice.keys():
        stock1_at_t = model1.time_slice[time_point]['Stock']
        stock2_at_t = model2.time_slice[time_point]['Stock']
        assert np.isclose(stock1_at_t, stock2_at_t, rtol=1e-10), \
            f"Stock at time {time_point} doesn't match: {stock1_at_t} vs {stock2_at_t}"


def test_stmx_vs_programmatic_model():
    """
    Verify that a model loaded from stmx produces identical results
    to the same model created programmatically when using matching sim_specs.
    
    This demonstrates that both approaches are equivalent.
    """
    stmx_path = Path(__file__).parent.parent / 'resources' / 'basic_test_models' / 'Goal_gap.stmx'
    
    # STMX-loaded model
    stmx_model = sdmodel(from_xmile=str(stmx_path))
    stmx_model.simulate(time=8, dt=0.5)
    
    # Programmatically created model (now with matching sim_specs)
    programmatic_model = GoalGap()
    programmatic_model.simulate(time=8, dt=0.5)
    
    # Compare results at multiple time points
    for t in [0, 2, 4, 6, 8]:
        stmx_stock = stmx_model.time_slice[t]['Stock']
        prog_stock = programmatic_model.time_slice[t]['Stock']
        assert np.isclose(stmx_stock, prog_stock, rtol=1e-10), \
            f"At t={t}: STMX ({stmx_stock}) != Programmatic ({prog_stock})"
    
    print(f"STMX model final Stock: {stmx_model.name_space['Stock']}")
    print(f"Programmatic model final Stock: {programmatic_model.name_space['Stock']}")


def test_stmx_parameter_change():
    """
    Test parameter changes during simulation with stmx-loaded model.
    This is the primary use case: load a model from file and interact with it.
    """
    stmx_path = Path(__file__).parent.parent / 'resources' / 'basic_test_models' / 'Goal_gap.stmx'
    
    # Model 1: Goal stays at 20 throughout
    model1 = sdmodel(from_xmile=str(stmx_path))
    model1.simulate(time=5, dt=0.5)
    model1.simulate(time=5, dt=0.5)
    stock1_final = model1.name_space['Stock']
    
    # Model 2: Change Goal from 20 to 50 after 5 time units
    model2 = sdmodel(from_xmile=str(stmx_path))
    model2.simulate(time=5, dt=0.5)
    
    stock_at_change = model2.name_space['Stock']
    print(f"STMX Stock before Goal change: {stock_at_change}")
    
    model2.replace_element_equation('Goal', 50)
    print(f"STMX Goal changed from 20 to 50")
    
    model2.simulate(time=5, dt=0.5)
    stock2_final = model2.name_space['Stock']
    
    print(f"STMX Model 1 final Stock (Goal=20 always): {stock1_final}")
    print(f"STMX Model 2 final Stock (Goal changed to 50): {stock2_final}")
    
    # Model 2 should have a higher Stock value
    assert stock2_final > stock1_final, \
        f"Parameter change didn't affect simulation: {stock2_final} vs {stock1_final}"
    
    assert model2.name_space['Goal'] == 50, \
        f"Goal wasn't updated in name_space: {model2.name_space['Goal']}"


def test_stmx_multiple_parameter_changes():
    """
    Test multiple parameter changes with stmx-loaded model.
    This simulates a realistic interactive game scenario where the model
    is loaded from a file and user makes multiple decisions.
    """
    stmx_path = Path(__file__).parent.parent / 'resources' / 'basic_test_models' / 'Goal_gap.stmx'
    model = sdmodel(from_xmile=str(stmx_path))
    
    # Initial simulation
    model.simulate(time=2, dt=0.5)
    stock_t2 = model.name_space['Stock']
    print(f"STMX t=2: Stock={stock_t2}, Goal={model.name_space['Goal']}")
    
    # Change Goal to 60
    model.replace_element_equation('Goal', 60)
    model.simulate(time=2, dt=0.5)
    stock_t4 = model.name_space['Stock']
    print(f"STMX t=4: Stock={stock_t4}, Goal={model.name_space['Goal']}")
    
    # Change AT (Adjustment_time in stmx file is named 'AT')
    model.replace_element_equation('AT', 2)
    model.simulate(time=2, dt=0.5)
    stock_t6 = model.name_space['Stock']
    print(f"STMX t=6: Stock={stock_t6}, AT={model.name_space['AT']}")
    
    # Verify reasonable behavior
    assert stock_t2 < 100, "Stock should decrease initially"
    assert model.sim_specs['current_time'] == 6, \
        f"Simulation time should be 6, got {model.sim_specs['current_time']}"
    
    # Verify new parameter values are in effect
    assert model.name_space['Goal'] == 60, "Goal should be 60"
    assert model.name_space['AT'] == 2, "AT should be 2"


def test_resetting_parameter_to_same_value():
    """
    Test that setting a parameter to its current value (no actual change)
    doesn't break the simulation continuity.
    
    This tests the 'setting' behavior itself, not the value change.
    """
    # Model 1: Baseline - run in segments without any resetting
    model1 = GoalGap()
    model1.simulate(time=2, dt=0.5)
    model1.simulate(time=2, dt=0.5)
    model1.simulate(time=2, dt=0.5)
    model1.simulate(time=2, dt=0.5)
    model1.simulate(time=2, dt=0.5)
    stock1_final = model1.name_space['Stock']
    
    # Model 2: Run in segments, but reset Goal to same value after each segment
    model2 = GoalGap()
    model2.simulate(time=2, dt=0.5)
    model2.replace_element_equation('Goal', 20)  # Reset to same value
    
    model2.simulate(time=2, dt=0.5)
    model2.replace_element_equation('Goal', 20)  # Reset to same value
    
    model2.simulate(time=2, dt=0.5)
    model2.replace_element_equation('Goal', 20)  # Reset to same value
    
    model2.simulate(time=2, dt=0.5)
    model2.replace_element_equation('Goal', 20)  # Reset to same value
    
    model2.simulate(time=2, dt=0.5)
    stock2_final = model2.name_space['Stock']
    
    print(f"Model 1 final Stock (no resetting): {stock1_final}")
    print(f"Model 2 final Stock (reset to same value): {stock2_final}")
    
    # They should be identical since we're not actually changing the value
    assert np.isclose(stock1_final, stock2_final, rtol=1e-10), \
        f"Resetting to same value affected results: {stock1_final} vs {stock2_final}"
    
    # Verify time slices match
    assert len(model1.time_slice) == len(model2.time_slice), \
        f"Different number of time slices: {len(model1.time_slice)} vs {len(model2.time_slice)}"


def test_parameter_change_affects_simulation():
    """
    Test that changing an auxiliary variable between simulation steps
    affects subsequent simulation behavior.
    """
    # Model 1: Goal stays at 20 throughout
    model1 = GoalGap()
    model1.simulate(time=5, dt=0.5)
    model1.simulate(time=5, dt=0.5)
    stock1_final = model1.name_space['Stock']
    
    # Model 2: Change Goal from 20 to 50 after 5 time units
    model2 = GoalGap()
    model2.simulate(time=5, dt=0.5)
    
    # Change the Goal
    stock_at_change = model2.name_space['Stock']
    print(f"Stock before Goal change: {stock_at_change}")
    
    model2.replace_element_equation('Goal', 50)
    print(f"Goal changed from 20 to 50")
    
    model2.simulate(time=5, dt=0.5)
    stock2_final = model2.name_space['Stock']
    
    print(f"Model 1 final Stock (Goal=20 always): {stock1_final}")
    print(f"Model 2 final Stock (Goal changed to 50): {stock2_final}")
    
    # Model 2 should have a higher Stock value because Goal was increased
    assert stock2_final > stock1_final, \
        f"Parameter change didn't affect simulation: {stock2_final} vs {stock1_final}"
    
    # Verify the parameter was actually changed in name_space
    assert model2.name_space['Goal'] == 50, \
        f"Goal wasn't updated in name_space: {model2.name_space['Goal']}"


def test_multiple_parameter_changes():
    """
    Test changing parameters multiple times during a simulation run.
    This simulates an interactive game scenario.
    """
    model = GoalGap()
    
    # Initial simulation
    model.simulate(time=2, dt=0.5)
    stock_t2 = model.name_space['Stock']
    print(f"t=2: Stock={stock_t2}, Goal={model.name_space['Goal']}")
    
    # Change Goal to 60
    model.replace_element_equation('Goal', 60)
    model.simulate(time=2, dt=0.5)
    stock_t4 = model.name_space['Stock']
    print(f"t=4: Stock={stock_t4}, Goal={model.name_space['Goal']}")
    
    # Change Goal to 30
    model.replace_element_equation('Goal', 30)
    model.simulate(time=2, dt=0.5)
    stock_t6 = model.name_space['Stock']
    print(f"t=6: Stock={stock_t6}, Goal={model.name_space['Goal']}")
    
    # Change Adjustment_time to 2 (faster adjustment)
    model.replace_element_equation('Adjustment_time', 2)
    model.simulate(time=2, dt=0.5)
    stock_t8 = model.name_space['Stock']
    print(f"t=8: Stock={stock_t8}, Adjustment_time={model.name_space['Adjustment_time']}")
    
    # Verify Stock is changing in reasonable ways
    assert stock_t2 < 100, "Stock should decrease initially (Goal < Stock)"
    # At t=2, Stock is ~85, so even with Goal=60, Stock > Goal, so it should continue decreasing
    assert stock_t4 < stock_t2, "Stock should continue decreasing because Stock (85) > Goal (60)"
    # At t=4, Stock is ~80, so with Goal=30, it should continue decreasing
    assert stock_t6 < stock_t4, "Stock should continue decreasing because Stock (80) > Goal (30)"
    
    # Verify final state
    assert model.sim_specs['current_time'] == 8, \
        f"Simulation time should be 8, got {model.sim_specs['current_time']}"
    
    # Verify all time points are in time_slice
    expected_time_points = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 
                           4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    for tp in expected_time_points:
        assert tp in model.time_slice or np.isclose(tp, list(model.time_slice.keys()), atol=1e-10).any(), \
            f"Time point {tp} not in time_slice"


def test_state_management_with_parameter_changes():
    """
    Test that model state is managed correctly when parameters change.
    """
    model = GoalGap()
    
    # Initial state after creation
    assert model.state == 'loaded', f"Initial state should be 'loaded', got '{model.state}'"
    
    # After first simulation
    model.simulate(time=2, dt=0.5)
    assert model.state == 'simulated', f"After simulation, state should be 'simulated', got '{model.state}'"
    
    # After parameter change
    model.replace_element_equation('Goal', 50)
    assert model.state == 'changed', f"After parameter change, state should be 'changed', got '{model.state}'"
    
    # After continuing simulation
    model.simulate(time=2, dt=0.5)
    assert model.state == 'simulated', f"After continuing simulation, state should be 'simulated', got '{model.state}'"


def test_time_continuity():
    """
    Test that TIME variable continues correctly across multiple simulate() calls.
    """
    model = GoalGap()
    
    # First period: 0 to 3
    model.simulate(time=3, dt=1)
    time_after_first = model.sim_specs['current_time']
    time_var_after_first = model.name_space['TIME']
    
    print(f"After first simulate: current_time={time_after_first}, TIME={time_var_after_first}")
    assert time_after_first == 3, f"current_time should be 3, got {time_after_first}"
    assert time_var_after_first == 3, f"TIME should be 3, got {time_var_after_first}"
    
    # Second period: 3 to 6
    model.simulate(time=3, dt=1)
    time_after_second = model.sim_specs['current_time']
    time_var_after_second = model.name_space['TIME']
    
    print(f"After second simulate: current_time={time_after_second}, TIME={time_var_after_second}")
    assert time_after_second == 6, f"current_time should be 6, got {time_after_second}"
    assert time_var_after_second == 6, f"TIME should be 6, got {time_var_after_second}"


def test_pause_parameter():
    """
    Test the pause parameter of simulate().
    
    According to docstring:
    - pause=True: simulation should pause after the specified time
    - If time is not specified, simulation should pause after last iteration
    
    This test may fail if pause is not yet implemented.
    """
    model = GoalGap()
    
    # Test pause=True with specified time
    model.simulate(time=2, dt=0.5, pause=True)
    print(f"After pause=True: state={model.state}, current_time={model.sim_specs['current_time']}")
    
    # Continue simulation
    model.simulate(time=2, dt=0.5, pause=False)
    print(f"After pause=False: state={model.state}, current_time={model.sim_specs['current_time']}")
    
    # This test is exploratory - just verify it doesn't crash
    assert model.sim_specs['current_time'] == 4, \
        f"Expected current_time=4, got {model.sim_specs['current_time']}"


def test_comparison_baseline_vs_interactive():
    """
    Create a baseline model and an interactive model that should produce the same results
    if parameters change to the same values at the right times.
    """
    # Baseline: single run with fixed parameters
    baseline = GoalGap()
    baseline.simulate(time=10, dt=0.5)
    baseline_stock = baseline.name_space['Stock']
    
    # Interactive: no parameter changes (should match baseline)
    interactive = GoalGap()
    for i in range(5):
        interactive.simulate(time=2, dt=0.5)
    interactive_stock = interactive.name_space['Stock']
    
    print(f"Baseline final Stock: {baseline_stock}")
    print(f"Interactive final Stock: {interactive_stock}")
    
    assert np.isclose(baseline_stock, interactive_stock, rtol=1e-10), \
        f"Baseline and interactive don't match: {baseline_stock} vs {interactive_stock}"


def test_parameter_change_immediate_effect():
    """
    Test that parameter changes take effect immediately in the next simulation step.
    """
    model = GoalGap()
    model.simulate(time=1, dt=0.5)
    
    # Record current values
    stock_before = model.name_space['Stock']
    goal_before = model.name_space['Goal']
    
    # Change Goal to a very different value
    model.replace_element_equation('Goal', 200)
    
    # Verify the change is reflected in name_space before next simulation
    # Note: This may or may not be how it currently works - test will reveal
    goal_after_change = model.name_space.get('Goal', 'not in name_space')
    print(f"Goal before change: {goal_before}")
    print(f"Goal after replace_element_equation: {goal_after_change}")
    
    # Simulate one more step
    model.simulate(time=1, dt=0.5)
    
    # The flow should now be much larger due to bigger gap
    flow_after = model.name_space['Flow']
    gap_after = model.name_space['Gap']
    
    print(f"Stock after: {model.name_space['Stock']}")
    print(f"Gap after: {gap_after}")
    print(f"Flow after: {flow_after}")
    
    # With Goal=200 and Stock starting around 92, Gap should be large (>90)
    # Stock will increase during the simulation, so gap will be less than 200 - 92 = 108
    assert gap_after > 90, f"Gap should be large with Goal=200, got {gap_after}"
    assert flow_after > 15, f"Flow should be large with big gap, got {flow_after}"


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
