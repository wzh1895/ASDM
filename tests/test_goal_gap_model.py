'''
- This module tests the ASDM framework component outputs to make sure they are robust and tested
- In the next iteration we will add python version testing and addition tests but we will start with  simple unit test

- Version 0.0.1 (First TDD Sample)

If needed during development - not needed in production
# import sys
# import os
# sys.path.append(
#      os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
'''

# Main dependencies
import unittest
import pandas as pd
from ASDM.Engine import Structure

# Initialsise the test model class
class GoalGap(Structure):
    def __init__(self):
        super(GoalGap, self).__init__()
        self.add_stock("Stock", 100)
        self.add_aux("Goal", 20)
        self.add_aux("Adjustment_time", 5)
        self.add_aux("Gap", "Goal-Stock")
        self.add_flow("Flow", "Gap/Adjustment_time", flow_to="Stock") 

# Goal Gap Tests
class Goal_Gap_Test(unittest.TestCase):
    
    def setUp(self):
        self.goal_gap_model = GoalGap()

    # Test that goal_gap model initialised successfully and dataframe successfully generated
    def test_goal_gap_model_exists_and_dataframe_generated(self):
        self.goal_gap_model.clear_last_run()
        self.goal_gap_model.simulate(simulation_time=20, dt=1)
        self.df_goal_gap = self.goal_gap_model.export_simulation_result()
        self.assertTrue(isinstance(self.df_goal_gap, type(pd.DataFrame())))
        self.assertTrue(isinstance(self.df_goal_gap['Stock'].to_list()[0], float))

    # Test that goal_gap model initialised successfully and dataframe successfully generated
    def test_goal_gap_model_simulation_behaves_as_expected_data(self):
        self.goal_gap_model.clear_last_run()
        self.goal_gap_model.simulate(simulation_time=20, dt=1)
        self.df_goal_gap = self.goal_gap_model.export_simulation_result()
        self.assertLess(self.df_goal_gap['Flow'][0], self.df_goal_gap['Flow'][1])
        self.assertEqual(self.df_goal_gap['Goal'][0], self.df_goal_gap['Goal'][1])
        self.assertEqual(self.df_goal_gap['Adjustment_time'][0], self.df_goal_gap['Adjustment_time'][1])
        self.assertGreater(self.df_goal_gap['Stock'][0], self.df_goal_gap['Stock'][1])


if __name__ == '__main__':
    unittest.main()