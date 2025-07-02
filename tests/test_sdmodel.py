import pytest
from asdm import sdmodel
from asdm.utilities import plot_time_series

class GoalGap(sdmodel):
    def __init__(self):
        super(GoalGap, self).__init__()
        self.add_stock("Stock", 100, in_flows=['Flow'])
        self.add_aux("Goal", 20)
        self.add_aux("Adjustment_time", 5)
        self.add_aux("Gap", "Goal-Stock")
        self.add_flow("Flow", "Gap/Adjustment_time")

@pytest.fixture
def goal_gap_model():
    return GoalGap()

def test_init(goal_gap_model):
    pass

def test_simulation(goal_gap_model):
    goal_gap_model.simulate(time=20, dt=1)