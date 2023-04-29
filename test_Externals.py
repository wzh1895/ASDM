from ASDM.ASDM import Structure, DataFeeder
import matplotlib.pyplot as plt

goal = DataFeeder(
    data=[20, 25, 30, 35, 30, 20, 15, 20, 25, 30, 35, 30, 20, 15]
)

class GoalGap(Structure):
    def __init__(self):
        super(GoalGap, self).__init__()
        self.add_stock("Stock", 100, in_flows=['Flow'])
        self.add_aux("Goal", goal)
        self.add_aux("Adjustment_time", 5)
        self.add_aux("Gap", "Goal-Stock")
        self.add_flow("Flow", "Gap/Adjustment_time")

model = GoalGap()

model.simulate(time=13)

result = model.export_simulation_result(format='df')

print(result)

result.plot()
plt.show()