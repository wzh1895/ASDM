from ASDM import Structure

class Infection(Structure):
    def __init__(self):
        super(Infection, self).__init__()
        self.add_stock("Infected", 100, in_flows=['IncreaseRate'])
        self.add_flow(
            name='IncreaseRate',
            equation='Infected*FractionalIncreaseRate',
            )
        self.add_aux(
            name='FractionalIncreaseRate',
            equation=0.1
            )
        
model = Infection()

sim_time = 7
dt = 1

model.clear_last_run()
model.simulate(time=sim_time, dt=dt)
df_model = model.export_simulation_result(format='df')
print(df_model)

model.replace_element_equation('FractionalIncreaseRate', 0.2)

model.clear_last_run()
model.simulate(time=sim_time, dt=dt)
df_model = model.export_simulation_result(format='df')
print(df_model)
