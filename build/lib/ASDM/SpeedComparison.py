from EngineOOPNewExe import Structure as Structure_new
from Engine import Structure as Structure_old
import timeit

def test_new():
    model = Structure_new(from_xmile='BuiltinTestModels/Goal_gap_array.stmx')
    model.parse()
    model.compile()
    model.simulate(time=10, dt=0.25)

t_new = timeit.timeit(test_new, number=100)
print('time new:', t_new)


def test_old():
    model = Structure_old(from_xmile='BuiltinTestModels/Goal_gap_array.stmx')
    model.simulate(simulation_time=10, dt=0.25)

t_old = timeit.timeit(test_old, number=100)
print('time old:', t_old)

print('{} times faster'.format(t_old/t_new))