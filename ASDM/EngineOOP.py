import copy
class Var(object):
    """
    The 'value' data structure for SD variables, like Int of Float.
    It does not have a name, but can be linked to a name in namespace.
    """
    def __init__(self, value=None):
        # if the value is not subscripted, wrap it with the default subscript

        if type(value) is not dict:
            self.value = {'nosubscript': value}
            self.historical_values = {'nosubscript':list()}
        else:
            self.value = value
            self.historical_values = dict()
            for k, _ in self.value.items():
                self.historical_values[k] = list()

        # self.historical_values.append(value)

    def set(self, new_value):
        if type(new_value) in [Var]:
            if list(new_value.value.keys()) == list(self.value.keys()):
                self.value = new_value.value
                for k, v in self.value.items():
                    self.historical_values[k].append(v)
            else:
                raise Exception("Var dimensions do not match: {} and {}".format(list(self.values.keys()), list(new_value.value.keys())))
        elif type(new_value) in [int, float]:
            if 'nosubscript' not in list(self.value.keys()):
                raise TypeError("Cannot assign a number to a subscripted variable.")
            else:
                self.value['nosubscript'] = new_value
                self.historical_values['nosubscript'].append(new_value)
        else:
            raise TypeError("Type of new value not suppported: {}".format(type(new_value)))

    def __add__(self, other):
        if type(other) in [Var]:
            try:
                new_value = dict()
                for s, v in self.value.items():
                    new_value[s] = v + other.value[s]
                return Var(value=new_value)
            except KeyError as e:
                raise e
        else:
            raise TypeError("Operator + cannot be used between {} and {}".format(type(self), type(other)))
    
    def __radd__(self, other):
        if type(other) in [Var]:
            try:
                new_value = dict()
                for s, v in self.value.items():
                    new_value[s] = v + other.value[s]
                return Var(value=new_value)
            except KeyError as e:
                raise e
        else:
            raise TypeError("Operator + cannot be used between {} and {}".format(type(other), type(self)))

    def __sub__(self, other):
        if type(other) in [Var]:
            try:
                new_value = dict()
                for s, v in self.value.items():
                    new_value[s] = v - other.value[s]
                return Var(value=new_value)
            except KeyError as e:
                raise e
        else:
            raise TypeError("Operator - cannot be used between {} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        if type(other) in [Var]:
            try:
                new_value = dict()
                for s, v in self.value.items():
                    new_value[s] = v / other.value[s]
                return Var(value=new_value)
            except KeyError as e:
                raise e
        else:
            raise TypeError("Operator / cannot be used between {} and {}".format(type(self), type(other)))

    def __eq__(self, other):
        if type(other) is not dict:
            if self.value['nosubscript'] == other:
                return True
            else:
                return False
        elif self.value == other:
            return True
        else:
            return False

    def get_history(self, nsteps=None):
        if nsteps is None:
            return self.historical_values
        elif nsteps <= len(self.historical_values):
            return historical_values[-1*nsteps:]
        else:
            raise Exception("Arg nsteps out of range: {}".format(nsteps))

    def __repr__(self):
        if list(self.value.keys()) == ['nosubscript']:
            return str(self.value['nosubscript'])
        else:
            return str(self.value)

    def __setitem__(self, item, item_value):
        self.value[item] = item_value

    def __getitem__(self, item):
        return self.value[item]


class SD_AbstractSyntaxTree(object):
    def __init__(self):
        pass

class Structure(object):
    # equations
    def __init__(self):
        self.stock_equations = {
            'tea_cup': 100
        }
        self.aux_equations = {
            'room': 20,
            'gap': 'room - tea_cup',
            'at': 5
        }
        self.flow_equations = {
            'change_tea_cup': 'gap/at'
        }
        
        # connections
        self.stock_flows = {
            'tea_cup':{
                'in':'change_tea_cup'
            }
        }

        # variable_values
        self.variables = {}

    # initialise stocks
    def init_stocks(self):
        for var, equation in self.stock_equations.items():
            if var not in self.variables.keys():
                self.variables[var] = Var()
            self.variables[var].set(eval(str(equation), None, self.variables))
        self.variables_stock = copy.copy(self.variables) # a shallow copy of self.variables
    
    # calculate flows backwards
    def backward_calculate(self, var, run_time_variables):
        # print('vars', self.variables)
        # print('vars_runtime', self.variables_runtime)
        if var not in run_time_variables.keys():
            if var not in self.variables.keys():
                self.variables[var] = Var()
            run_time_variables[var] = self.variables[var] # creat a reference to the actual Var()
        equation = (self.flow_equations | self.aux_equations)[var]
        # print('calculating: {} = {}'.format(var, equation))
        while True:
            try:
                run_time_variables[var].set(eval(str(equation), None, run_time_variables))
                # print('calculated: {}: {}'.format(var, run_time_variables[var]))
                break
            except NameError as e:
                required_var = e.args[0].split("'")[1]
                # print('Requiring:', required_var)
                self.backward_calculate(required_var, run_time_variables)

    def calculate_flows(self):
        self.variables_runtime = copy.copy(self.variables_stock)
        for var, _ in self.flow_equations.items():
            self.backward_calculate(var, self.variables_runtime)
    
    # update stocks
    def update_stocks(self):
        for stock, connections in self.stock_flows.items():
            for direction, flow in connections.items():
                if direction == 'in':
                    self.variables[stock].set(self.variables[stock] + self.variables[flow])
                elif direction == 'out':
                    self.variables[stock].set(self.variables[stock] - self.variables[flow])
                else:
                    raise Exception("Invalid flow direction: {}".format(direction))

    def simulate(self, t = 20):
        self.init_stocks()
        for _ in range(t):
            self.calculate_flows()
            self.update_stocks()
        self.calculate_flows()

        print('\nSummary:')
        print(self.variables)
        for k, v in self.variables.items():
            print('Current:', k, v)
            print('History:', k, v.get_history())
        print('\n')


model = Structure()
model.simulate()

import matplotlib.pyplot as plt
plt.plot(model.variables['tea_cup'].get_history()['nosubscript'])
plt.show()