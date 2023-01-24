import copy
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from pprint import pprint

class Var(object):
    """
    The 'value' data structure for SD variables, like Int or Float.
    It does not have a name, but can be linked to a name in the namespace.
    """
    def __init__(self, value=None, copy=None, dims=None):
        
        if copy is not None: 
            if  type(copy) is Var:
                if type(copy.value) is dict:
                    self.historical_values=dict()
                    self.value = dict()
                    for k, v in copy.value.items():
                        self.value[k] = 0
                        self.historical_values[k] = list()
                else:
                    self.historical_values = {'nosubscript':list()}
                    self.value = 0
            else:
                raise TypeError
        
        elif dims is not None:
            self.value = dict()
            self.historical_values = dict()
            for k in dims:
                self.value[k] = 0
                self.historical_values[k] = list()
        
        else:
            if type(value) is dict:
                self.value = value
                self.historical_values = dict()
                for k, _ in self.value.items():
                    self.historical_values[k] = list()
            else:
                self.value = value
                self.historical_values = {'nosubscript':list()}
            
        # self.historical_values.append(value)

    def set(self, new_value):
        '''
        N.B. SVar = SubscriptedVar

        Var <- int/float OK
        SVar <- int/float NO
        Var <- Var OK
        SVar <- Var NO
        SVar <- SVar OK if dimensions match
        '''
        if type(self.value) is dict: # self is SVar
            if type(new_value) is Var and type(new_value.value) is dict: # SVar <- SVar
                if list(new_value.value.keys()) == list(self.value.keys()):
                    for k, v in new_value.value.items():
                        self.value[k] = v
                        self.historical_values[k].append(v)
                else:
                    raise Exception("Dimensions don't match.")
            else:
                raise Exception("Type of new value {} doesn't match".format(type(new_value)))
        else: # self is Var
            if type(new_value) is Var and type(new_value.value) is dict: #Var <- Svar
                raise Exception("Var<-Svar, NO")
            elif type(new_value) is Var and type(new_value.value) in [int, float]: #Var <- Var
                self.value = new_value.value 
                self.historical_values['nosubscript'].append(new_value.value)
            elif type(new_value) in [int, float]:
                self.value = new_value
                self.historical_values['nosubscript'].append(new_value)
            else:
                raise Exception("Unknown match issue: {}({}) and {}({})".format(type(self.value), self.value, type(new_value), new_value))

    def __add__(self, other):
        if type(other) in [int, float]:
            if type(self.value) in [int, float]:
                return self.value + other
            else:
                raise TypeError("Operator + cannot be used between {} ({}) and {} ({})".format(type(self), self, type(other), other))
        elif type(other) in [Var]:
            if type(self.value) in [int, float]:
                if type(other.value) in [int, float]:
                    return self.value + other.value
                else:
                    raise TypeError("Operator + cannot be used between {} ({}) and {} ({})".format(type(self), self, type(other), other))
            elif type(self.value) is dict:
                try:
                    new_value = dict()
                    for s, v in self.value.items():
                        new_value[s] = v - other.value[s]
                    return Var(value=new_value)
                except KeyError as e:
                    raise e
        else:
            raise TypeError("Operator + cannot be used between {} ({}) and {} ({})".format(type(self), self, type(other), other))

    def __radd__(self, other):
        if type(other) in [int, float]:
            if type(self.value) in [int, float]:
                return self.value + other
            else:
                raise TypeError("Operator + cannot be used between {} ({}) and {} ({})".format(type(other), other, type(self), self))
        elif type(other) in [Var]:
            if type(self.value) in [int, float]:
                if type(other.value) in [int, float]:
                    return self.value + other.value
                else:
                    raise TypeError("Operator + cannot be used between {} ({}) and {} ({})".format(type(other), other, type(self), self))
            elif type(self.value) is dict:
                try:
                    new_value = dict()
                    for s, v in self.value.items():
                        new_value[s] = v - other.value[s]
                    return Var(value=new_value)
                except KeyError as e:
                    raise e
        else:
            raise TypeError("Operator + cannot be used between {} ({}) and {} ({})".format(type(other), other, type(self), self))
    
    def __mul__(self, other):
        if type(self.value) is dict and (type(other) in [Var] and type(other.value) is dict):
            try:
                new_value = dict()
                for s, v in self.value.items():
                    new_value[s] = v * other.value[s]
                return Var(value=new_value)
            except KeyError as e:
                raise e
        
        elif type(self.value) is dict and type(other) in [int, float]:
            new_value = dict()
            for s, v in self.value.items():
                new_value[s] = v * other
            return Var(value=new_value)
            
        elif type(self.value) is dict and type(other.value) in [int, float]:
            new_value = dict()
            for s, v in self.value.items():
                new_value[s] = v * other.value
            return Var(value=new_value)
        elif type(self.value) in [int, float] and type(other) in [int, float]:
            return self.value * other

        elif type(self.value) in [int, float] and type(other.value) in [int, float]:
            return self.value * other.value
        else:
            raise TypeError("Operator * cannot be used between {} ({}) and {} ({})".format(type(self), self, type(other), other))

    def __rmul__(self, other):
        if type(self.value) is dict and (type(other) in [Var] and type(other.value) is dict):
            try:
                new_value = dict()
                for s, v in self.value.items():
                    new_value[s] = v * other.value[s]
                return Var(value=new_value)
            except KeyError as e:
                raise e
        elif type(self.value) is dict and type(other.value) in [int, float]:
            new_value = dict()
            for s, v in self.value.items():
                new_value[s] = v * other.value
            return Var(value=new_value)
        elif type(self.value) is dict and type(other) in [int, float]:
            new_value = dict()
            for s, v in self.value.items():
                new_value[s] = v * other
            return Var(value=new_value)
        elif type(self.value) in [int, float] and type(other) in [int, float]:
            return self.value * other

        elif type(self.value) in [int, float] and type(other.value) in [int, float]:
            return self.value * other.value
        else:
            raise TypeError("Operator * cannot be used between {} ({}) and {} ({})".format(type(other), other), type(self), self)

    def __sub__(self, other):
        if type(other) in [int, float]:
            if type(self.value) in [int, float]:
                return self.value - other
            else:
                raise TypeError("Operator - cannot be used between {} ({}) and {} ({})".format(type(self), self, type(other), other))
        elif type(other) in [Var]:
            if type(self.value) in [int, float]:
                if type(other.value) in [int, float]:
                    return self.value - other.value
                else:
                    new_value = dict()
                    for s, v in other.value.items():
                        new_value[s] = self.value - v
                    return Var(value=new_value)
            elif type(self.value) is dict:
                try:
                    new_value = dict()
                    for s, v in self.value.items():
                        new_value[s] = v - other.value[s]
                    return Var(value=new_value)
                except KeyError as e:
                    raise e
        else:
            raise TypeError("Operator - cannot be used between {} ({}) and {} ({})".format(type(self), self, type(other), other))

    def __truediv__(self, other):    

        if type(self.value) is dict and (type(other) in [Var] and type(other.value) is dict):
            try:
                new_value = dict()
                for s, v in self.value.items():
                    new_value[s] = v / other.value[s]
                return Var(value=new_value)
            except KeyError as e:
                raise e
        elif type(self.value) is dict and type(other.value) in [int, float]:
            new_value = dict()
            for s, v in self.value.items():
                new_value[s] = v / other.value
            return Var(value=new_value)
        elif type(self.value) is dict and type(other) in [int, float]:
            new_value = dict()
            for s, v in self.value.items():
                new_value[s] = v / other
            return Var(value=new_value)
        elif type(self.value) in [int, float] and type(other) in [int, float]:
            return self.value / other

        elif type(self.value) in [int, float] and type(other.value) in [int, float]:
            return self.value / other.value
        else:
            raise TypeError("Operator / cannot be used between {} ({}) and {} ({})".format(type(self), self, type(other), other))

    def __eq__(self, other):
        if self.value == other:
            return True
        else:
            return False
    
    def keys(self):
        return self.value.keys()

    def get_history(self, nsteps=None):
        if nsteps is None:
            return self.historical_values
        elif nsteps <= len(self.historical_values):
            return self.historical_values[-1*nsteps:]
        else:
            raise Exception("Arg nsteps out of range: {}".format(nsteps))

    def __repr__(self):
        if type(self.value) is dict:
            return 'SVar'+str(self.value)
        else:
            return str(self.value)

    def __setitem__(self, item, item_value):
        self.value[item] = item_value
        self.historical_values[item].append(item_value)

    def __getitem__(self, item):
        return self.value[item]


class GraphFunc(object):
    def __init__(self, xscale, yscale, ypts):
        self.xscale = xscale
        self.yscale = yscale
        self.ypts = ypts
        self.eqn = None
        
        from scipy.interpolate import interp1d

        self.xpts = np.linspace(self.xscale[0], self.xscale[1], num=len(self.ypts))
        self.interp_func = interp1d(self.xpts, self.ypts, kind='linear')

    def __call__(self, input):
        return self.interp_func(input)


class Conveyor(object):
    def __init__(self, length, eqn):
        self.length_time_units = length
        self.equation = eqn
        self.length_steps = None # to be decided at runtime
        self.initial_total = None # to be decided when initialising stocks
        self.conveyor = list()
    
    def initialize(self, length, value, leak_fraction=None):
        self.initial_total = value
        self.length_steps = length
        if leak_fraction is None or leak_fraction == 0:
            for _ in range(self.length_steps):
                self.conveyor.append(self.initial_total/self.length_steps)
        else:
            # print('Conveyor Initial Total:', self.initial_total)
            # print('Conveyor Leak fraction:', leak_fraction)
            # length_steps * output + nleaks * leak = initial_total
            # leak = output * (leak_fraction / length_in_steps)
            # ==> length_steps * output + n(length) * (output * (leak_fraction/length_in_steps)) = initial_total
            # print('Conveyor Length in steps:', self.length_steps)
            n_leak = 0
            for i in range(1, self.length_steps+1):
                n_leak += i
            # print('Conveyor Nleaks:', n_leak)
            output = self.initial_total / (self.length_steps + n_leak * (leak_fraction / self.length_steps))
            # print('Conveyor Output:', output)
            leak = output * (leak_fraction/self.length_steps)
            # print('Conveyor Leak:', leak)
            # generate slats
            for i in range(self.length_steps):
                self.conveyor.append(output + (i+1)*leak)
            self.conveyor.reverse()
        # print('Conveyor Initialised:', self.conveyor)

    def inflow(self, value):
        self.conveyor = [value] + self.conveyor

    def outflow(self):
        output = self.conveyor.pop()
        return output

    def level(self):
        return sum(self.conveyor)

    def leak_linear(self, leak_fraction):
        leaked = 0
        for i in range(self.length_steps):
            # keep slats non-negative
            if self.conveyor[i] > 0:
                already_leak_fraction = leak_fraction*((i+1)/self.length_steps) # 1+1: position indicator starting from 1
                remaining_fraction = 1-already_leak_fraction
                original = self.conveyor[i] / remaining_fraction
                i_leaked = original * leak_fraction / self.length_steps
                # print(i_leaked)
                if self.conveyor[i] - i_leaked < 0:
                    i_leaked = self.conveyor[i]
                    self.conveyor[i] = 0
                else:
                    self.conveyor[i] = self.conveyor[i] - i_leaked
                leaked = leaked + i_leaked
        # print('Leaked conveyor:', self.conveyor)
        # print('Leaked:', leaked)
        return leaked


class Structure(object):
    # equations
    def __init__(self, from_xmile=None):
        # dimensions
        self.var_dimensions = dict()
        
        # stocks
        self.stock_equations = dict()
        self.stock_positivity = dict()

        # connections
        self.stock_flows = dict()
        
        # flow
        self.flow_equations = dict()
        self.flow_positivity = dict()
        
        # aux
        self.aux_equations = dict()

        # variable_values
        self.variables = dict()

        # sim_specs
        self.sim_specs = {
            'initial_time': 0,
            'current_time': 0,
            'dt': 0.25,
            'simulation_time': 13,
            'time_units' :'Weeks'
        }

        # If the model is based on an XMILE file
        if from_xmile is not None:
            from pathlib import Path
            xmile_path = Path(from_xmile)
            if xmile_path.exists():
                with open(xmile_path) as f:
                    xmile_content = f.read().encode()
                    f.close()
                from bs4 import BeautifulSoup

                # read sim_specs
                sim_specs_root = BeautifulSoup(xmile_content, 'xml').find('sim_specs')
                time_units = sim_specs_root.get('time_units')
                sim_start = float(sim_specs_root.find('start').text)
                sim_stop = float(sim_specs_root.find('stop').text)
                sim_duration = sim_stop - sim_start
                sim_dt_root = sim_specs_root.find('dt')
                sim_dt = float(sim_dt_root.text)
                if sim_dt_root.get('reciprocal') == 'true':
                    sim_dt = 1/sim_dt
                
                self.sim_specs['initial_time'] = sim_start
                self.sim_specs['current_time'] = self.sim_specs['initial_time']
                self.sim_specs['dt'] = sim_dt
                self.sim_specs['simulation_time'] = sim_duration
                self.sim_specs['time_units'] = time_units

                # read subscritps
                try:
                    subscripts_root = BeautifulSoup(xmile_content, 'xml').find('dimensions')
                    dimensions = subscripts_root.findAll('dim')

                    dims = dict()
                    for dimension in dimensions:
                        name = dimension.get('name')
                        try:
                            size = dimension.get('size')
                            dims[name] = [str(i) for i in range(1, int(size)+1)]
                        except:
                            elems = dimension.findAll('elem')
                            elem_names = list()
                            for elem in elems:
                                elem_names.append(elem.get('name'))
                            dims[name] = elem_names
                    self.subscripts = dims
                except AttributeError:
                    pass
                
                # read variables
                variables_root = BeautifulSoup(xmile_content, 'xml').find('variables') # omit names in view
                stocks = variables_root.findAll('stock')
                flows = variables_root.findAll('flow')
                auxiliaries = variables_root.findAll('aux')
                
                # read graph functions
                def read_graph_func(var):
                    xscale = [
                        float(var.find('gf').find('xscale').get('min')),
                        float(var.find('gf').find('xscale').get('max'))
                    ]
                    yscale = [
                        float(var.find('gf').find('yscale').get('min')),
                        float(var.find('gf').find('yscale').get('max'))
                    ]
                    ypts = [float(t) for t in var.find('gf').find('ypts').text.split(',')]

                    equation = GraphFunc(xscale, yscale, ypts)
                    return equation

                # create var subscripted equation
                def subscripted_equation(var):
                    # print('Reading variable from XMILE: {}'.format(var.get('name')))
                    if var.find('dimensions'):
                        self.var_dimensions[var.get('name')] = list()
                        # print('Processing XMILE subscripted definition for:', var.get('name'))
                        var_dimensions = var.find('dimensions').findAll('dim')
                        # print('Found dimensions {}:'.format(var), var_dimensions)

                        var_dims = dict()
                        for dimension in var_dimensions:
                            name = dimension.get('name')
                            self.var_dimensions[var.get('name')].append(name)
                            # print(dimension)
                            # print(name)
                            var_dims[name] = dims[name]
                        var_subscripted_eqn = dict()

                        var_elements = var.findAll('element')
                        if len(var_elements) != 0:
                            for var_element in var_elements:

                                element_combination_text = var_element.get('subscript') # something like "1, First"
                                element_combination_text = self.process_subscript(element_combination_text) # "1, First" -> 1__cmm__First
                                # list_of_elements = list_of_elements_text.split(', ')
                                # tuple_of_elements = tuple(list_of_elements)
                                if var.find('conveyor'):
                                    equation = var_element.find('eqn').text
                                    length = var.find('len').text
                                    equation = Conveyor(length, equation)
                                elif var_element.find('gf'): 
                                    equation = read_graph_func(var_element)
                                    equation.eqn = var.find('eqn').text # subscripted graph function must share the same eqn
                                elif var_element.find('eqn'): # eqn is per element
                                    element_equation = var_element.find('eqn').text
                                    equation = element_equation
                                var_subscripted_eqn[element_combination_text] = equation

                        else: # all elements share the same equation
                            if var.find('conveyor'):
                                equation = var.find('eqn').text
                                length = int(var.find('len').text)
                                equation = Conveyor(length, equation)
                            elif var.find('gf'):
                                equation = read_graph_func(var)
                                equation.eqn = var.find('eqn').text
                            elif var.find('eqn'):
                                var_equation = var.find('eqn').text
                                equation = var_equation
                            else:
                                raise Exception('No meaningful definition found for variable {}'.format(var.get('name')))
                            
                            # fetch lists of elements and generate elements trings
                            element_combinations = product(*list(var_dims.values()))
                            # element_combination_texts = ['__cmm__'.join(cmb) for cmb in element_combinations]
                            element_combination_texts = [','.join(cmb) for cmb in element_combinations]
                            # print('ec', element_combination_texts)

                            for ect in element_combination_texts:
                                var_subscripted_eqn[ect] =equation
                        return(var_subscripted_eqn)
                    else:
                        self.var_dimensions[var.get('name')] = ['nosubscript']
                        # print('Processing XMILE definition for:', var.get('name'))
                        var_subscripted_eqn = dict()
                        if var.find('conveyor'):
                            equation = var.find('eqn').text
                            length = var.find('len').text
                            equation = Conveyor(length, equation)
                        elif var.find('gf'):
                            equation = read_graph_func(var)
                            equation.eqn = var.find('eqn').text
                        elif var.find('eqn'):
                            var_equation = var.find('eqn').text
                            equation = self.equation_handler(var_equation) # str -> float
                        # var_subscripted_eqn['nosubscript'] = equation
                        # return(var_subscripted_eqn)
                        return equation
                        

                # create stocks
                for stock in stocks:
                    
                    non_negative = False
                    if stock.find('non_negative'):
                        # print('nonnegstock', stock)
                        non_negative = True

                    inflows = stock.findAll('inflow')
                    outflows = stock.findAll('outflow')
                    self.add_stock(
                        self.name_handler(stock.get('name')), 
                        equation=subscripted_equation(stock), 
                        non_negative=non_negative,
                        in_flows=[f.text for f in inflows],
                        out_flows=[f.text for f in outflows]
                        )
                    
                # create auxiliaries
                for auxiliary in auxiliaries:
                    # self.add_aux(self.name_handler(auxiliary.get('name')), equation=auxiliary.find('eqn').text)
                    self.add_aux(self.name_handler(auxiliary.get('name')), equation=subscripted_equation(auxiliary))

                # create flows
                for flow in flows:
                    
                    # check if flow is a leakage flow
                    if flow.find('leak'):
                        leak = True
                    else:
                        leak = False

                    # check if can be negative
                    non_negative = False
                    if flow.find('non_negative'):
                        non_negative = True
                    self.add_flow(self.name_handler(flow.get('name')), equation=subscripted_equation(flow), leak=leak, non_negative=non_negative)

            else:
                raise Exception("Specified model file does not exist.")
    
    # utilities
    def name_handler(self, name):
        return name.replace(' ', '_').replace('\\n', '_')

    def equation_handler(self, equation):
        if type(equation) is str and equation.isdigit():
            equation = float(equation)
        return equation

    @staticmethod
    def process_subscript(subscript):
        # subscript = subscript.replace(',', '__cmm__').replace(' ', '')
        subscript = subscript.replace(',', ',').replace(' ', '')
        return subscript

    # model building
    def add_stock(self, name, equation, non_negative=True, in_flows=[], out_flows=[]):
        # if type(equation) is not dict:
        #     equation = {'nosubscript': equation}
        if type(equation) in [int, float] or (type(equation) is str and equation.isdigit()):
            equation = Var(equation)
        self.stock_equations[name] = equation
        self.stock_positivity[name] = non_negative
        connections = dict()
        if len(in_flows) != 0:
            connections['in'] = in_flows
        if len(out_flows) != 0:
            connections['out'] = out_flows
        self.stock_flows[name] = connections
    
    def add_flow(self, name, equation, leak=None, non_negative=False):
        # if type(equation) is not dict:
        #     equation = {'nosubscript': equation}
        if type(equation) in [int, float] or (type(equation) is str and equation.isdigit()):
            equation = Var(equation)
        self.flow_equations[name] = equation
        self.flow_positivity[name] = non_negative
    
    def add_aux(self, name, equation):
        # if type(equation) is not dict:
        #     equation = {'nosubscript': equation}
        if type(equation) in [int, float] or (type(equation) is str and equation.isdigit()):
            equation = Var(equation)
        self.aux_equations[name] = equation

    # initialise stocks
    def init_stocks(self):
        for var, equation in self.stock_equations.items():
            # subscripted calculation
            if type(equation) is dict:
                new_value = Var(dims=equation.keys())
                for sub, sub_equation in equation.items():
                    new_value[sub] = eval(str(sub_equation), None, self.variables)
            else:
                new_value = Var(eval(str(equation), None, self.variables))
            self.variables[var] = new_value
        self.variables_stock = copy.copy(self.variables) # a shallow copy of self.variables
    
    # calculate flows backwards
    def backward_calculate(self, var, run_time_variables):
        # print('vars', self.variables)
        # print('vars_runtime', self.variables_runtime)
        equation = (self.flow_equations | self.aux_equations)[var]
        if var not in run_time_variables.keys():
            if var not in self.variables.keys():
                if type(equation) is dict:
                    self.variables[var] = Var(dims=equation.keys())
                else:
                    self.variables[var] = Var()
            run_time_variables[var] = self.variables[var] # creat a reference to the actual Var(); runt_time_variables like a reference mask
        # print('calculating: {} = {}'.format(var, equation))
        
        if type(equation) is dict:
            for sub, sub_equation in equation.items():
                while True:
                    try:
                        ###
                        # TODO:Problem: eval() does not take contextual 'sub' so it cannot process equations like 1 - a[*], which yields a[1-*]
                        # Workaround: calculate all the elements and only take the one needed.
                        ###
                        new_value_sub = (eval(str(sub_equation), None, run_time_variables)) # calculate a new value and push it into historical values.
                        
                        if type(new_value_sub) is Var: # for the 1-a[*] condition
                            new_value_sub = new_value_sub[sub]
                        run_time_variables[var][sub] = new_value_sub
                        
                        # print('calculated: {}[{}]: {}'.format(var, sub, run_time_variables[var][sub]))
                        break
                    except NameError as e:
                        required_var = e.args[0].split("'")[1]
                        # print('Requiring:', required_var)
                        self.backward_calculate(required_var, run_time_variables)
        else:
            while True:
                try:
                    new_value = (eval(str(equation), None, run_time_variables))
                    run_time_variables[var].set(new_value)
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

    def calculate_isolated_vars(self):
        for var in ((self.flow_equations.keys() | self.aux_equations.keys()) - self.variables_runtime.keys()):
            self.backward_calculate(var, self.variables_runtime)
    
    # update stocks
    def update_stocks(self):
        for stock, connections in self.stock_flows.items():
            for direction, flows in connections.items():
                total_flow_effect = Var(copy=self.variables[stock])
                if direction == 'in':
                    for flow in flows:
                        total_flow_effect.set(total_flow_effect + self.variables[flow])
                elif direction == 'out':
                    for flow in flows:
                        total_flow_effect.set(total_flow_effect - self.variables[flow])
                else:
                    raise Exception("Invalid flow direction: {}".format(direction))
            self.variables[stock].set(self.variables[stock]+total_flow_effect * self.sim_specs['dt'])

    def simulate(self, t=None, dt=None):
        if t is None:
            t = self.sim_specs['simulation_time']
        if dt is None:
            dt = self.sim_specs['dt']
        steps = t/dt

        self.init_stocks()
        for s in range(int(steps)):
            self.sim_specs['current_time'] = s/dt
            # print('--step {}--'.format(s))
            self.calculate_flows()
            self.calculate_isolated_vars()
            self.update_stocks()
        self.calculate_flows()
        self.calculate_isolated_vars()

    def summary(self):
        print('\nSummary:\n')
        print('------------- Definitions -------------')
        pprint(self.stock_equations | self.flow_equations | self.aux_equations)
        print('')
        print('-------------  Sim specs  -------------')
        pprint(self.sim_specs)
        print('')
        print('-------------   Runtime   -------------')
        pprint(self.variables)
        print('')
        print('-------------   History   -------------')
        for k, v in self.variables.items():
            print(k)
            print('Current:', k, v)
            print('History:')
            for sub, vals in v.get_history().items():
                print(sub)
                print(vals)
                print('  Length:', len(vals))
            print('')
        print('')
        print


# test 1

# model = Structure()
# model.add_stock(name='tea_cup', equation=100, non_negative=True, in_flows=['change_tea_cup'])
# model.add_flow(name='change_tea_cup', equation='gap/at', non_negative=False)
# model.add_aux(name='room', equation=20)
# model.add_aux(name='gap', equation='room - tea_cup')
# model.add_aux(name='at', equation=5)
# model.summary()
# model.simulate()


# test 2
# model = Structure(from_xmile='BuiltinTestModels/Goal_gap.stmx')
# model.simulate()
# model.summary()

# test 3
# model = Structure(from_xmile='BuiltinTestModels/Goal_gap.stmx')
model = Structure(from_xmile='BuiltinTestModels/Goal_gap_array.stmx')
# model = Structure(from_xmile='BuiltinTestModels/Array_parallel_reference.stmx')

model.simulate()
model.summary()

fig, ax = plt.subplots()

for name, var in model.variables.items():
    for k, v in var.get_history().items():
        ax.plot(v, label='{}[{}]'.format(name,k))

ax.legend()
plt.show()


# TODO: isolated structures (not connected to a flow) (Created 8 November 2022) (Done 9 November 2022)
# TODO: 