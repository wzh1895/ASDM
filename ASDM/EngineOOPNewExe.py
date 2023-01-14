import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from pprint import pprint

from Parser import Parser
from Solver import Solver


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
        self.dimension_elements = dict()
        
        # stocks
        self.stock_equations = dict()
        self.stock_equations_parsed = dict()
        self.stock_positivity = dict()

        # connections
        self.stock_flows = dict()
        
        # flow
        self.flow_equations = dict()
        self.flow_equations_parsed = dict()
        self.flow_positivity = dict()
        
        # aux
        self.aux_equations = dict()
        self.aux_equations_parsed = dict()

        # variable_values
        self.var_history = dict()
        self.name_space = dict()

        # solver
        self.solver = Solver(
            dimension_elements=self.dimension_elements,
            name_space=self.name_space,
            var_history=self.var_history
            )

        # sequences
        self.seq_init_stocks = list()
        self.seq_flow_aux = list()

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
                    self.dimension_elements.update(dims) # need to use update here to do the 'True' assignment
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
                            # print('dim name:', name)
                            var_dims[name] = dims[name]
                        
                        var_subscripted_eqn = dict()
                        var_elements = var.findAll('element')
                        if len(var_elements) != 0:
                            for var_element in var_elements:

                                element_combination_text = var_element.get('subscript') # something like "1, First"
                                elements = self.process_subscript(element_combination_text) # "1, First" -> 1__cmm__First
                                # list_of_elements = element_combination_text.split(', ')
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
                                var_subscripted_eqn[elements] = equation

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

                            for ect in element_combinations:
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
                            equation = var.find('eqn').text
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

    @staticmethod
    def process_subscript(subscript):
        # subscript = subscript.replace(',', '__cmm__').replace(' ', '')
        subscript = tuple(subscript.replace(' ', '').split(','))
        return subscript

    # model building
    def add_stock(self, name, equation, non_negative=True, in_flows=[], out_flows=[]):
        self.stock_equations[name] = equation
        self.stock_positivity[name] = non_negative
        connections = dict()
        if len(in_flows) != 0:
            connections['in'] = in_flows
        if len(out_flows) != 0:
            connections['out'] = out_flows
        self.stock_flows[name] = connections
    
    def add_flow(self, name, equation, leak=None, non_negative=False):
        self.flow_equations[name] = equation
        self.flow_positivity[name] = non_negative
    
    def add_aux(self, name, equation):
        self.aux_equations[name] = equation

    def parse(self):
        parser = Parser()
        # string equation -> calculation tree
        for var, equation in self.stock_equations.items():
            parsed_equation = parser.parse(equation)
            self.stock_equations_parsed[var] = parsed_equation
            # print('{} equation: {}'.format(var, equation))
            # print('{} parsed_e: {}'.format(var, parsed_equation.nodes(data=True)))
        
        for var, equation in self.flow_equations.items():
            parsed_equation = parser.parse(equation)
            self.flow_equations_parsed[var] = parsed_equation
            # print('{} equation: {}'.format(var, equation))
            # print('{} parsed_e: {}'.format(var, parsed_equation.nodes(data=True)))
        
        for var, equation in self.aux_equations.items():
            parsed_equation = parser.parse(equation)
            self.aux_equations_parsed[var] = parsed_equation
            # print('{} equation: {}'.format(var, equation))
            # print('{} parsed_e: {}'.format(var, parsed_equation.nodes(data=True)))

    def iter_trace(self, seq, var, subscript=None):
        # print('iter_tracing:', var)
        self.var_history[var] = list()
        if subscript is not None:
            parsed_equation = (self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)[var][subscript]
            # print('i1')
        else:
            parsed_equation = (self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)[var]
            # print('i2')
        # print('type var', type(parsed_equation))
        if type(parsed_equation) is dict:
            for k, g in parsed_equation.items():
                self.iter_trace(seq=seq, var=var, subscript=k)
        else:
            # print('i3', seq)
            if var in seq:
                # print('i3.1', 'var in seq')
                seq.remove(var)
            seq.append(var)
            # print('i4', seq)
            leafs = [x for x in parsed_equation.nodes() if parsed_equation.out_degree(x)==0]
            for leaf in leafs:
                # print('i4.1')
                if parsed_equation.nodes[leaf]['operator'][0] in ['EQUALS', 'SPAREN']:
                    operands = parsed_equation.nodes[leaf]['operands']
                    # print('i5', operands)
                    if operands[0][0] == 'NUMBER': # if 'NUMBER' then pass, as numbers (e.g. 100) do not have a node
                        # print('i5.0') 
                        pass
                    elif operands[0][0] == 'NAME': # this refers to a variable like 'a'
                        # print('i5.1')
                        var_dependent = parsed_equation.nodes[leaf]['operands'][0][1]
                        # print('i5.2', var_dependent)
                        self.iter_trace(seq=seq, var=var_dependent)
                    elif operands[0][0] == 'FUNC': # this refers to a subscripted variable like 'a[ele1]'
                        # print('i5.3')
                        # need to find that 'SPAREN' node
                        var_dependent_node_id = parsed_equation.nodes[leaf]['operands'][0][2]
                        var_dependent = parsed_equation.nodes[var_dependent_node_id]['operands'][0][1]
                        # print('var_dependent2', var_dependent)
                        self.iter_trace(seq=seq, var=var_dependent)
    
    # calculation tree -> sequence of variables to evaluate
    def compile(self):
        # print('compiling stocks...')
        # sequence 1: initialising stocks
        for var, parsed_equation in self.stock_equations_parsed.items():
            if type(parsed_equation) is dict():
                for subscript in parsed_equation:
                    self.iter_trace(seq=self.seq_init_stocks, var=var, subscript=subscript)
            else:
                self.iter_trace(seq=self.seq_init_stocks, var=var)
        self.seq_init_stocks.reverse()

        # print('seq_init_stock', self.seq_init_stocks)
        # print('compiling flows...')
        # sequence 2: evaluating flows and aux. this sequence should not include stocks because:
        # (1) they are just inputs and already in name space 
        # (2) they are updated later when all flows are calculated
        for var in (self.flow_equations_parsed | self.aux_equations_parsed).keys(): # all flows & auxes, not just flows
            # print('var3', var)
            self.iter_trace(var=var, seq=self.seq_flow_aux)
        self.seq_flow_aux.reverse()
        # remove stocks from the sequence
        for var in self.seq_flow_aux:
            if var in self.stock_equations_parsed.keys():
                self.seq_flow_aux.remove(var)
        
        # print('seq_flow_aux', self.seq_flow_aux)
    
    def init_stocks(self):
        for var in self.seq_init_stocks:
            value = self.solver.calculate_node(self.stock_equations_parsed[var])
            self.name_space[var] = value
            self.var_history[var].append(value)

    def calculate_flow_auxes(self):
        for var in self.seq_flow_aux:
            # print('calculating var:', var)
            value = self.solver.calculate_node((self.flow_equations_parsed | self.aux_equations_parsed)[var])
            self.name_space[var] = value
            self.var_history[var].append(value)

    def update_stocks(self):
            for stock, connections in self.stock_flows.items():
                if type(self.stock_equations_parsed[stock]) is dict:
                    total_flow_effect = dict()
                    for kse in self.stock_equations_parsed[stock]:
                        total_flow_effect[kse] = 0
                else:
                    total_flow_effect = 0
                for direction, flows in connections.items():
                    if direction == 'in':
                        for flow in flows:
                            try:
                                total_flow_effect += self.name_space[flow]
                            except TypeError:
                                for kf, vf in self.name_space[flow].items():
                                    total_flow_effect[kf] += vf
                    elif direction == 'out':
                        for flow in flows:
                            try:
                                total_flow_effect -= self.name_space[flow]
                            except TypeError:
                                for kf, vf in self.name_space[flow].items():
                                    total_flow_effect[kf] -= vf
                    else:
                        raise Exception("Invalid flow direction: {}".format(direction))
                try:
                    value = self.name_space[stock] + total_flow_effect * self.sim_specs['dt']
                except TypeError:
                    value = dict()
                    for fk, fv in total_flow_effect.items():
                        value[fk] = self.name_space[stock][fk] + fv * self.sim_specs['dt']
                self.name_space[stock] = value
                self.var_history[stock].append(value)

    def simulate(self, time=None, dt=None):
        if time is None:
            time = self.sim_specs['simulation_time']
        if dt is None:
            dt = self.sim_specs['dt']
        steps = time/dt

        self.init_stocks()
        # print('name space 1:', self.name_space)
        for s in range(int(steps)):
            self.sim_specs['current_time'] = s/dt
            # print('--step {}--'.format(s))
            self.calculate_flow_auxes()
            self.update_stocks()
            # print('name space 2:', self.name_space)
        self.calculate_flow_auxes()
        # print('name space 3:', self.name_space)

    def summary(self):
        print('\nSummary:\n')
        print('------------- Definitions -------------')
        pprint(self.stock_equations | self.flow_equations | self.aux_equations)
        print('')
        print('-------------  Sim specs  -------------')
        pprint(self.sim_specs)
        print('')
        print('-------------   Runtime   -------------')
        pprint(self.name_space)
        print('')
        print('-------------   History   -------------')
        for k, v in self.name_space.items():
            print(k)
            print('Current:', k, v)
            print('History:')
            for var, history in self.var_history.items():
                print(var)
                print(history)
                print('  Length:', len(history))
            print('')
        print('')
        print


model = Structure(from_xmile='BuiltinTestModels/Goal_gap.stmx')
# model = Structure(from_xmile='BuiltinTestModels/Goal_gap_array.stmx')
# model = Structure(from_xmile='BuiltinTestModels/Array_parallel_reference.stmx')
# model = Structure(from_xmile='BuiltinTestModels/Array_cross_reference.stmx')
# model = Structure(from_xmile='BuiltinTestModels/Array_cross_reference_inference.stmx')

model.parse()
model.compile()
# model.simulate(time=1, dt=1)
model.simulate()
# model.summary()

fig, ax = plt.subplots()

for name, history in model.var_history.items():
    # print('var:', name) 
    # print('history:')
    # pprint(history)
    if type(history[0]) is dict:
        hh = {}
        for k in history[0].keys():
            hh[k] = [h[k] for h in history]
            ax.plot(hh[k], label='{}[{}]'.format(name, k))
    else:
        ax.plot(history, label='{}'.format(name))

ax.legend()
plt.show()
