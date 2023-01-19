import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from pprint import pprint

from Parser import Parser
from Solver import Solver
from Functions import GraphFunc, Conveyor

from copy import deepcopy

class Structure(object):
    # equations
    def __init__(self, from_xmile=None):
        # sim_specs
        self.sim_specs = {
            'initial_time': 0,
            'current_time': 0,
            'dt': 0.25,
            'simulation_time': 13,
            'time_units' :'Weeks'
        }

        # dimensions
        self.var_dimensions = dict()
        self.dimension_elements = dict()
        
        # stocks
        self.stock_equations = dict()
        self.stock_equations_parsed = dict()
        self.stock_non_negative = dict()

        # discrete variables
        self.conveyors = dict()

        # connections
        self.stock_flows = dict()
        
        # flow
        self.flow_equations = dict()
        self.flow_equations_parsed = dict()
        self.flow_positivity = dict()
        self.flow_leak = dict()
        
        # aux
        self.aux_equations = dict()
        self.aux_equations_parsed = dict()

        # variable_values
        self.var_history = {'TIME':[]}
        self.name_space = dict()
        self.time_slice = dict()

        # env variables
        self.env_variables = {
            'TIME': 0
        }
        self.name_space.update(self.env_variables)

        # parser
        self.parser = Parser()

        # solver
        self.solver = Solver(
            sim_specs=self.sim_specs,
            dimension_elements=self.dimension_elements,
            name_space=self.name_space,
            )

        # sequences
        self.seq_init_conveyors = list()
        self.seq_init_stocks = list()
        self.seq_flow_aux = list()


        # custom functions
        self.custom_functions = {}

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
                self.sim_specs['current_time'] = sim_start
                self.env_variables['TIME'] = sim_start
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
        self.stock_non_negative[name] = non_negative
        connections = dict()
        if len(in_flows) != 0:
            connections['in'] = in_flows
        if len(out_flows) != 0:
            connections['out'] = out_flows
        self.stock_flows[name] = connections
    
    def add_flow(self, name, equation, leak=None, non_negative=False):
        self.flow_equations[name] = equation
        self.flow_positivity[name] = non_negative
        if leak:
            self.flow_leak[name] = True
        else:
            self.flow_leak[name] = False
    
    def add_aux(self, name, equation):
        self.aux_equations[name] = equation

    def parse_0(self, equations, parsed_equations):
        for var, equation in equations.items():
            if type(equation) is GraphFunc:
                self.parser.functions.update({var:var}) # make name var also a function name and add it to the parser
                self.parser.patterns_custom_func.update(
                    {
                        var+'__LPAREN__FUNC__RPAREN':{ # GraphFunc(var, expr, etc.)
                            'token':['FUNC',var],
                            'operator':[var],
                            'operand':['FUNC']
                        },
                    }
                )
                self.solver.custom_functions.update({var:equation})
                equation = var+'('+ equation.eqn + ')'  # make equation into form like var(eqn), 
                                                # where eqn is the euqaiton whose outcome is the input to GraphFunc var()
                                                # this is also how Vensim handles GraphFunc
                parsed_equation = self.parser.parse(equation)
                parsed_equations[var] = parsed_equation
                # print('{} equation: {}'.format(var, equation))
                # print('{} parsed_e: {}'.format(var, parsed_equation.nodes(data=True)))
            
            elif type(equation) is Conveyor: # TODO we should also consider arrayed conveyors
                self.conveyors[var] = {
                    'conveyor': equation, # the Conveyor object
                    'inflow': self.stock_flows[var]['in'],
                    'outflow': self.stock_flows[var]['out'],
                    'outputflow': [], # this list should have a fixed length of 1
                    'leakflow': {},
                }
                for flow in self.stock_flows[var]['out']:
                    if self.flow_leak[flow]:
                        self.conveyors[var]['leakflow'][flow] = 0
                    else:
                        self.conveyors[var]['outputflow'].append(flow)
                equation_length = equation.length_time_units # this is the equation for its length
                parsed_equation_len = self.parser.parse(equation_length)

                equation_init_value = equation.equation # this is the equation for its initial value
                parsed_equation_val = self.parser.parse(equation_init_value)

                parsed_equations[var] = {
                    'len': parsed_equation_len,
                    'val': parsed_equation_val
                }

            else:
                parsed_equation = self.parser.parse(equation)
                parsed_equations[var] = parsed_equation
                # print('{} equation: {}'.format(var, equation))
                # print('{} parsed_e: {}'.format(var, parsed_equation.nodes(data=True)))

    def parse(self):
        # string equation -> calculation tree

        self.parse_0(self.stock_equations, self.stock_equations_parsed)
        self.parse_0(self.flow_equations, self.flow_equations_parsed)
        self.parse_0(self.aux_equations, self.aux_equations_parsed)

    def iter_trace(self, seq, var, subscript=None):
        # print('iter_tracing:', var)
        if var in self.env_variables: # var is a built-in variable like TIME
            pass # skip, as they will already be in name_space by the time of simulation
        elif var in self.conveyors: # var is a conveyor (stock)
            self.var_history[var] = list()
            # parsed_equation_len = self.stock_equations_parsed[var]['len']
            # parsed_equation_val = self.stock_equations_parsed[var]['val']
            for eqn, parsed_eqn in self.stock_equations_parsed[var].items():
                leafs = [x for x in parsed_eqn.nodes() if parsed_eqn.out_degree(x)==0]
                for leaf in leafs:
                    # print('i4.1')
                    if parsed_eqn.nodes[leaf]['operator'][0] in ['EQUALS', 'SPAREN']:
                        operands = parsed_eqn.nodes[leaf]['operands']
                        # print('i5', operands)
                        if operands[0][0] == 'NUMBER': # if 'NUMBER' then pass, as numbers (e.g. 100) do not have a node
                            # print('i5.0') 
                            pass
                        elif operands[0][0] == 'NAME': # this refers to a variable like 'a'
                            # print('i5.1', operands[0][0])
                            var_dependent = parsed_eqn.nodes[leaf]['operands'][0][1]
                            # print('i5.2', var_dependent)
                            self.iter_trace(seq=self.seq_init_conveyors, var=var_dependent)
                        elif operands[0][0] == 'FUNC': # this refers to a subscripted variable like 'a[ele1]'
                            # print('i5.3')
                            # need to find that 'SPAREN' node
                            var_dependent_node_id = parsed_eqn.nodes[leaf]['operands'][0][2]
                            var_dependent = parsed_eqn.nodes[var_dependent_node_id]['operands'][0][1]
                            # print('var_dependent2', var_dependent)
                            self.iter_trace(seq=self.seq_init_conveyors, var=var_dependent)

        else: # var is a user-defined varialbe
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
                # print('i3', var)
                if var in seq:
                    # print('i3.1', 'var {} in seq'.format(var), var in seq)
                    seq.remove(var)
                seq.append(var)
                # print('i3.2', 'var {} in seq'.format(var), var in seq)
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
        self.seq_init_conveyors.reverse()
        # print('seq_init_conveyors', self.seq_init_conveyors)

        # print('compiling flows...')
        # sequence 2: evaluating flows and aux. this sequence should not include stocks because:
        # (1) they are just inputs and already in name space 
        # (2) they are updated later when all flows are calculated
        for var in (self.flow_equations_parsed | self.aux_equations_parsed).keys(): # all flows & auxes, not just flows
            # print('var3', var)
            self.iter_trace(var=var, seq=self.seq_flow_aux)
        self.seq_flow_aux.reverse()
        
        # # remove stocks from seq_init_stocks
        # for var in self.stock_equations_parsed.keys():
        #     if var in self.seq_init_stocks:
        #         self.seq_init_stocks.remove(var)
        #         print('removed 1:', var)

        # remove stocks from seq_flow_aux
        for var in self.seq_flow_aux:
            if var in self.stock_equations_parsed.keys():
                self.seq_flow_aux.remove(var)
        # print('seq_flow_aux', self.seq_flow_aux)

        # remove outflows and leakflows that are connected to a conveyor from seq_flow_aux, 
        # as they will be calculated separately
        for c, con in self.conveyors.items():
            for f in con['outflow']+list(con['leakflow'].keys()):
                if f in self.seq_init_stocks:
                    self.seq_init_stocks.remove(f)
                if f in self.seq_flow_aux:
                    self.seq_flow_aux.remove(f)
        
        # # remove variables used in init_stocks and init_conveyors from seq_flow_aux
        # for var in self.seq_init_stocks + self.seq_init_conveyors:
        #     if var in self.seq_flow_aux:
        #         self.seq_flow_aux.remove(var)
        #         print('removed 5', var)
                
    
    def init_stocks(self):
        
        # calculate the variables needed for initialising conveyors
        for var in self.seq_init_conveyors:
            value = self.solver.calculate_node((self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)[var]) # stock init might need flow or aux equations
            self.name_space[var] = value
            # self.var_history[var].append(value)

        # initialise the Conveyors and calculate their outflows and leaks
        for conveyor_name, conveyor in self.conveyors.items():
            conveyor_length = self.solver.calculate_node(self.stock_equations_parsed[conveyor_name]['len'])
            length_steps = int(conveyor_length/self.sim_specs['dt'])
            conveyor_value = self.solver.calculate_node(self.stock_equations_parsed[conveyor_name]['val'])
            leak_flows = conveyor['leakflow']
            
            if len(leak_flows) != 0:
                for leak_flow in leak_flows.keys():
                    leak_flows[leak_flow] = self.solver.calculate_node(self.flow_equations_parsed[leak_flow])
                    leak_fraction = leak_flows[leak_flow]
                    conveyor['conveyor'].initialize(length_steps, conveyor_value, leak_fraction)
            else:
                conveyor['conveyor'].initialize(length_steps, conveyor_value, leak_fraction=0)
            # level
            self.name_space[conveyor_name] = conveyor['conveyor'].level()
            # self.var_history[conveyor_name].append(conveyor['conveyor'].level())
            # leak
            for leak_flow, leak_fraction in conveyor['leakflow'].items():
                # leaked_value = conveyor['conveyor'].leak_linear_value(self.solver.calculate_node(self.flow_equations_parsed[flow]))
                leaked_value = conveyor['conveyor'].leak_linear(leak_fraction)
                self.name_space[leak_flow] = leaked_value / self.sim_specs['dt'] # TODO: we should also consider when leak flows are subscripted
                # self.var_history[leak_flow].append(leaked_value)
            # out
            for flow in conveyor['outputflow']:
                # outflow_value = conveyor['conveyor'].outflow_value()
                outflow_value = conveyor['conveyor'].outflow()
                self.name_space[flow] = outflow_value / self.sim_specs['dt']
                # self.var_history[flow].append(outflow_value)

        # initialise the 'SD' sort of stocks
        for var in self.seq_init_stocks:
            value = self.solver.calculate_node((self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)[var]) # stock init might need flow or aux equations
            self.name_space[var] = value
            # self.var_history[var].append(value)

    def calculate_flow_auxes(self):
        for var in self.seq_flow_aux:
            # print('calculating var:', var)
            value = self.solver.calculate_node((self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)[var])
            if var in self.flow_equations:
                if self.flow_positivity[var]:
                    if type(value) is dict:
                        for sub, subval in value.items():
                            if subval < 0:
                                value[sub] = 0
                    else:
                        if value < 0:
                            value = 0            
            self.name_space[var] = value
            # self.var_history[var].append(value)

    def update_stocks(self):
        for stock, connections in self.stock_flows.items(): 
            if stock not in self.conveyors: # stock is a normal SD stock
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
                # self.var_history[stock].append(value)

        for conveyor_name, conveyor in self.conveyors.items(): # Stock is a Conveyor
            # if type(self.stock_equations_parsed[stock]) is dict: # it must be dict {'len':xxx, 'val':xxx}
            #     total_flow_effect = dict()
            #     for kse in self.stock_equations_parsed[stock]:
            #         total_flow_effect[kse] = 0
            # else:
            #     total_flow_effect = 0
            total_flow_effect = 0
            connections = self.stock_flows[conveyor_name]
            for direction, flows in connections.items():
                if direction == 'in':
                    for flow in flows:
                        # try:
                        #     total_flow_effect += self.name_space[flow]
                        # except TypeError:
                        #     for kf, vf in self.name_space[flow].items():
                        #         total_flow_effect[kf] += vf
                        total_flow_effect += self.name_space[flow]

            # in
            conveyor['conveyor'].inflow(total_flow_effect * self.sim_specs['dt'])
            
            # level
            value = conveyor['conveyor'].level()
            self.name_space[conveyor_name] = value
            # self.var_history[conveyor_name].append(value)
            
            # leak
            for leak_flow, leak_fraction in conveyor['leakflow'].items(): # TODO: multiple leak flows
                v_leak = conveyor['conveyor'].leak_linear(leak_fraction) #TODO we should support multiple leak flows
                self.name_space[leak_flow] = v_leak / self.sim_specs['dt']
                # self.var_history[leak_flow].append(v_leak)
            
            # out
            v_out = conveyor['conveyor'].outflow()
            self.name_space[conveyor['outputflow'][0]] = v_out / self.sim_specs['dt']
            # self.var_history[conveyor['outputflow'][0]].append(v_out)
            
            # try:
            #     value = self.name_space[stock] + total_flow_effect * self.sim_specs['dt']
            # except TypeError:
            #     value = dict()
            #     for fk, fv in total_flow_effect.items():
            #         value[fk] = self.name_space[stock][fk] + fv * self.sim_specs['dt']
                
    def simulate(self, time=None, dt=None):
        if time is None:
            time = self.sim_specs['simulation_time']
        if dt is None:
            dt = self.sim_specs['dt']
        steps = time/dt

        self.init_stocks()
        # print('name space 1:', self.name_space)
        for s in range(int(steps)):
            # print('--step {}--'.format(s))
            self.calculate_flow_auxes()

            self.time_slice[self.sim_specs['current_time']] = deepcopy(self.name_space)

            self.sim_specs['current_time'] += dt
            self.name_space ['TIME'] += dt
            
            self.update_stocks()

            # self.name_space = dict()
            # self.name_space.update(self.env_variables)
            
            # print('name space 2:', self.name_space)
        self.calculate_flow_auxes()
        self.time_slice[self.sim_specs['current_time']] = deepcopy(self.name_space)

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


if __name__ == '__main__':

    #### Test Models ###

    # model = Structure(from_xmile='BuiltinTestModels/Min_Max.stmx')
    # model = Structure(from_xmile='BuiltinTestModels/Non-negative_stocks.stmx')
    # model = Structure(from_xmile='BuiltinTestModels/Isolated_var.stmx')

    # model = Structure(from_xmile='BuiltinTestModels/Goal_gap.stmx')

    # model = Structure(from_xmile='BuiltinTestModels/Goal_gap_array.stmx')
    # model = Structure(from_xmile='BuiltinTestModels/Array_parallel_reference.stmx')
    # model = Structure(from_xmile='BuiltinTestModels/Array_cross_reference.stmx')
    # model = Structure(from_xmile='BuiltinTestModels/Array_cross_reference_inference.stmx')
    
    # model = Structure(from_xmile='BuiltInTestModels/Built_in_vars.stmx')
    
    # model = Structure(from_xmile='BuiltInTestModels/Logic.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/IF_THEN_ELSE.stmx')
    
    # model = Structure(from_xmile='BuiltInTestModels/Graph_function.stmx')

    # model = Structure(from_xmile='BuiltInTestModels/INIT.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/Delays.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/Delays2.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/History.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/Time_related_functions.stmx')
    
    # model = Structure(from_xmile='BuiltInTestModels/Conveyor.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/Conveyor_leakage.stmx')

    ### Production Models ###
    # model = Structure(from_xmile='TestModels/TempTest.stmx')
    # model = Structure(from_xmile='TestModels/TempTest2.stmx')
    # model = Structure(from_xmile='TestModels/TempTest3.stmx')

    # model = Structure(from_xmile='TestModels/Elective Recovery Model.stmx')
    model = Structure(from_xmile='TestModels/Elective Recovery Model_renamed.stmx')

    model.parse()
    model.compile()
    # model.simulate(time=1, dt=1)
    model.simulate()
    # model.summary()

    fig, ax = plt.subplots()

    # pprint(model.var_history)

    for time, slice in model.time_slice.items():
        # print('time:', time) 
        # print('slice:')
        # pprint(slice)
        for var, value in slice.items():
            if type(value) is dict:
                for sub, subvalue in value.items():
                    if var+'[{}]'.format(sub[0]) in model.var_history.keys():
                        model.var_history[var+'[{}]'.format(sub[0])].append(subvalue)
                    else:
                        model.var_history[var+'[{}]'.format(sub[0])] = [subvalue]
                if var in model.var_history:
                    del model.var_history[var]
            else:
                model.var_history[var].append(value)

    ## ERM ###

    vars_to_view = [
        'thirteen_wk_wait_for_urgent_treatment', 
        # 'Negative_test_results', 
        # 'COVID_modified_percent_urgent', 
        # 'Undergoing_diagnostic_tests',
        # 'Positive_test_results_urgent',
        # 'Less_than_6mth_to_urgent',
        # 'Between_6_to_12mth_wait_to_urgent',
        # 'Between_12_to_24mth_wait_to_urgent',
        # 'Urgent_treatment',
        # 'Total_treatment_capacity',
        'Routine_treatment'
        ]

    for var, history in model.var_history.items():
        if var in vars_to_view:
            # print(var, ':')
            # print('   ', history)
            pass
        
        if var in vars_to_view:
            if type(history[0]) is dict:
                hh = {}
                for k in history[0].keys():
                    hh[k] = [h[k] for h in history]
                    ax.plot(hh[k], label='{}[{}]'.format(var, k))
            else:
                ax.plot(history, label='{}'.format(var))

    ## OTHER ###

    # for var, history in model.var_history.items():
    #     print(var, ':')
    #     print('   ', history)
    
    #     if type(history[0]) is dict:
    #         hh = {}
    #         for k in history[0].keys():
    #             hh[k] = [h[k] for h in history]
    #             ax.plot(hh[k], label='{}[{}]'.format(var, k))
    #     else:
    #         ax.plot(history, label='{}'.format(var))

    ax.legend()
    plt.show()

    # pprint(model.var_history)

    # import pandas as pd
    # df_asdm = pd.DataFrame.from_dict(model.var_history)
    # df_asdm.to_csv('erm_asdm.csv')