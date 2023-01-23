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
        self.leak_conveyors = dict()
        self.outflow_conveyors = dict()
        
        # flow
        self.flow_positivity = dict()
        self.flow_equations = dict()
        self.flow_equations_parsed = dict()
        
        # aux
        self.aux_equations = dict()
        self.aux_equations_parsed = dict()

        # graph_functions
        self.graph_functions = dict()

        # variable_values
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
                    
                    is_conveyor = False
                    if stock.find('conveyor'):
                        is_conveyor = True

                    inflows = stock.findAll('inflow')
                    outflows = stock.findAll('outflow')
                    self.add_stock(
                        self.name_handler(stock.get('name')), 
                        equation=subscripted_equation(stock), 
                        non_negative=non_negative,
                        is_conveyor=is_conveyor,
                        in_flows=[f.text for f in inflows],
                        out_flows=[f.text for f in outflows],
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
    def add_stock(self, name, equation, non_negative=True, is_conveyor=False, in_flows=[], out_flows=[]):
        self.stock_equations[name] = equation
        self.stock_non_negative[name] = non_negative
        connections = dict()
        if len(in_flows) != 0:
            connections['in'] = in_flows
        if len(out_flows) != 0:
            connections['out'] = out_flows
        self.stock_flows[name] = connections
    
    def add_flow(self, name, equation, leak=None, non_negative=False):
        self.flow_positivity[name] = non_negative
        if leak:
            self.leak_conveyors[name] = None # to be filled when parsing the conveyor
        self.flow_equations[name] = equation
    
    def add_aux(self, name, equation):
        self.aux_equations[name] = equation

    def parse_0(self, equations, parsed_equations):
        for var, equation in equations.items():
            if type(equation) is GraphFunc:
                gfunc_name = 'GFUNC{}'.format(len(self.graph_functions))
                self.graph_functions[gfunc_name] = equation # just for length ... for now
                self.parser.functions.update({gfunc_name:gfunc_name}) # make name var also a function name and add it to the parser
                self.parser.patterns_custom_func.update(
                    {
                        gfunc_name+'__LPAREN__FUNC__RPAREN':{ # GraphFunc(var, expr, etc.)
                            'token':['FUNC', gfunc_name],
                            'operator':[gfunc_name],
                            'operand':['FUNC']
                        },
                    }
                )
                self.solver.custom_functions.update({gfunc_name:equation})
                equation = gfunc_name+'('+ equation.eqn + ')'  # make equation into form like var(eqn), 
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
                    if flow in self.leak_conveyors:
                        self.conveyors[var]['leakflow'][flow] = 0
                        self.leak_conveyors[flow] = var
                    else:
                        self.conveyors[var]['outputflow'].append(flow)
                        self.outflow_conveyors[flow] = var
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
        # self.parse_0(self.flow_leak_equations, self.flow_leak_equations_parsed)
        self.parse_0(self.aux_equations, self.aux_equations_parsed)

    def iter_trace(self, seq, var, subscript=None, mode=None):
        # print('  Iter_tracing:', var)
        if var in self.env_variables: # var is a built-in variable like TIME
            pass # skip, as they will already be in name_space by the time of simulation

        # var is a leakflow or an outflow from a conveyor. In this case the conveyor needs to be initialised
        elif var in self.leak_conveyors:
            # requiring a leakflow's value triggers the calculation of its connected conveyor

            # if mode is not 'leak_frac', something other than the conveyor is requiring the leak_flow; then conveyor needs to be calculated. Otherwise it is the conveyor that requires it
            if mode != 'leak_frac':  
                if self.leak_conveyors[var] not in seq:
                    self.iter_trace(seq=seq, var=self.leak_conveyors[var], subscript=subscript)
                else:
                    # the conveyor, which is pre-required for this leak_flow, needs to be move forward so that it is calculated before this leak_flow
                    seq.remove(self.leak_conveyors[var])
                    self.iter_trace(seq=seq, var=self.leak_conveyors[var], subscript=subscript)
            # leakflow should not be added to seq, as
            # (1) their equation yiels leakfraction, not the actualy flow value;
            # (2) they are calculated when the conveyor is initialised or updated
            # However, leak_fraction is calculated using leakflow's equation. 
            leafs = [x for x in self.flow_equations_parsed[var].nodes() if self.flow_equations_parsed[var].out_degree(x)==0]
            for leaf in leafs:
                # print('i4.1')
                if self.flow_equations_parsed[var].nodes[leaf]['operator'][0] in ['EQUALS', 'SPAREN']:
                    operands = self.flow_equations_parsed[var].nodes[leaf]['operands']
                    # print('i5', operands)
                    if operands[0][0] == 'NUMBER': # if 'NUMBER' then pass, as numbers (e.g. 100) do not have a node
                        # print('i5.0') 
                        pass
                    elif operands[0][0] == 'NAME': # this refers to a variable like 'a'
                        # print('i5.1', operands[0][0])
                        var_dependent = self.flow_equations_parsed[var].nodes[leaf]['operands'][0][1]
                        # print('i5.2', var_dependent)
                        self.iter_trace(seq=seq, var=var_dependent)
                    elif operands[0][0] == 'FUNC': # this refers to a subscripted variable like 'a[ele1]'
                        # print('i5.3')
                        # need to find that 'SPAREN' node
                        var_dependent_node_id = self.flow_equations_parsed[var].nodes[leaf]['operands'][0][2]
                        var_dependent = self.flow_equations_parsed[var].nodes[var_dependent_node_id]['operands'][0][1]
                        # print('var_dependent2', var_dependent)
                        self.iter_trace(seq=seq, var=var_dependent)

        elif var in self.outflow_conveyors:
            # requiring an outflow's value triggers the calculation of its connected conveyor
            self.iter_trace(seq=seq, var=self.outflow_conveyors[var], subscript=subscript)
            # convoutflows should not be added to seq, as they are calculated when the conveyor is initialised or updated
        
        elif var in self.conveyors: # var is a conveyor (stock)
            # self.var_history[var] = list()
            # These two variables only need to be calculated once to initialise the conveyor 
            if var in seq: # add conveyor to seq
                seq.remove(var)
            seq.append(var)
            
            for _, parsed_eqn in self.stock_equations_parsed[var].items(): # ['len', 'val']
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
                            self.iter_trace(seq=seq, var=var_dependent)
                        elif operands[0][0] == 'FUNC': # this refers to a subscripted variable like 'a[ele1]'
                            # print('i5.3')
                            # need to find that 'SPAREN' node
                            var_dependent_node_id = parsed_eqn.nodes[leaf]['operands'][0][2]
                            var_dependent = parsed_eqn.nodes[var_dependent_node_id]['operands'][0][1]
                            # print('var_dependent2', var_dependent)
                            self.iter_trace(seq=seq, var=var_dependent)
            
            # conveyor also requries leak_frac
            for leak_flow in self.conveyors[var]['leakflow'].keys():
                if leak_flow in seq:
                    seq.remove(leak_flow)
                seq.append(leak_flow)
                self.iter_trace(seq=seq, var=leak_flow, mode='leak_frac')


        else: # var is a user-defined varialbe
            # self.var_history[var] = list()
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
        # print('Start compiling...')
        for var in (self.stock_equations_parsed | self.aux_equations_parsed | self.flow_equations_parsed).keys():
            # print('Compiling:', var)
            self.iter_trace(var=var, seq=self.seq_flow_aux)
        self.seq_flow_aux.reverse()
    
    def calculate_variables(self):
        for var in self.seq_flow_aux:    
            # print('Engine Calculating:', var, 'current name space', self.name_space)
            if var in self.conveyors:
                if not self.conveyors[var]['conveyor'].is_initialized:
                    conveyor_length = self.solver.calculate_node(self.stock_equations_parsed[var]['len'])
                    length_steps = int(conveyor_length/self.sim_specs['dt'])
                    conveyor_value = self.solver.calculate_node(self.stock_equations_parsed[var]['val'])
                    leak_flows = self.conveyors[var]['leakflow']
                    if len(leak_flows) == 0:
                        leak_fraction = 0
                    else:
                        for leak_flow in leak_flows.keys():
                            # leak_flows[leak_flow] = self.solver.calculate_node(self.flow_equations_parsed[leak_flow])
                            leak_fraction = leak_flows[leak_flow] # TODO multiple leakflows
                    self.conveyors[var]['conveyor'].initialize(length_steps, conveyor_value, leak_fraction)
                
                # level
                value = self.conveyors[var]['conveyor'].level()

                self.name_space[var] = value
                # leak
                for leak_flow, leak_fraction in self.conveyors[var]['leakflow'].items():
                    # leaked_value = conveyor['conveyor'].leak_linear_value(self.solver.calculate_node(self.flow_equations_parsed[flow]))
                    leaked_value = self.conveyors[var]['conveyor'].leak_linear()
                    self.name_space[leak_flow] = leaked_value / self.sim_specs['dt'] # TODO: we should also consider when leak flows are subscripted
                    # self.var_history[leak_flow].append(leaked_value / self.sim_specs['dt'])
                # out
                for outputflow in self.conveyors[var]['outputflow']:
                    # outflow_value = conveyor['conveyor'].outflow_value()
                    outflow_value = self.conveyors[var]['conveyor'].outflow()
                    self.name_space[outputflow] = outflow_value / self.sim_specs['dt']
                    # self.var_history[outputflow].append(outflow_value / self.sim_specs['dt'])
            else:
                value = self.solver.calculate_node((self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)[var])
                if var in self.leak_conveyors.keys():
                    self.conveyors[self.leak_conveyors[var]]['leakflow'][var] = value
                else:
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

                # remove stock from seq
                if stock in self.seq_flow_aux:
                    self.seq_flow_aux.remove(stock)

        for conveyor_name, conveyor in self.conveyors.items(): # Stock is a Conveyor
            total_flow_effect = 0
            connections = self.stock_flows[conveyor_name]
            for direction, flows in connections.items():
                if direction == 'in':
                    for flow in flows:
                        total_flow_effect += self.name_space[flow]

            # in
            conveyor['conveyor'].inflow(total_flow_effect * self.sim_specs['dt'])
            
            # level
            value = conveyor['conveyor'].level()
            self.name_space[conveyor_name] = value
                
    def simulate(self, time=None, dt=None):
        if time is None:
            time = self.sim_specs['simulation_time']
        if dt is None:
            dt = self.sim_specs['dt']
        steps = time/dt

        for s in range(int(steps)+1):
            # print('--step {}--'.format(s))
            self.calculate_variables()

            self.time_slice[self.sim_specs['current_time']] = deepcopy(self.name_space)

            self.sim_specs['current_time'] += dt
            self.name_space['TIME'] += dt
            
            self.update_stocks()

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
    # model = Structure(from_xmile='BuiltInTestModels/Conveyor_leakage1.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/Conveyor_leakage2.stmx')
    # model = Structure(from_xmile='BuiltInTestModels/Conveyor_initialisation.stmx')

    ### Production Models ###

    # model = Structure(from_xmile='TestModels/Elective Recovery Model.stmx')
    model = Structure(from_xmile='TestModels/Elective Recovery Model_renamed.stmx')
    # model = Structure(from_xmile='/Users/jonker/Library/CloudStorage/OneDrive-UniversityofStrathclyde/PhD_Progress/CaseStudies/CaseStudy1/CaseStudy1Codes/CaseStudy1Models/CS1SFD3.stmx')

    model.parse()
    model.compile()
    # model.simulate(time=1, dt=1)
    model.simulate()
    # model.summary()

    fig, ax = plt.subplots()

    plot_history = dict()
    for time, slice in model.time_slice.items():
        # print('time:', time) 
        # print('slice:')
        # pprint(slice)
        for var, value in slice.items():
            if type(value) is dict:
                for sub, subvalue in value.items():
                    if var+'[{}]'.format(', '.join(sub)) in plot_history.keys():
                        plot_history[var+'[{}]'.format(', '.join(sub))].append(subvalue)
                    else:
                        plot_history[var+'[{}]'.format(', '.join(sub))] = [subvalue]
                # if var in model.var_history:
                #     del model.var_history[var]
            else:
                if var in plot_history:
                    plot_history[var].append(value)
                else:
                    plot_history[var] = [value]

    vars_to_view = [
        # 'thirteen_wk_wait_for_urgent_treatment', 
        # 'Negative_test_results', 
        # 'COVID_modified_percent_urgent', 
        # 'Undergoing_diagnostic_tests',
        # 'Positive_test_results_urgent',
        # 'Less_than_6mth_to_urgent',
        # 'Between_6_to_12mth_wait_to_urgent',
        # 'Between_12_to_24mth_wait_to_urgent',
        # 'Urgent_treatment',
        # 'Total_treatment_capacity',
        # 'Routine_treatment',
        # 'Net_COVID_induced_changes_in_underlying_health_needs'
        ]

    for var, history in plot_history.items():
        # pprint(var)
        if (len(vars_to_view)) == 0 or (len(vars_to_view) != 0 and var in vars_to_view):
            # print('    ', var, ':', len(history))
            if type(history[0]) is dict:
                hh = {}
                for k in history[0].keys():
                    hh[k] = [h[k] for h in history]
                    ax.plot(hh[k], label='{}[{}]'.format(var, k))
                # print(hh)
            else:
                ax.plot(history, label='{}'.format(var))
                # print(history)

    ax.legend()
    plt.show()


    import pandas as pd
    df_asdm = pd.DataFrame.from_dict(plot_history)
    df_asdm.to_csv('asdm.csv')