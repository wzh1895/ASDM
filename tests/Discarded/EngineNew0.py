from re import sub
import networkx as nx
import numpy as np
from tqdm.notebook import tnrange
from copy import copy, deepcopy
import re


class Var(object):
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.custom_attrs = set()
    
    # def __repr__(self):
    #     repr_dict = dict(zip(self.custom_attrs, [self.__dict__[a] for a in self.custom_attrs]))
    #     # representation = ''
    #     # for k, v in repr_dict.items():
    #     #     representation = representation + k + ':'+ str(v) +'\n'
    #     representation = str(repr_dict)
    #     return 'Var:' + representation

    def __setitem__(self, item, value): # support for a[1] = b
        self.custom_attrs.add(item)
        setattr(self, item, value)

    def __getitem__(self, item):
        return getattr(self, item) # support for b = a[1]
    
    def __add__(self, other):
        if type(other) is self.__class__:
            if set(self.dimensions) != set(other.dimensions):
                raise ValueError('The two Vars do not have the same dimensions')
            if self.custom_attrs != other.custom_attrs:
                raise ValueError('The two Vars do not have the same elements')
            new_value = self.__class__(self.dimensions)
            for attr in self.custom_attrs:
                new_value[attr] = float(self.__dict__[attr]) + float(other[attr])
            return new_value
        
        elif other == 0:
            new_value = copy(self)
            return new_value

    def __radd__(self, other):
        if type(other) is self.__class__:
            if set(self.dimensions) != set(other.dimensions):
                raise ValueError('The two Vars do not have the same dimensions')
            if self.custom_attrs != other.custom_attrs:
                raise ValueError('The two Vars do not have the same elements')
            new_value = self.__class__(self.dimensions)
            for attr in self.custom_attrs:
                new_value[attr] = float(self.__dict__[attr]) + float(other[attr])
            return new_value

        elif other == 0:
            new_value = copy(self)
            return new_value
    
    def __iadd__(self, other):
        if type(other) is self.__class__:
            if set(self.dimensions) != set(other.dimensions):
                raise ValueError('The two Vars do not have the same dimensions')
            if self.custom_attrs != other.custom_attrs:
                raise ValueError('The two Vars do not have the same elements')
            new_value = self.__class__(self.dimensions)
            for attr in self.custom_attrs:
                new_value[attr] = float(self.__dict__[attr]) + float(other[attr])
            return new_value

        elif other == 0:
            new_value = copy(self)
            return new_value
    
    def __mul__(self, other):
        if type(other) in [int, float, np.int64, np.float64]:
            new_value = self.__class__(self.dimensions)
            for attr in self.custom_attrs:
                new_value[attr] = float(self.__dict__[attr]) * other
            return new_value
        else:
            raise TypeError('Var can only be multiplied by numbers.')

    def __rmul__(self, other):
        if type(other) in [int, float, np.int64, np.float64]:
            new_value = self.__class__(self.dimensions)
            for attr in self.custom_attrs:
                new_value[attr] = float(self.__dict__[attr]) * other
            return new_value
        else:
            raise TypeError('Var can only be multiplied by numbers.')
    


class Structure(object):
    def __init__(self, subscripts=None, from_xmile=None):
        # causal loop diagram
        # there are three types of connections
        # 1. those used to forward calculate flows
        self.cld_flows = nx.DiGraph()
        # 2. those used to initialise stocks
        self.cld_init_stocks = nx.DiGraph()
        # 3. those used to update stocks with flows
        self.cld_update_stocks = nx.DiGraph()

        # run specs
        self.run_specs = {
            'initial_time': 0,
            'dt': 0.5,
            'simulation_time': 2
        }
        self.current_step = 1
        self.current_time = 0

        # equations & values
        self.var_eqns = dict()
        self.var_eqns_runtime = dict() # after initialisation, stocks should not have equations
        self.historical_values = list() # 2-d array

        # collections of variables by type
        self.stocks = list()
        self.flows = list()
        self.auxiliaries = list()

        # collections of all dependencies
        self.all_dependencies = dict()

        # subscripts
        self.dimensions = dict()
        self.var_dimensions = dict()
        self.elements_mapping = dict()

    ##################
    # Model definition
    ##################

    def add_stock(self, name, equation, dims=None, inflows=None, outflows=None):
        self.cld_flows.add_node(name)
        if inflows is not None:
            for flow in inflows:
                self.cld_update_stocks.add_edge(flow, name, sign=1)
        if outflows is not None:
            for flow in outflows:
                self.cld_update_stocks.add_edge(flow, name, sign=-1)

        self.stocks.append(name)
        self.var_eqns[name] = equation
        self.var_dimensions[name] = dims
        print('Engine: adding stock: {}={}'.format(name, self.var_eqns[name]))

    def add_flow(self, name, equation, dims=None, flow_from=None, flow_to=None):
        self.cld_flows.add_node(name)
        if flow_from is not None:
            self.cld_update_stocks.add_edge(name, flow_from, sign=-1)
        if flow_to is not None:
            self.cld_update_stocks.add_edge(name, flow_to, sign=1)

        self.flows.append(name)
        self.var_eqns[name] = equation
        self.var_dimensions[name] = dims
        print('Engine: adding flow : {}={}'.format(name, self.var_eqns[name]), 'Dims:', dims)

    def add_aux(self, name, equation, dims=None):
        self.cld_flows.add_node(name)
        self.auxiliaries.append(name)
        self.var_eqns[name] = equation
        self.var_dimensions[name] = dims
        print('Engine: adding aux  : {}={}'.format(name, self.var_eqns[name]), 'Dims:', dims)
    
    ###################
    # Model Compilation
    ###################
    @staticmethod
    def remove_bracket(input_str): # a[b] --> a
        # print('input eqn:', input_str)
        output_str = input_str
        while '[' in output_str:
            a, b = output_str.split('[', 1) # 1: only count for the first occasion of [
            b, c = b.split(']', 1)
            output_str = a+c
        # print('output eqn:', output_str)
        return output_str

    # Regular expression cannot handle multiple paired () or [] e.g. a[ele1]+b[ele2]
    # we process them manually by encoding a combination of elements e.g. 'ele1, 1' into a unique string 'Eele1_cmm__spc_1'
    # where '_cmm_' = ',' and '_spc_' = ' '
    @staticmethod
    def process_element(input_str): # '1, Element_1' --> 'E1__comma__Element_1'
        # print('process element:', input_str)
        input_str_1 = input_str.replace(' ', '_spc_')
        eles = input_str_1.split(',')
        output_str = 'E'+'_cmm_'.join(eles)
        # print('process element output:', output_str)
        return output_str
    
    @staticmethod
    def reverse_element(input_str): # 'E1__comma__Element_1' --> '1, Element_1'
        # print('reverse element:', input_str)
        output_str = input_str.replace('_cmm_', ',').replace('_spc_', ' ')[1:]
        # print('reverse element output:', output_str)
        return output_str
    
    def generate_stock_cld(self, vars, diagram):
        print('sss0')
        for var in vars:
            equation = self.var_eqns[var]
            print('sss1.0', var, equation)
            self.all_dependencies[var] = set()
            if type(equation) is str:
                print('sss2.0', var, equation)
                cause_vars = dict()
                success = False
                while not success: 
                    try: 
                        eval(equation, cause_vars)
                        if var not in diagram.nodes:
                            diagram.add_node(var) # add node (stock) to CLD (e.g. stock=10)
                        success = True
                    except NameError as e:
                        s = e.args[0]
                        p = s.split("'")[1]
                        diagram.add_edge(p, var) # add edge to CLD (e.g. stock=init_stock=10)
                        cause_vars[p] = 1
                
                self.var_eqns_compiled[var] = equation
            
            elif type(equation) is dict:
                print('sss2.1', var, equation)
                print('found eqn subscripted: {}={}'.format(var, equation))
                subscripted_var = Var(self.var_dimensions[var])
                for ele, eqn in equation.items():
                    print('sss2.2.0', var, ele, eqn)
                    processed_ele = self.process_element(ele)
                    print('sss2.2.1', var, ele, processed_ele)
                    self.elements_mapping[ele] = processed_ele
                    subscripted_var[processed_ele] = eqn
                    subscripted_var.custom_attrs.add(processed_ele)
                    # self.var_eqns_compiled[processed_ele] = processed_ele

                    cause_vars = dict()
                    success = False
                    while not success:
                        eqn_1 = self.remove_bracket(eqn)
                        try:
                            eval(eqn_1, cause_vars)
                            if var not in diagram.nodes:
                                diagram.add_node(var)
                            success = True
                        except NameError as e:
                            s = e.args[0]
                            p = s.split("'")[1]
                            diagram.add_edge(p, var)
                            cause_vars[p] = 1
                        except TypeError as e:
                            print('TypeError:', eqn_1)
                            raise e
                
                self.var_eqns_compiled[var] = subscripted_var
                    
    def generate_flow_cld(self, vars, diagram):
        for var in vars:
            self.all_dependencies[var] = set()
            equation = self.var_eqns[var]
            if type(equation) is str:
                # remove all subscripts [*] from equation
                equation_1 = self.remove_bracket(equation)

                cause_vars = dict()
                success = False
                
                # generate CLD
                while not success: 
                    try: 
                        eval(equation_1, cause_vars)
                        if var not in diagram.nodes:
                            diagram.add_node(var)
                        success = True
                    except NameError as e:
                        s = e.args[0]
                        p = s.split("'")[1]
                        link_edge = (p, var)
                        if link_edge not in diagram.edges:
                            print('new edge', link_edge)
                            diagram.add_edge(p, var) # add edge to CLD
                        self.all_dependencies[var].add(p) # put all_dependencies into a dictionary for faster access
                        cause_vars[p] = 1
                    except TypeError as e:
                        print('TypeError:', equation_1)
                        raise e
                
                self.var_eqns_compiled[var] = equation

            elif type(equation) is dict: # with subscripts
                # print('found eqn subscripted: {}={}'.format(var, equation))
                subscripted_var = Var(self.var_dimensions[var])
                for ele, eqn in equation.items():
                    processed_ele = self.process_element(ele)
                    # print('p_ele', processed_ele)
                    self.elements_mapping[ele] = processed_ele
                    subscripted_var[processed_ele] = eqn
                    subscripted_var.custom_attrs.add(processed_ele)
                    # self.var_eqns_compiled[processed_ele] = processed_ele # add subscript elements to compiled namespace
                    
                    cause_vars = dict()
                    success = False
                    while not success:
                        eqn_1 = self.remove_bracket(eqn)
                        try:
                            eval(eqn_1, cause_vars)
                            if var not in diagram.nodes:
                                print('new edge:', link_edge)
                                diagram.add_node(var)
                            success = True
                        except NameError as e:
                            s = e.args[0]
                            p = s.split("'")[1]
                            link_edge = (p, var)
                            if link_edge not in diagram.edges:
                                diagram.add_edge(p, var) # add edge to CLD
                            self.all_dependencies[var].add(p) # put all_dependencies into a dictionary for faster access
                            cause_vars[p] = 1
                        except TypeError as e:
                            print('TypeError:', eqn_1)
                            raise e
                        
                self.var_eqns_compiled[var] = subscripted_var
    
    def process_elements_in_equations(self):
        print('Compiled equations:', self.var_eqns_compiled)
        for var, eqn in self.var_eqns_compiled.items():
            print('pe0', var, eqn)
            if isinstance(eqn, str):
                print('pe1.0', var, eqn)
                for ele, p_ele in self.elements_mapping.items():
                    print('pe1.1.0', '['+ele+']', '['+p_ele+']')
                    self.var_eqns_compiled[var] = self.var_eqns_compiled[var].replace('['+ele+']', '['+p_ele+']') # include [] in search. avoid 'b=a+1' being affected by element 'a'. only b=a[a]+1 get matched.
                print('pe1.1', var, self.var_eqns_compiled[var])
            elif isinstance(eqn, Var):
                print('pe2.0', var, eqn)
                new_eqn = Var(self.var_dimensions[var])
                for attr in list(eqn.custom_attrs):
                    print('pe2.1.0', attr)
                    new_attr_eqn_string = eqn[attr]
                    for ele, p_ele in self.elements_mapping.items():
                        print('pe2.1.1.0', '['+ele+']', '['+p_ele+']')
                        print('pe2.1.1.1', eqn[attr])
                        new_attr_eqn_string = new_attr_eqn_string.replace('['+ele+']', '['+p_ele+']')
                    new_eqn[attr] = new_attr_eqn_string
                print('pe2.1', new_eqn)
                self.var_eqns_compiled[var] = new_eqn
            print('pe3', var, self.var_eqns_compiled[var])
    
        print('Compiled equations2:', self.var_eqns_compiled)

    def compile(self):
        print('starting compile...')
        self.var_eqns_compiled = deepcopy(self.var_eqns)
        # stocks
        self.generate_stock_cld(self.stocks, self.cld_init_stocks)
        # flows+auxiliaries
        self.generate_flow_cld(self.flows+self.auxiliaries, self.cld_flows)

        # print('compiled dependencies:', self.all_dependencies)

        self.process_elements_in_equations()

        # add all processed elements to var_eqns_compiled
        for ele, p_ele in self.elements_mapping.items():
            self.var_eqns_compiled[p_ele] = p_ele

        print('compiling finished:', self.var_eqns_compiled)

    ############
    # Simulation
    ############

    def simulate(self, simulation_time=None, dt=None):
        # Generate all clds (and compile subscripted variables)
        self.compile()

        if simulation_time is not None:
            self.run_specs['simulation_time'] = simulation_time
        if dt is not None:
            self.run_specs['dt'] = dt
        total_steps = int(self.run_specs['simulation_time']/self.run_specs['dt'])
        
        step_values = dict()
        all_variables = self.stocks + self.flows + self.auxiliaries # a full set of variables for detection of isolated vars
        all_variables = set(all_variables)

        # init stocks
        print('\nSTEP {}\n'.format(self.current_step))
        print('\nSimulate: init stocks')
        self.var_eqns_runtime = deepcopy(self.var_eqns_compiled)

        for stock in self.stocks:
            self.var_eqns_runtime[stock] = self.init_stocks(stock, diagram=self.cld_init_stocks, calculated=step_values)
        
        print('VarEqnsRuntime:', self.var_eqns_runtime)
        # calculate flows and update stocks
        for step in range(total_steps):
            # step 2
            print('\nSimulate: update flows_1')
            for flow in self.flows:
                self.update_flow_aux(flow, diagram=self.cld_flows, calculated=step_values)
            
            # step 3
            # some isolated vars need to be updated (e.g., stock_init-->stock)
            print('\nSimulate: update leftovers_1')
            left_overs = all_variables.difference(set(step_values.keys()))
            for left_over_var in left_overs:
                self.update_flow_aux(left_over_var, diagram=self.cld_flows, calculated=step_values) # although not affecting flows, their all_dependencies are stored in cld_flows 
            
            # self.historical_values.append(step_values.values())
            self.historical_values.append(step_values)

            # move to next step
            self.current_step += 1
            self.current_time += self.run_specs['dt']
            print('\nSTEP {}\n'.format(self.current_step))

            # step 1
            new_step_values = dict()
            for stock in self.stocks:
                v = self.update_stocks(stock, diagram=self.cld_update_stocks, calculated=step_values)
                new_step_values[stock] = v
                self.var_eqns_runtime[stock] = str(v)
            
            step_values = new_step_values

        # step 2
        for flow in self.flows:
            print('\nSimulate: update flows_2')
            self.update_flow_aux(flow, diagram=self.cld_flows, calculated=step_values)
            
        # step 3
        print('\nSimulate: update leftovers_2')
        left_overs = all_variables.difference(set(step_values.keys()))
        for left_over_var in left_overs:
            print('Updating leftover:', left_over_var)
            self.update_flow_aux(left_over_var, diagram=self.cld_flows, calculated=step_values) # although not affecting flows, their all_dependencies are stored in cld_flows 
            
        # self.historical_values.append(step_values.values())
        self.historical_values.append(step_values)
        
    def init_stocks(self, var, diagram, calculated=dict()):
        print('iii0, init stock:', var)
        dependencies = list(diagram.predecessors(var))
        equation = self.var_eqns_compiled[var]
        for dependency in list(dependencies):
            self.init_stocks(dependency, diagram, calculated)
        
        # if len(dependencies) == 0:
        #     equation = self.var_eqns_compiled[var]
        
        if type(equation) is str:
            print('iii1', equation)
            value = eval(self.var_eqns_compiled[var], {}, calculated)
            calculated[var] = value
        elif type(equation) is Var:
            print('iii2.0', equation)
            value = Var(self.var_dimensions[var])
            for attr in list(equation.custom_attrs):
                v = eval(equation[attr], self.var_eqns_runtime, calculated)
                value[attr] = v
            print('iii2.1', value)
            calculated[var] = value
        
        # else:
        #     for dependency in list(dependencies):
        #         self.init_stocks(dependency, diagram, calculated)
        #     eqation = self.var_eqns_compiled[var]

        #     value = eval(self.var_eqns[var], {}, calculated)
        print('iii3', value)
        return value

    def update_flow_aux(self, var, diagram, calculated=dict()):
        print('aaa0.1', var)
        print('aaa0.2', diagram.edges)
        print('aaa0.3', calculated)
        dependencies = self.all_dependencies[var]
        print('aaa0.4', dependencies)
        if len(dependencies) == 0:
            print('aaa1', var)
            equation = self.var_eqns_runtime[var]
            print('aaa1.0', 'compiled equation of {}'.format(var), equation, type(equation))
            if type(equation) is str:
                print('aaa1.1.0', equation, type(equation))
                value = eval(equation, {}, calculated)
                print('aaa1.1.1', value, type(value))
                calculated[var] = value
            elif isinstance(equation, Var): # this is a subscripted constant (3) or expression (3*2) or e.g. rbinom(1)
                subscripted_value = Var(self.var_dimensions[var])
                print('aaa1.2 custom attrs of {}'.format(var), equation.custom_attrs)
                for attr in list(equation.custom_attrs):
                    print('aaa1.3.1', attr)
                    print('aaa1.3.2', equation[attr])
                    subscripted_value[attr] = str(eval(equation[attr])) # 3*2 -> 6; 3->3
                    print('aaa1.3.3 custom attr {} of {}:'.format(attr, var), subscripted_value[attr])
                    # print('success')
                print('aaa1.3.3.1', subscripted_value)
                calculated[var] = subscripted_value
                print('aaa1.3.4', calculated[var].custom_attrs)
            else:
                print('TypeError while evaluating:', self.var_eqns_runtime[var])
                raise e
            print('aaa1.4', calculated)
            print('aaa1_out', calculated[var], '\n')
        else: # var has dependencies
            print('aaa2', var)
            print('aaa2.0', 'compiled equation of {}:'.format(var), self.var_eqns_compiled[var])

            for dependency in list(dependencies):
                print('aaa2.1', dependency)
                self.update_flow_aux(dependency, diagram=diagram, calculated=calculated)
            print('solved dependencies for {}:'.format(var), calculated)
            try:
                eqn_runtime = self.var_eqns_runtime[var]
                print('aaa2.2 eqn_runtime for {}'.format(var), eqn_runtime)
                value = eval(eqn_runtime, self.var_eqns_runtime, calculated)
                print('value for {}'.format(value))
                calculated[var] = value
                print('calculated after {}'.format(var), calculated)
            except NameError as e:
                print('NameError while evaluating:', eqn_runtime)
                # print('var_eqns_runtime:', self.var_eqns_runtime)
                # print('calculated:', calculated)
                raise e
            except TypeError as e:
                print('aaa2.3 TypeError of {}:'.format(var), eqn_runtime)
                if isinstance(eqn_runtime, Var):
                    subscripted_value = Var(self.var_dimensions[var])
                    for attr in list(eqn_runtime.custom_attrs):
                        print('aaa2.3.1', var, attr, eqn_runtime[attr])
                        print('aaa2.3.2', calculated)
                        subscripted_value[attr] = str(eval(eqn_runtime[attr], self.var_eqns_runtime, calculated))
                        print('aaa2.3.3 custom attr {} of {}:'.format(attr, var), subscripted_value[attr])
                calculated[var] = subscripted_value
                # print('TypeError while evaluating:', self.var_eqns_runtime[var])
                # print('var_eqns_runtime:', self.var_eqns_runtime)
                # print('calculated', calculated)
            print('aaa2_out', calculated[var], '\n')
        
    def update_stocks(self, stock, diagram, calculated):
        print('bbb0.1', stock)
        print('bbb0.2', diagram.edges)
        delta_value = 0
        flows = list(diagram.in_edges(stock))
        print('bbb0.3', flows)
        # print(diagram.edges(data=True))
        for flow in flows:
            print('bbb1.0', flow)
            print('bbb1.1', calculated[flow[0]])
            p = calculated[flow[0]] * diagram.edges[flow]['sign']
            print('bbb1.2', p)
            delta_value += p
        delta_value = delta_value * self.run_specs['dt']
        print('bbb0.4', calculated[stock], delta_value)
        new_value = calculated[stock] + delta_value
        print('bbb2', stock, new_value)
        return new_value

    #################
    # Data Processing
    #################

    def export_simulation_result(self):
        import pandas as pd
        df_output = pd.DataFrame.from_records(data=self.historical_values)
        return df_output

if __name__ == '__main__':

    test_set = 7

    if test_set == 0:
        x = 'x'
        y = 'y'

        a = Var('dim1')
        a.x = 10
        a[y] = 20
        b = 30


        v = eval("b + a[x] + a[y] ")
        print(v)

    if test_set == 1:
        model = Structure()
        model.add_aux(name='initial_stock', equation='100')
        model.add_stock(name='stock1', equation='initial_stock', inflows=['flow1'])
        model.add_stock(name='stock2', equation='100')
        
        model.add_flow(name='flow1', equation='gap1/at1', flow_to='stock1')
        model.add_aux(name='goal1', equation='20')
        model.add_aux(name='gap1', equation='goal1-stock1')
        model.add_aux(name='at1', equation='5')
    
    elif test_set == 2:
        model = Structure()
        model.subscripts = {'dim1': ['x', 'y'], 'dim2': ['1', '2']}
        # model.add_aux(name='a', equation='5')
        model.add_aux(name='b', dims=['dim1', 'dim2'], equation={'x, 1':'100', 'y, 1':'200', 'x, 2':'300', 'y, 2':'400'})
        model.add_aux(name='c', equation='b[x, 2]')
        # model.add_flow(name='flow1', equation='aux1 *2 + aux2[ele1] * 1')
    
    elif test_set == 3:
        model = Structure()
        print(model.replace_bracket('a[ele1, 1]+b[ele2, 2]'))

    elif test_set == 4:
        model = Structure()
        print(model.remove_bracket('a[ele1]+b[ele2]'))
    
    elif test_set == 5:
        model = Structure()
        a0 = '1, Element_1, a'
        a= model.process_element(a0)
        b = model.reverse_element(a)
        print(b)

    elif test_set == 6:
        model = Structure()
        model.subscripts = {'dim1': ['x', 'y'], 'dim2': ['1', '2']}
        model.add_aux(name='b', dims=['dim1', 'dim2'], equation={
            'x, 1':'100', 
            'y, 1':'200', 
            'x, 2':'300', 
            'y, 2':'400'}
            )
        model.add_flow(name='c', dims=['dim1', 'dim2'], equation = {
            'x, 1':'b[x, 1]',
            'x, 2':'b[x, 2]',
            'y, 1':'b[y, 1]',
            'y, 2':'b[y, 2]'}
            )
    
    elif test_set == 7:
        model = Structure()
        model.add_aux(name='init_stock', dims=['dim1'], equation={
            'x': '100',
            'y': '200'}
        )
        model.add_aux(name='goal', dims=['dim1'], equation={
            'x': '10',
            'y': '20'}
        )
        model.add_aux(name='gap', dims=['dim1'], equation={
            'x': 'goal[x]-stock[x]',
            'y': 'goal[y]-stock[y]'}
        )
        model.add_aux(name='adj_time', dims=['dim1'], equation={
            'x': '5',
            'y': '3'}
        )
        model.add_stock(name='stock', dims=['dim1'], equation={
            'x': 'init_stock[x]',
            'y': 'init_stock[y]'}
            )
        model.add_flow(name='flow', dims=['dim1'], equation={
            'x': 'gap[x]/adj_time[x]',
            'y': 'gap[y]/adj_time[y]'},
            flow_to='stock')
        

    model.compile()

    print('\nCLDs:')
    print('1 CLD flows:', model.cld_flows.edges(data=True))
    print('2 CLD init_stocks:', model.cld_init_stocks.edges(data=True)) # single stocks won't show here
    print('3 CLD update_stocks:', model.cld_update_stocks.edges(data=True))

    print('\nEquations defined:')
    print(model.var_eqns)
    
    print('\nEquations compiled:')
    print(model.var_eqns_compiled)

    model.simulate(simulation_time=1, dt=1)

    print('\nResults')
    print(model.export_simulation_result())
    print('\nPASS\n')
