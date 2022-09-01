from concurrent.futures import process
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
    
    def __repr__(self):
        repr_dict = dict(zip(self.custom_attrs, [self.__dict__[a] for a in self.custom_attrs]))
        # representation = ''
        # for k, v in repr_dict.items():
        #     representation = representation + k + ':'+ str(v) +'\n'
        representation = str(repr_dict)
        return 'Var:' + representation

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
        # 0. proto CLD
        self.cld = nx.DiGraph()
        # there are three types of sub-CLDs
        # 1. those used to forward calculate flows
        self.cld_flows = nx.DiGraph()
        # 2. those used to initialise stocks
        self.cld_init_stocks = nx.DiGraph()
        # 3. those used to update stocks with flows
        self.cld_update_stocks = nx.DiGraph()

        # collections of variables by type
        self.stocks = list()
        self.flows = list()
        self.auxiliaries = list()
        
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
        self.var_eqns_compiled = dict()
        self.var_eqns_runtime = dict() # after initialisation, stocks should not have equations
        
        # historical values
        self.historical_values = list() # 2-d array

        # subscripts
        self.dimensions = dict()
        self.var_dimensions = dict()
        self.var_elements = dict()
        self.elements_mapping = dict()

        # collective namespace
        # self.name_space = dict() # a__ele__b__comma__space__1__ele -> a[b, 1]

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

        output_str = '__ele1__'+input_str.replace(',', '__cmm__').replace(' ','')+'__ele2__'
        return output_str
    
    @staticmethod
    def reverse_element(input_str): # 'E1__comma__Element_1' --> '1, Element_1'
        # print('reverse element:', input_str)
        output_str = input_str.replace('_cmm_', ',').replace('_spc_', ' ')[1:]
        # print('reverse element output:', output_str)
        return output_str
    
    @staticmethod
    def process_equation(input_str): # a[1, b] --> a__ele1__1__cmm__b__ele2__
        output_str = input_str.replace('[', '__ele1__').replace(']', '__ele2__').replace(',', '__cmm__').replace(' ','')
        return output_str
    
    def recursive_trace(self, var, diagram, n_steps=None, type_stop_con=None):
        if var in diagram.nodes:
            # if stop_con is not None:
            #     not_end = 
            not_end = True
            while not_end and n_steps != 0:
                dependencies = list(diagram.predecessors(var))
                if len(dependencies) != 0:
                    for dependency in dependencies:
                        diagram.add_edge(dependency, var)
                        self.recursive_trace()

        else:
            diagram.nodes[var] # trigger ValueError


    def generate_stock_cld(self, vars, diagram):
        # print('sss0')
        for var in vars:
            equation = self.var_eqns[var]
            # print('sss1.0', var, equation)
            # self.all_dependencies[var] = set()
            
            if type(equation) is str:
                # print('sss1.1', var, equation)

                # self.name_space[var] = equation

                cause_vars = dict()
                success = False
                while not success: 
                    equation_1 = self.remove_bracket(equation)
                    try: 
                        eval(equation_1, cause_vars)
                        if var not in diagram.nodes:
                            diagram.add_node(var) # add node (stock) to CLD (e.g. stock=10)
                        success = True
                        
                    except NameError as e:
                        s = e.args[0]
                        p = s.split("'")[1]
                        diagram.add_edge(p, var) # add edge to CLD (e.g. stock=init_stock=10)
                        # put dependency p to name_space

                        cause_vars[p] = 1
                    except TypeError as e:
                        print('TypeError:', equation_1)
                        raise e
                
                processed_equation = self.process_equation(equation)
                try:
                    processed_equation = float(processed_equation) # constants str -> float
                except:
                    pass

                self.var_eqns_compiled[var] = self.process_equation(equation)
            
            elif type(equation) is dict:
                # print('sss1.2', var, equation)
                # print('found eqn subscripted: {}={}'.format(var, equation))
                subscripted_equation = Var(self.var_dimensions[var])
                self.var_elements[var] = set()
                for ele, eqn in equation.items():
                    # print('sss2.2.0', var, ele, eqn)
                    processed_ele = self.process_element(ele)
                    # print('sss2.2.1', var, ele, processed_ele)
                    self.elements_mapping[ele] = processed_ele
                    
                    processed_eqn = self.process_equation(eqn)
                    try:
                        processed_eqn = float(processed_eqn)
                    except:
                        pass
                    
                    subscripted_equation[processed_ele] = self.process_equation(eqn)
                    subscripted_equation.custom_attrs.add(processed_ele)
                    
                    self.var_eqns_compiled[var+processed_ele] = subscripted_equation[processed_ele]
                    self.var_elements[var].add(processed_ele)

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
                            # self.all_dependencies[var].add(p)
                            cause_vars[p] = 1
                        except TypeError as e:
                            print('TypeError:', eqn_1)
                            raise e
                
                self.var_eqns_compiled[var] = subscripted_equation
                    
    def generate_flow_cld(self, vars, diagram):
        for var in vars:
            # self.all_dependencies[var] = set()
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
                        # self.all_dependencies[var].add(p) # put all_dependencies into a dictionary for faster access
                        cause_vars[p] = 1
                    except TypeError as e:
                        print('TypeError:', equation_1)
                        raise e
                
                processed_equation = self.process_equation(equation)
                try:
                    processed_equation = float(processed_equation) # constants str -> float
                except:
                    pass
                self.var_eqns_compiled[var] = processed_equation

            elif type(equation) is dict: # with subscripts
                # print('found eqn subscripted: {}={}'.format(var, equation))
                subscripted_equation = Var(self.var_dimensions[var])
                self.var_elements[var] = set()
                for ele, eqn in equation.items():
                    processed_ele = self.process_element(ele)
                    # print('p_ele', processed_ele)
                    self.elements_mapping[ele] = processed_ele

                    processed_eqn = self.process_equation(eqn)
                    try:
                        processed_eqn = float(processed_eqn)
                    except:
                        pass

                    subscripted_equation[processed_ele] = processed_eqn
                    subscripted_equation.custom_attrs.add(processed_ele)

                    self.var_eqns_compiled[var+processed_ele] = subscripted_equation[processed_ele]
                    self.var_elements[var].add(processed_ele)
                    
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
                            # self.all_dependencies[var].add(p) # put all_dependencies into a dictionary for faster access
                            cause_vars[p] = 1
                        except TypeError as e:
                            print('TypeError:', eqn_1)
                            raise e
                        
                # self.var_eqns_compiled[var] = subscripted_equation


    def compile(self):
        # print('starting compile...')
        self.var_eqns_compiled = deepcopy(self.var_eqns)
        # stocks
        self.generate_stock_cld(self.stocks, self.cld_init_stocks)
        # flows+auxiliaries
        self.generate_flow_cld(self.flows+self.auxiliaries, self.cld_flows)

        # print('\ncompiling finished:', self.var_eqns_compiled)

        # print('\nnamespace:', self.name_space)

        # print('\nvar delements', self.var_elements)

    ############
    # Simulation
    ############

    def evaluation(self, expression, gb, lc):
        try:
            value = eval(expression, gb, lc)
        except TypeError as e:
            if type(expression) in [int, float, np.int64, np.float64]:
                value = expression
            else:
                raise e

        return value


    def init_stock(self, var, diagram, calculated, if_stock):
        self.visited.add(var)
        print('iii0, init stock:', var)
        
        dependencies = list(diagram.predecessors(var))

        try:
            elements = self.var_elements[var]
        
            for element in elements:
                var_ele = var+element
                var_ele_eqn = self.var_eqns_compiled[var_ele]

                for dependency in dependencies:
                    if dependency in self.stocks:
                        ifstk = True
                    else:
                        ifstk = False
                    self.init_stock(dependency, diagram, calculated, if_stock=ifstk)
                
                # try:
                #     var_ele_v = eval(var_ele_eqn, {}, calculated)
                # except TypeError:
                #     if type(var_ele_eqn) in [int, float, np.int64, np.float64]:
                #         var_ele_v = var_ele_eqn

                var_ele_v = self.evaluation(var_ele_eqn, gb={}, lc=calculated)
                
                calculated[var_ele] = var_ele_v
                
                # check if this var_ele is of stock
                # if so, update the runtime namespace to replace a stock's initilisaion expression
                if if_stock:
                    self.var_eqns_runtime[var_ele] = str(var_ele_v) # str needed for eval()
        
        except: # the model does not use arrays
            
            var_eqn = self.var_eqns_compiled[var]

            for dependency in dependencies:
                if dependency in self.stocks:
                    ifstk = True
                else:
                    ifstk = False
                self.init_stock(dependency, diagram, calculated, if_stock=ifstk)
            
            # try:
            #     var_v = eval(var_eqn, {}, calculated)
            # except TypeError: # when var_eqn has been converted to number at the compiling stage
            #     if type(var_eqn) in [int, float, np.int64, np.float64]:
            #         var_v = var_eqn 

            var_v = self.evaluation(var_eqn, gb={}, lc=calculated)

            calculated[var] = var_v

            if if_stock:
                self.var_eqns_runtime[var] = str(var_v)
        

    def update_flow_aux(self, var, diagram, calculated=dict()):
        self.visited.add(var)
        dependencies = list(diagram.predecessors(var))
        # print('ufa1', calculated)
        try:
            elements = self.var_elements[var]
        
            for element in elements:
                var_ele = var+element
                var_ele_eqn = self.var_eqns_runtime[var_ele]

                for dependency in dependencies:
                    self.update_flow_aux(dependency, diagram, calculated)
                
                # e.g., constants are not included in calculated yet, so self.var_eqns_runtime is necessary
                # try:
                #     var_ele_v = eval(var_ele_eqn, self.var_eqns_runtime, calculated)
                # except TypeError:
                #     if type(var_ele_eqn) in [int, float, np.int64, np.float64]:
                #         var_ele_v = var_ele_eqn

                var_ele_v = self.evaluation(var_ele_eqn, gb=self.var_eqns_runtime, lc=calculated)

                calculated[var_ele] = var_ele_v
                # print('ufa', var_ele, var_ele_v)
        except: # var is not subscripted
            var_eqn = self.var_eqns_runtime[var]
            
            for dependency in dependencies:
                self.update_flow_aux(dependency, diagram, calculated)

            # try:
            #     var_v = eval(var_eqn, self.var_eqns_runtime, calculated)
            # except TypeError:
            #     if type(var_eqn) in [int, float, np.int64, np.float64]:
            #         var_v = var_eqn

            var_v = self.evaluation(var_eqn, gb=self.var_eqns_runtime, lc=calculated)

            calculated[var] = var_v

        
    def update_stock(self, stock, diagram, calculated, to_be_calculated):
        self.visited.add(stock)
        # print('bbb0.1', stock)
        # print('bbb0.2', diagram.edges)
        delta_value = 0
        flows = list(diagram.in_edges(stock))
        # print('bbb0.3', flows)

        try:
            elements = self.var_elements[stock]
            for element in elements:
                data_value = 0
                stock_ele = stock+element
                stock_ele_v_current = calculated[stock_ele]

                for flow in flows:
                    flow_var = flow[0]
                    flow_ele = flow_var+element
                    # print('bbb1.0', flow_ele)
                    flow_ele_v = calculated[flow_ele]
                    # print('bbb1.1', flow_ele_v)
                    p = flow_ele_v * diagram.edges[flow]['sign']
                    # print('bbb1.2', p)
                    delta_value = data_value + p
                
                delta_value = delta_value * self.run_specs['dt']
                
                stock_ele_v_new = stock_ele_v_current + delta_value

                to_be_calculated[stock_ele] = stock_ele_v_new

                self.var_eqns_runtime[stock_ele] = str(stock_ele_v_new)
        
        except:
            data_value = 0
            stock_v_current = calculated[stock]

            for flow in flows:
                flow_var = flow[0]
                # print('bbb1.0', flow_ele)
                flow_v = calculated[flow_var]
                # print('bbb1.1', flow_ele_v)
                p = flow_v * diagram.edges[flow]['sign']
                # print('bbb1.2', p)
                delta_value = data_value + p
            
            delta_value = delta_value * self.run_specs['dt']
            
            stock_v_new = stock_v_current + delta_value

            to_be_calculated[stock] = stock_v_new

            self.var_eqns_runtime[stock] = str(stock_v_new)
    

    def simulate(self, simulation_time=None, dt=None):
        # Generate all clds (and compile subscripted variables)
        self.compile()

        if simulation_time is not None:
            self.run_specs['simulation_time'] = simulation_time
        if dt is not None:
            self.run_specs['dt'] = dt
        total_steps = int(self.run_specs['simulation_time']/self.run_specs['dt'])
        
        step_values = dict()
        all_variables = set(self.stocks + self.flows + self.auxiliaries) # a full set of variables for detection of isolated vars
        self.visited = set() # for leftovers
        
        # init stocks
        # print('\nSTEP {}\n'.format(self.current_step))
        # print('\nSimulate: init stocks')
        # self.var_eqns_runtime = deepcopy(self.var_eqns_compiled)

        self.var_eqns_runtime = deepcopy(self.var_eqns_compiled)

        # print('zzz0', self.var_eqns_runtime)

        for stock in self.stocks:
            self.init_stock(stock, diagram=self.cld_init_stocks, calculated=step_values, if_stock=True)

        # print('zzz1', self.var_eqns_runtime)

        # calculate flows and update stocks
        for step in range(total_steps):
            # step 2
            # print('\nSimulate: update flows_1')
            for flow in self.flows:
                self.update_flow_aux(flow, diagram=self.cld_flows, calculated=step_values)
            
            # print('zzz2', step_values)
            
            # step 3
            # some isolated vars need to be updated (e.g., stock_init-->stock)
            # print('\nSimulate: update leftovers_1')
            left_overs = all_variables.difference(self.visited)
            # print('leftovers:', left_overs)
            for left_over_var in left_overs:
                self.update_flow_aux(left_over_var, diagram=self.cld_flows, calculated=step_values) # although not affecting flows, their all_dependencies are stored in cld_flows 
            
            # self.historical_values.append(step_values.values())
            self.historical_values.append(step_values)

            # print('zzz3', step_values)

            # move to next step
            self.current_step += 1
            self.current_time += self.run_specs['dt']
            # print('\nSTEP {}\n'.format(self.current_step))
            
            self.visited = set()

            # step 1
            new_step_values = dict()
            for stock in self.stocks:
                self.update_stock(stock, diagram=self.cld_update_stocks, calculated=step_values, to_be_calculated=new_step_values)
                # new_step_values[stock] = v
                # self.var_eqns_runtime[stock] = str(v)

            step_values = new_step_values
            # print('zzz4', step_values)

        # step 2
        for flow in self.flows:
            # print('\nSimulate: update flows_2')
            self.update_flow_aux(flow, diagram=self.cld_flows, calculated=step_values)
            
        # step 3
        # print('\nSimulate: update leftovers_2')
        left_overs = all_variables.difference(self.visited)
        # print('leftovers:', left_overs)
        for left_over_var in left_overs:
            # print('Updating leftover:', left_over_var)
            self.update_flow_aux(left_over_var, diagram=self.cld_flows, calculated=step_values) # although not affecting flows, their all_dependencies are stored in cld_flows 
            
        # self.historical_values.append(step_values.values())
        self.historical_values.append(step_values)

    #################
    # Data Processing
    #################

    def export_simulation_result(self):
        import pandas as pd
        df_output = pd.DataFrame.from_records(data=self.historical_values)
        return df_output

if __name__ == '__main__':

    def test(test_set):

        if test_set == 1:
            model = Structure()
            model.add_aux(name='initial_stock', equation='100')
            model.add_stock(name='stock1', equation='initial_stock', inflows=['flow1'])
            model.add_stock(name='stock2', equation='200')
            
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
        
        # elif test_set == 3:
        #     model = Structure()
        #     print(model.replace_bracket('a[ele1, 1]+b[ele2, 2]'))

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

        elif test_set == 8:
            model = Structure()
            a = model.process_equation('a[b,1]')
            print(a)

        model.compile()
        
        print('\nCLDs:')
        print('1 CLD flows:', model.cld_flows.edges(data=True))
        print('2 CLD init_stocks:', model.cld_init_stocks.edges(data=True)) # single stocks won't show here
        print('3 CLD update_stocks:', model.cld_update_stocks.edges(data=True))
        
        model.simulate(simulation_time=2, dt=1)

        # print('\nEquations defined:')
        # print(model.var_eqns)
        
        # print('\nEquations compiled:')
        # print(model.var_eqns_compiled)

        print('\nResults')
        print(model.export_simulation_result().transpose())
        print('\nPASS\n')


for test_set in [1, 2, 4, 5, 6, 7, 8]:
    print('\n'+'*'*10)
    print('Test Set {}'.format(test_set))
    print('*'*10 + '\n')
    test(test_set)