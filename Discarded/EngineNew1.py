import networkx as nx
import numpy as np
from tqdm.notebook import tnrange
from copy import copy, deepcopy
from itertools import product
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

        # flags
        self.if_compiled = False

        # equations & values
        self.var_eqns = dict()
        self.var_eqns_compiled = dict()
        self.var_eqns_runtime = dict() # after initialisation, stocks should not have equations
        
        # historical values
        self.historical_values = list() # 2-d array

        # subscripts
        self.dimensions = dict() # dim:[elements]
        self.var_dimensions = dict()
        self.var_elements = dict() # x:__ele1__x__ele2__
        self.elements_mapping = dict()
        self.var_names_mapping = dict() # stock[x]: stock__ele1__x__ele2__

        # If the model is based on an XMILE file
        if from_xmile is not None:
            from pathlib import Path
            xmile_path = Path(from_xmile)
            if not xmile_path.exists():
                raise Exception("Specified model file does not exist.")
            else:
                with open(xmile_path) as f:
                    xmile_content = f.read().encode()
                    f.close()
                from bs4 import BeautifulSoup

                # read sim_specs
                sim_specs_root = BeautifulSoup(xmile_content, 'xml').find('sim_specs')
                sim_start = float(sim_specs_root.find('start').text)
                sim_stop = float(sim_specs_root.find('stop').text)
                sim_duration = sim_stop - sim_start
                sim_dt_root = sim_specs_root.find('dt')
                sim_dt = float(sim_dt_root.text)
                if sim_dt_root.get('reciprocal') == 'true':
                    sim_dt = 1/sim_dt
                
                self.run_specs['initial_time'] = sim_start
                self.run_specs['dt'] = sim_dt
                self.run_specs['simulation_time'] = sim_duration
                self.current_time = sim_start

                # read subscritps
                try:
                    subscripts_root = BeautifulSoup(xmile_content, 'xml').find('dimensions')
                    dimensions = subscripts_root.findAll('dim')

                    # dims = dict()
                    for dimension in dimensions:
                        name = dimension.get('name')
                        try:
                            size = dimension.get('size')
                            self.dimensions[name] = [str(i) for i in range(1, int(size)+1)]
                        except:
                            elems = dimension.findAll('elem')
                            elem_names = list()
                            for elem in elems:
                                elem_names.append(elem.get('name'))
                            self.dimensions[name] = elem_names
                    print('Dimensions found in XMILE:', self.dimensions)
                except AttributeError:
                    pass
                
                # read variables
                variables_root = BeautifulSoup(xmile_content, 'xml').find('variables') # omit names in view
                stocks = variables_root.findAll('stock')
                flows = variables_root.findAll('flow')
                auxiliaries = variables_root.findAll('aux')

                inflow_stock = dict()
                outflow_stock = dict()
                
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
                        print('Processing XMILE definition for:', var.get('name'))
                        self.var_dimensions[var.get('name')] = list()
                        var_dimensions = var.find('dimensions').findAll('dim')
                        # print(var_dimensions)

                        for dimension in var_dimensions:
                            dim_name = dimension.get('name')
                            self.var_dimensions[var.get('name')].append(dim_name)

                        var_elements = var.findAll('element')
                        var_element_equations = dict()
                        if len(var_elements) != 0:
                            for var_element in var_elements:
                                list_of_elements_string = var_element.get('subscript') # something like "1, First"
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
                                # var_subscripted_eqn.sub_contents[tuple_of_elements] = equation
                                var_element_equations[list_of_elements_string] = equation

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
                            
                            # generate element list based on dims
                            
                            involved_elements = list()
                            for d in self.var_dimensions[var.get('name')]:
                                involved_elements.append(self.dimensions[d])
                            
                            involved_elements_tuples = product(*involved_elements)
                            for involved_elements_tuple in involved_elements_tuples:
                                involved_elements_string = ', '.join(involved_elements_tuple)
                                var_element_equations[involved_elements_string] = equation
                                
                        return(var_element_equations)
                    else:
                        if var.find('conveyor'):
                            equation = var.find('eqn').text
                            length = var.find('len').text
                            equation = Conveyor(length, equation)
                        elif var.find('gf'):
                            equation = read_graph_func(var)
                            equation.eqn = var.find('eqn').text
                        elif var.find('eqn'):
                            var_equation = var.find('eqn').text
                            equation = var_equation
                        return(equation)
                        

                # create stocks
                for stock in stocks:
                    
                    non_negative = False
                    if stock.find('non_negative'):
                        # print('nonnegstock', stock)
                        non_negative = True

                    self.add_stock(self.name_handler(stock.get('name')), equation=subscripted_equation(stock), non_negative=non_negative)
                    
                    inflows = stock.findAll('inflow')
                    if len(inflows) != 0:
                        for inflow in inflows:
                            inflow_stock[inflow.text]=self.name_handler(stock.get('name'))
                    outflows = stock.findAll('outflow')
                    if len(outflows) != 0:
                        for outflow in outflows:
                            outflow_stock[outflow.text]=self.name_handler(stock.get('name'))

                # create auxiliaries
                for auxiliary in auxiliaries:
                    # self.add_aux(self.name_handler(auxiliary.get('name')), equation=auxiliary.find('eqn').text)
                    self.add_aux(self.name_handler(auxiliary.get('name')), equation=subscripted_equation(auxiliary))

                # create flows
                for flow in flows:
                    if self.name_handler(flow.get('name')) in inflow_stock.keys():
                        flow_to = inflow_stock[self.name_handler(flow.get('name'))]
                    else:
                        flow_to = None
                    
                    if self.name_handler(flow.get('name')) in outflow_stock.keys():
                        flow_from = outflow_stock[self.name_handler(flow.get('name'))]
                    else:
                        flow_from = None
                    
                    # check if flow is a leakage flow
                    if flow.find('leak'):
                        leak = True
                    else:
                        leak = False

                    # check if can be negative
                    non_negative = False
                    if flow.find('non_negative'):
                        non_negative = True
                    self.add_flow(self.name_handler(flow.get('name')), equation=subscripted_equation(flow), flow_from=flow_from, flow_to=flow_to, leak=leak, non_negative=non_negative)
        

    ##################
    # Model definition
    ##################

    def add_stock(self, name, equation, dims=None, inflows=None, outflows=None, non_negative=False):
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

    def add_flow(self, name, equation, dims=None, flow_from=None, flow_to=None, leak=None, non_negative=False):
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
    def name_handler(name):
        return name.replace(' ', '_').replace('\\n', '_')

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
    def reverse_var_element(input_str): # 'var__ele1__1__cmm__First__ele2__' --> 'var[1, First]'
        # print('reverse element:', input_str)
        var = input_str.split('__ele1__', 1)[0]
        try:
            element = input_str.split('__ele1__', 1)[1].split('__ele2__', 1)[0]
            element = '__ele1__'+element+'__ele2__'
        except IndexError as e: # input_str doesn't have element part
            # print(input_str)
            element = None
            # raise e
        return (var, element)
    
    def process_equation(self, input_str, default_element=None): # a[1, b] --> a__ele1__1__cmm__b__ele2__
        # 1. convert a[b] to a__ele1__b__ele2__
        output_str = input_str.replace('[', '__ele1__').replace(']', '__ele2__').replace(',', '__cmm__').replace(' ','')

        # 2. convert '1' (str) to 1.0 (float)
        try:
            output_str = float(output_str)
        except:
            pass
        return output_str
    
    # def process_equation_2(self, input_str, default_element=None):
    #     # based on context, convert a to a[ele] when a is subscripted but not specified in equation
    #     # this is not needed for variables that are not arrayed
    #     if default_element is not None:
            
    #         success = False
    #         while not success:
    #         try:
    #             eval(output_str, )
    
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
            
            if type(equation) is str:
                # print('sss1.1', var, equation)

                self.var_names_mapping[var] = var

                # get a list of inedge
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
                        print('sss1.1.1 TypeError:', equation_1)
                        raise e
                
                # flatten equation
                processed_equation = self.process_equation(equation)

                self.var_eqns_compiled[var] = processed_equation
            
            elif type(equation) is dict:
                # print('sss1.2', var, equation)
                # print('found eqn subscripted: {}={}'.format(var, equation))
                self.var_elements[var] = set()
                for ele, eqn in equation.items():
                    # print('sss1.2.0', var, ele, eqn)
                    processed_ele = self.process_element(ele)
                    # print('sss1.2.1', var, ele, processed_ele)
                    self.elements_mapping[ele] = processed_ele
                    
                    # flatten equation
                    processed_eqn = self.process_equation(eqn, default_element=ele)

                    # print('sss1.2.2', var, ele, processed_eqn)
                    self.var_eqns_compiled[var+processed_ele] = processed_eqn
                    self.var_names_mapping[var+processed_ele] = var+'['+ele+']'
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
                            cause_vars[p] = 1
                        except TypeError as e:
                            print('sss1.2.2.2 TypeError:', eqn_1)
                            raise e
                    
    def generate_flow_cld(self, vars, diagram):
        for var in vars:
            equation = self.var_eqns[var]
            if type(equation) is str:

                self.var_names_mapping[var] = var
                # remove all subscripts [*] from equation
                equation_1 = self.remove_bracket(equation)

                cause_vars = dict()
                cause_vars['TIME'] = 1
                success = False
                
                # generate CLD
                while not success: 
                    try: 
                        # eval(equation_1, cause_vars)
                        self.evaluation(equation, gb={}, lc=cause_vars)
                        if var not in diagram.nodes:
                            diagram.add_node(var)
                        success = True
                    except NameError as e:
                        s = e.args[0]
                        p = s.split("'")[1]

                        if p in self.var_eqns_compiled.keys():
                            pass
                        elif (p, var) not in diagram.edges:
                            print('new edge', (p, var))
                            diagram.add_edge(p, var) # add edge to CLD
                        cause_vars[p] = 1
                    except TypeError as e:
                        print('TypeError:', equation_1)
                        raise e

                    except SyntaxError as e:
                        print('SyntaxError:', equation_1)
                        print(e.args)
                        raise e
                
                processed_equation = self.process_equation(equation)
                try:
                    processed_equation = float(processed_equation) # constants str -> float
                except:
                    pass
                self.var_eqns_compiled[var] = processed_equation

            elif type(equation) is dict: # with subscripts
                # print('found eqn subscripted: {}={}'.format(var, equation))
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

                    # subscripted_equation[processed_ele] = processed_eqn
                    # subscripted_equation.custom_attrs.add(processed_ele)

                    self.var_eqns_compiled[var+processed_ele] = processed_eqn
                    self.var_names_mapping[var+processed_ele] = var+'['+ele+']'
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
                            cause_vars[p] = 1
                        except TypeError as e:
                            print('TypeError:', eqn_1)
                            raise e
                        
                # self.var_eqns_compiled[var] = subscripted_equation


    def compile(self):
        print('\npre-compile var check:')
        print('Stocks:', self.stocks)
        print('Flows:', self.flows)
        print('Aux:', self.auxiliaries)

        print('Definitions:')
        from pprint import pprint
        pprint(self.var_eqns)
        print('\nstarting compile...')
        
        # self.var_eqns_compiled = deepcopy(self.var_eqns)
        self.var_eqns_compiled = dict()
        
        # built-in variables
        self.var_eqns_compiled['TIME'] = self.current_time
        
        # stocks
        self.generate_stock_cld(self.stocks, self.cld_init_stocks)
        # flows+auxiliaries
        self.generate_flow_cld(self.flows+self.auxiliaries, self.cld_flows)

        print('\nvar elements:')
        pprint(self.var_elements)
        print('\ncompiled equations:')
        pprint(self.var_eqns_compiled)

        for var_element, compiled_equation in self.var_eqns_compiled.items():
            if type(compiled_equation) is str:
                print('ccc0', var_element, compiled_equation)
                var, element = self.reverse_var_element(var_element)
                print('ccc1', var, element)
                if var in self.var_elements.keys(): # this var is arrayed so need to make sure all arrayed variables in its equation come with element
                    cause_vars = dict()
                    success = False
                    while not success:
                        try:
                            eval(compiled_equation, cause_vars)
                            success = True
                        except NameError as e:
                            s = e.args[0]
                            p = s.split("'")[1]
                            print('ccc1.1', var, compiled_equation, p)
                            if p in self.var_eqns_compiled.keys(): # this means there will not be a problem in finding it in runtime
                                cause_vars[p] = 1
                            else: # this is when equation includes AT but AT is arrayed so only at[ele] exists in equation
                                compiled_equation = re.sub('(?<!_)'+p+'(?!_)', p+element, compiled_equation)
                                self.var_eqns_compiled[var_element] = compiled_equation
                                print('ccc1.1.2', var, compiled_equation)
                                cause_vars[p+element] = 1
        
        print('\ncompiled equations2:')
        pprint(self.var_eqns_compiled)

        self.if_compiled = True

    ############
    # Simulation
    ############

    # This development of evaluation of conditionals is not good as it requries too many callbacks.
    # Now it seems an integrated parser for all (expression, var name, etc is still the way to go.  3 Sept 2022

    # def evaluation_conditional(self, expression, gb, lc):
    #     # print('calc h1')
    #     con_ifs = list() # this is a stack - last in (append) first out (pop)

    #     expression_1 = expression

    #     # find the last IF
    #     expression_0 = expression_1.split('IF', 1)[0]
    #     while 'IF' in expression_1:
    #         if_then_else = expression_1.split('IF', 1)[1]
    #         print('if_then_else', if_then_else)
    #         con_ifs.append(if_then_else)
    #         expression_1 = if_then_else
        
    #     print('con_ifs', con_ifs)

    #     while len(con_ifs) != 0:
    #         print('vvv', con_ifs)
    #         ifthenelse = con_ifs.pop()
    #         # find the first THEN and ELSE after the last IF
    #         con_then = ifthenelse.split('THEN', 1)[1].split('ELSE', 1)[0]
    #         print('con_then', con_then)
    #         con_else = ifthenelse.split('THEN', 1)[1].split('ELSE', 1)[1]
    #         print('con_else0', con_else)
    #         # after else, there could be other THEN or ELSE from parent IF-THEN-ELSE statements. consider IF (IF THEN ELSE) THEN ELSE
    #         # as we are sure there is no more IFs, this con_else must end by next THEN or ELSE
    #         if 'THEN' in con_else:
    #             con_else = con_else.split('THEN', 1)[0]
    #             print('con_else1', con_else)
    #         elif 'ELSE' in con_else:
    #             con_else = con_else.split('ELSE', 1)[0]
    #             print('con_else2', con_else)

    #         con_if = ifthenelse.split('THEN', 1)[0]
    #         print('con_if', con_if)
            
    #         print('bool0')
    #         if bool(self.evaluation(con_if, gb=gb, lc=lc)):
    #             print('bool1')
    #             value = self.evaluation(con_then, gb=gb, lc=lc)
    #         else:
    #             print('boo2')
    #             value = self.evaluation(con_else, gb=gb, lc=lc)
        
    #         print('Evaled con:', expression, 'outcome:', value)

    #         if len(con_ifs) != 0:
    #             con_ifs[-1] =  con_ifs[-1].replace('IF'+con_if+'THEN'+con_then+'ELSE'+con_else, str(value))

    #     print('to return', expression_0+str(value))
    #     return expression_0 + str(value)
            
    def evaluation(self, expression, gb, lc):
        print('eee', expression)
        try:
            print('eee0', expression)
            value = eval(expression, gb, lc)
        except TypeError as e: # stock = 10
            if type(expression) in [int, float, np.int64, np.float64]:
                value = expression
            else:
                print('TypeError for {} - {}'.format(expression, type(expression)))
                raise e
        
        except NameError as e: # could be IF a > b --compile--> IFa>b, IFa is seen as a name
            value = self.evaluation_conditional(expression, gb=gb, lc=lc)
            # value = self.evaluation(con_value, gb=gb, lc=lc)

        except SyntaxError as e: # IF THEN ELSE
            err_expression = e.args[1][3]
            # print('bbb', err_expression, 'bbb')
            con_value = self.evaluation_conditional(err_expression, gb=gb, lc=lc)
            new_expression = expression.replace(err_expression, str(con_value))
            print('eee2', new_expression)
            value = self.evaluation(new_expression, gb=gb, lc=lc)
        
        return value

    def init_stock(self, var, diagram, calculated, if_stock):
        self.visited.add(var)
        # print('iii0, init stock:', var)
        
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

            var_v = self.evaluation(var_eqn, gb={}, lc=calculated)

            calculated[var] = var_v

            if if_stock:
                self.var_eqns_runtime[var] = str(var_v)
        

    def update_flow_aux(self, var, diagram, calculated=dict()):
        self.visited.add(var)
        dependencies = list(diagram.predecessors(var))
        # print('ufa1', var)
        try:
            elements = self.var_elements[var]
        
            for element in elements:
                var_ele = var+element
                var_ele_eqn = self.var_eqns_runtime[var_ele]

                for dependency in dependencies:
                    self.update_flow_aux(dependency, diagram, calculated)

                # print('ufa1.1', var_ele_eqn)
                var_ele_v = self.evaluation(var_ele_eqn, gb=self.var_eqns_runtime, lc=calculated)

                calculated[var_ele] = var_ele_v
                # print('ufa1.2', var_ele, var_ele_v)
        except: # var is not subscripted
            var_eqn = self.var_eqns_runtime[var]
            
            for dependency in dependencies:
                self.update_flow_aux(dependency, diagram, calculated)

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
        if not self.if_compiled:
            self.compile()

        print('\nstarting simulation...')

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
            self.var_eqns_runtime['TIME'] = self.current_time
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
        df_output = df_output.rename(columns=self.var_names_mapping)
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
        
        elif test_set == 3:
            model = Structure()
            print(model.evaluation_conditional('IF 1<2 THEN 3 ELSE 4', {}, {}))
            # print(model.evaluation_conditional('IF 1<2 THEN (IF 10<11 THEN 9 ELSE 8) ELSE 4', {}, {}))
            # print(model.evaluation_conditional('IF 1<2 THEN IF 10<11 THEN 9 ELSE 8 ELSE 4', {}, {}))

        elif test_set == 4:
            model = Structure()
            print(model.remove_bracket('a[ele1]+b[ele2]'))
        
        # elif test_set == 5:
        #     model = Structure()
        #     a0 = '1, Element_1, a'
        #     a= model.process_element(a0)
        #     b = model.reverse_element(a)
        #     print(b)

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

        
        elif test_set == 9:
            # model = Structure(from_xmile='BuiltInTestModels/Goal_gap.stmx')
            # model = Structure(from_xmile='BuiltInTestModels/Goal_gap_array.stmx')
            # model = Structure(from_xmile='BuiltInTestModels/Built_in_vars.stmx')
            model = Structure(from_xmile='BuiltInTestModels/IF_THEN_ELSE.stmx')
            # model = Structure(from_xmile='BuiltInTestModels/Graph_function.stmx ')
            # model = Structure(from_xmile='BuiltInTestModels/Array_cross_reference.stmx ')
            # model = Structure(from_xmile='BuiltInTestModels/Delays.stmx ')
            # model = Structure(from_xmile='BuiltInTestModels/Logic.stmx ')
            # model = Structure(from_xmile='BuiltInTestModels/Conveyor.stmx ')
            # model = Structure(from_xmile='BuiltInTestModels/Conveyor_leakage.stmx ')
            # model = Structure(from_xmile='TestModels\Endoscopy v3b.stmx')
            # model = Structure(from_xmile='TestModels\Elective Recovery Model flattened.stmx')
        

        model.compile()
        
        print('\nCLDs:')
        print('1 CLD flows:', model.cld_flows.edges(data=True))
        print('2 CLD init_stocks:', model.cld_init_stocks.edges(data=True)) # single stocks won't show here
        print('3 CLD update_stocks:', model.cld_update_stocks.edges(data=True))

        model.simulate(simulation_time=5, dt=0.5)
        
        # print('\nEquations compiled:')
        # print(model.var_eqns_compiled)

        print('\nResults')
        print(model.export_simulation_result().transpose())
        print('\nPASS\n')


    # for test_set in [1, 2, 4, 6, 7, 8]:
    for test_set in [9]:
        print('\n'+'*'*10)
        print('Test Set {}'.format(test_set))
        print('*'*10 + '\n')
        test(test_set)