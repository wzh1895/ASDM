from scipy import stats
from copy import deepcopy
from itertools import product
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint
import random
import math
import re

# random.seed(1)

class UidManager(object):
    def __init__(self):
        self.uid = 0

    def get_new_uid(self):
        self.uid += 1
        return self.uid

    def current(self):
        return self.uid


class NameManager(object):
    def __init__(self):
        self.stock_id = 0
        self.flow_id = 0
        self.variable_id = 0
        self.parameter_id = 0

    def get_new_name(self, element_type):
        if element_type == 'stock':
            self.stock_id += 1
            return 'stock_' + str(self.stock_id)
        elif element_type == 'flow':
            self.flow_id += 1
            return 'flow_' + str(self.flow_id)
        elif element_type == 'variable':
            self.variable_id += 1
            return 'variable_' + str(self.variable_id)
        elif element_type == 'parameter':
            self.parameter_id += 1
            return 'parameter_' + str(self.parameter_id)

class Data(object):
    def __init__(self, data_source):
        """
        data_source: a DataFrame structure

        """
        self.ts = data_source # we plan to convert more types of data_source to time_series


class DataFeeder(object):
    def __init__(self, data, from_step=1):
        """
        data: a Data object

        """
        self.ts_enumerator = enumerate(data.ts.values)
        self.from_step = from_step
        # self.n = 1

    def __call__(self, current_step): # make a datafeeder callable
        if current_step >= self.from_step:
            ix, dp = next(self.ts_enumerator)

            return(float(dp))
        else:
            return None


class ExtFunc(object):
    def __init__(self, external_function, arguments=None):
        """
        external_function: a function that takes an input and returns an output
        *args: arguments from within the model for that external_function

        """
        self.external_function = external_function
        self.args = list()
        if type(arguments) is list:
            self.args = arguments
        else:
            self.args.append(arguments)

    def evaluate(self, args):
        v = self.external_function(*args)
        return v


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
    # def __init__(self, sfd=None, uid_manager=None, name_manager=None, subscripts=None, uid_element_name=None, subscript_manager=None):
    def __init__(self, subscripts={'default_sub':['default_ele']}, from_xmile=None):
        # debug functionalities
        self.n_indentation = 0
        
        # Make alias for function names
        self.LINEAR = self.linear
        self.SUBTRACTION = self.subtraction
        self.DIVISION = self.division
        self.ADDITION = self.addition
        self.MULTIPLICATION = self.multiplication
        self.RBINOM = self.rbinom
        self.DELAY1 = self.delay1
        self.INIT = self.init

        self.function_names = [
            self.LINEAR, 
            self.SUBTRACTION, 
            self.DIVISION, 
            self.ADDITION, 
            self.MULTIPLICATION,
            self.RBINOM,
            self.DELAY1,
            self.INIT
        ]

        self.custom_functions = {
            'rbinom': self.rbinom,
            'delay1': self.delay1,
            'delay': self.delay,
            'DELAY': self.delay
            # 'init': self.init
        }

        self.time_related_functions = {
            'init': self.init,
            'INIT': self.init,  # Stella uses capital INIT
            'delay': self.delay,
            'DELAY': self.delay
        }

        # Polarity table for functions
        self.function_polarities = {
            self.LINEAR: {-1:'positive'},
            self.ADDITION: {-1:'positive'},
            self.MULTIPLICATION: {-1:'positive'},
            self.SUBTRACTION: {0:'positive', 1:'negative'},
            self.DIVISION: {0:'positive', 1:'negative'},
            self.RBINOM: {-1:'positive'},
            self.DELAY1: {0:'positive', 1:'negative', 2:'positive'},
            self.INIT: {0:'positive'}
        }

        # Define equation-text converter
        self.name_operator_mapping = {self.ADDITION: '+', self.SUBTRACTION: '-', self.MULTIPLICATION: '*', self.DIVISION: '/'}

        self.sfd = nx.DiGraph()
        self.__uid_manager = UidManager()
        self.__name_manager = NameManager()
        self.__uid_element_name = dict()
        self.__time_slice_values = dict() # a time-slice of name values
        self.__name_values = dict() # centralised simulation data manager
        self.__built_in_variables = dict()
        self.__expression_values = dict() # centralised data manager for init(expression)
        self.__name_external_data = dict() # centralised external data manager
        
        # Specify if subscript is used
        self.subscripts = subscripts

        # Define cumulative registers for time-related functions, such as delays
        self.delay_registers = dict()
        
        self.initial_time = 0
        self.current_time = self.initial_time
        self.current_step = 1 # object-wise global indicator for current simulation step. start from 1 (values after the 1st step)
        self.dt = 0.25
        self.simulation_time = 25
        self.time_units = 'Weeks'

        # Initialisation indicator
        self.is_initialised = False

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
                
                self.initial_time = sim_start
                self.current_time = self.initial_time
                self.dt = sim_dt
                self.simulation_time = sim_duration
                self.time_units = time_units

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
                    print(dims)
                    self.subscripts = dims
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
                        # print('Processing XMILE subscripted definition for:', var.get('name'))
                        var_dimensions = var.find('dimensions').findAll('dim')
                        # print('Found dimensions {}:'.format(var), var_dimensions)

                        var_dims = dict()
                        for dimension in var_dimensions:
                            name = dimension.get('name')
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
                            element_combination_texts = ['__cmm__'.join(cmb) for cmb in element_combinations]
                            # print('ec', element_combination_texts)

                            for ect in element_combination_texts:
                                var_subscripted_eqn[ect] =equation
                        return(var_subscripted_eqn)
                    else:
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
                            equation = var_equation
                        var_subscripted_eqn['nosubscript'] = equation
                        return(var_subscripted_eqn)
                        

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

            else:
                raise Exception("Specified model file does not exist.")

    def name_handler(self, name):
        return name.replace(' ', '_').replace('\\n', '_')

    # Define functions
    
    def linear(self, x, a=1, b=0):
        return a * float(x) + b

    def addition(self, *args):
        s = 0
        for arg in args:
            s = s + float(arg)
        return s
    
    def subtraction(self, x, y):
        return float(x) - float(y)
    
    def division(self, x, y):
        return float(x) / float(y)

    def multiplication(self, *args):
        product = 1
        for arg in args:
            product = product * float(arg)
        return product
    
    def rbinom(self, n, p):
        return stats.binom.rvs(int(n), p, size=1)[0]
    
    def delay(self, subscript, input, delay_time, initial_value=None):
        delay_time = float(delay_time)
        try:
            output = self.__name_values[self.current_time-delay_time][input][subscript]
            # print(input, 'aa')
        except KeyError: # current time < delay time
            if initial_value is not None: # initial value is supplied
                output = initial_value
                # print(input, 'bb')
            else: # initial values is not supplied
                try:
                    output = self.__name_values[self.initial_time][input][subscript]
                    # print(input, 'cc')
                except KeyError: # the delayed variable has not been calculated for once
                    output = self.calculate_experiment(input, subscript)
                    # print(input, 'dd')
        return output

    def delay1(self, input, delay_time, initial_value=None):
        delay_time = float(delay_time)
        if initial_value is not None:
            initial_value = float(initial_value) 
        # print(input, delay_time, initial_value)
        # create a register for the cumulative effect
        if input not in self.delay_registers.keys():
            self.delay_registers[input] = dict({self.DELAY1:0}) # delay1:cumulative value
        
        # initialise the delay register
        if self.current_step == 1:
            if initial_value is None: # initial value not specified, using input
                self.delay_registers[input][self.DELAY1] = delay_time*input
            else:  # when initial value is specified
                self.delay_registers[input][self.DELAY1] = delay_time*initial_value
        # decide the output value
        output = self.delay_registers[input][self.DELAY1]/delay_time
        
        # adjust the register's value
        self.delay_registers[input][self.DELAY1] += (input-output)*self.dt

        return output

    def init(self, subscript, var):
        # print('calculating init', var)
        # case 1: var is a variable in the model
        if var in self.sfd.nodes:
            v = self.__name_values[self.initial_time][var][subscript]
        # case 2: var is an expression
        else:
            if var in self.__expression_values.keys():
                v = self.__expression_values[var]
            else:
                equation = var
                value = None
                while type(value) not in [int, float, np.int64]: # if value has not become a number (taking care of the numpy data types)
                
                    # decide if the remaining expression is a time-related function (like init(), delay1(), delay())
                    func_names = re.findall(r"(\w+)[(].+[)]", str(var))
                    # print('0',func_names, 'in', equation)
                    if len(func_names) != 0:
                        func_name = func_names[0]
                        if func_name in self.time_related_functions.keys():
                            func_args = re.findall(r"\w+[(](.+)[)]", str(equation))
                            # print('1',func_args)
                            func_args_split = func_args[0].split(",")
                            # print('2',func_names[0], func_args_split)

                            # pass args to the corresponding time-related function
                            func_args_full = func_args_split + [subscript]
                            init_value = self.time_related_functions[func_names[0]](*func_args_full)
                            
                            # replace the init() parts in the equation with their values
                            init_value_str = str(init_value)
                            init_func_str = func_name+'\('+func_args[0]+'\)' # use '\' to mark '(' and ')', otherwise Python will see them as reg grammar

                            equation = re.sub(init_func_str, init_value_str, equation) # in case init() is a part of an equation, substitue init() with its value
                            # print('1', init_value_str, init_func_str, equation)
                    
                    try:
                        value = eval(str(equation), self.custom_functions)
                    except NameError as e:
                        s = e.args[0]
                        p = s.split("'")[1]
                        val = self.calculate_experiment(p, subscript)
                        val_str = str(val)
                        reg = '(?<!_)'+p+'(?!_)' # negative lookahead/behind to makesure p is not _p/p_/_p_
                        equation = re.sub(reg, val_str, equation)
                self.__expression_values[var] = value
                v = value
        return v

    def get_function_polarity(self, function, para_position):
        func_pol = self.function_polarities[function]
        if -1 in func_pol.keys():
            return func_pol[-1]
        else:
            return func_pol[para_position]

    def text_to_digit(self, text):
        try:
            digit = float(text)
            return digit
        except ValueError:
            return text

    def parsing_addition(self, equation):
        factors = equation.split('+')
        for i in range(len(factors)):
            factors[i] = self.text_to_digit(factors[i])
        return factors

    def parsing_multiplication(self, equation):
        factors = equation.split('*')
        for i in range(len(factors)):
            factors[i] = self.text_to_digit(factors[i])
        return factors

    def parsing_division(self, equation):
        factors = equation.split('/')
        for i in range(len(factors)):
            factors[i] = self.text_to_digit(factors[i])
        return factors

    def parsing_subtract(self, equation):
        factors = equation.split('-')
        for i in range(len(factors)):
            factors[i] = self.text_to_digit(factors[i])
        return factors

    def parsing_rbinom(self, equation):
        # rbinom takes 2 arguments: n (trials) and p (probability of success)
        # looks like rbinom(n, p)
        a = equation.split('(')[1]
        b = a.split(')')[0]
        c, d = b.split(',')
        c = c.strip()
        d = d.strip()
        factors = [self.text_to_digit(c), self.text_to_digit(d)]
        return factors
    
    def parsing_delay1(self, equation):
        # delay1 takes 2 (3) arguments: input and delay time
        # looks like delay1(input, delay)
        a = equation.split('(')[1]
        b = a.split(')')[0]
        factors = b.split(',')
        for i in range(len(factors)):
            factors[i] = factors[i].strip()
        return factors
    
    def parsing_init(self, equation):
        # init takes 2 arguments: input and subscript
        # looks like init(input:subscript.element)
        print('parsing init', equation)
        a = equation.split('(')[1]
        b = a.split(')')[0]
        factors = [b]
        return factors

    def equation_to_text(self, equation):
        if type(equation) == int or type(equation) == float:
            return str(equation)
        try:
            equation[0].isdigit()  # if it's a number
            return str(equation)
        except AttributeError:
            if equation[0] in [self.ADDITION, self.SUBTRACTION, self.MULTIPLICATION, self.DIVISION]:
                return str(equation[1]) + self.name_operator_mapping[equation[0]] + str(equation[2])
            elif equation[0] == self.LINEAR:
                return str(equation[1])

    def __add_element(self, element_name, element_type, flow_from=None, flow_to=None, x=0, y=0, equation=None, points=None, external=False, non_negative=False, leak=None):
        uid = self.__uid_manager.get_new_uid()

        # construct subscripted equation indexer
        # the 1st column stores the equation{function or initial value}, indexed by subscripts
        if type(equation) is Data:  # wrap external data into DataFeeder
            equation = DataFeeder(equation)

        # create a name_values binding for its simulation data
        v_dict = dict()
        for k in equation.keys():
            v_dict[k] = None
        self.__time_slice_values[element_name] = v_dict

        # create a node in the SFD graph
        self.sfd.add_node(element_name,
                        uid=uid,
                        element_type=element_type,
                        flow_from=flow_from,
                        flow_to=flow_to,
                        pos=[x, y],
                        equation=equation,
                        points=points,
                        external=external,
                        non_negative=non_negative,
                        leak=leak)
        # print('Engine: adding element:', element_name, 'equation:', equation)

        return uid

    # def __add_function_dependencies(self, element_name, function, subscript):  # confirm bunch of dependency found in a function
    def __add_function_dependencies(self, element_name, function):  # confirm bunch of dependency found in a function
        for i in range(len(function[1:])):
            from_variable = function[1:][i]
            if type(from_variable) == str:
                # print('Engine: adding causal link, from {} to {}, at subscript {}'.format(from_variable, element_name, subscript))
                print('Engine: adding causal link, from {} to {}, at subscript {}'.format(from_variable, element_name))
                self.__add_dependency(
                    from_element=from_variable,
                    to_element=element_name,
                    # subscript=subscript,
                    uid=self.__uid_manager.get_new_uid(),
                    polarity=self.get_function_polarity(function[0], i)
                    )

    # def __add_dependency(self, from_element, to_element, subscript, uid=0, angle=None, polarity=None, display=True):
    def __add_dependency(self, from_element, to_element, uid=0, angle=None, polarity=None, display=True):
        if not self.sfd.has_edge(from_element, to_element):
            self.sfd.add_edge(
                from_element,
                to_element,
                # subscripts=[subscript],
                uid=uid,
                angle=angle,
                length=0,  # for automated generation of CLD
                trend=1,  # for automated generation of CLD
                polarity=polarity,
                rad=0,  # for automated generation of CLD
                display=display)  # display as a flag for to or not to display
        
        # else:
        #     if subscript not in self.sfd.edges[from_element, to_element]['subscripts']:
        #         self.sfd.edges[from_element, to_element]['subscripts'].append(subscript)

    def __get_element_by_uid(self, uid):
        # print("Uid_Element_Name, ", self.uid_element_name)
        return self.sfd.nodes[self.__uid_element_name[uid]]

    def __get_element_name_by_uid(self, uid):
        return self.__uid_element_name[uid]

    def print_elements(self):
        print('Engine: All elements in this SFD:')
        print(self.sfd.nodes.data())

    def print_element(self, name):
        print('Engine: Attributes of element {}:'.format(name))
        print(self.sfd.nodes[name])

    def print_causal_links(self):
        print('Engine: All causal links in this SFD:')
        print(self.sfd.edges)

    def print_causal_link(self, from_element, to_element):
        print('Engine: dependency from {} to {}:'.format(from_element, to_element))
        print(self.sfd[from_element][to_element])

    def get_all_certain_type(self, element_type):
        # able to handle both single type and multiple types
        elements = list()
        if type(element_type) != list:
            element_types = [element_type]
        else:
            element_types = element_type

        for ele_tp in element_types:
            for node, attributes in self.sfd.nodes.data():
                try:
                    if attributes['element_type'] == ele_tp:
                        elements.append(node)
                except KeyError:
                    print('node:', node)
        # print(elements, "Found for", element_types)
        return elements

    def set_external(self, element_name):
        self.sfd.nodes[element_name]["external"] = True

    def __get_coordinate(self, name):
        """
        Get the coordinate of a specified variable
        :param name:
        :return: coordinate of the variable in a tuple
        """
        return self.sfd.nodes[name]['pos']

    # Simulate a structure based on a certain set of parameters
    def simulate(self, simulation_time=None, dt=None):
        if simulation_time is not None:
            self.simulation_time = simulation_time
        if dt is not None:
            self.dt = dt
        total_steps = int(self.simulation_time / self.dt)
        
        # have a dict for all visited (calculated) variables (F/V/P) in this model
        # this is also a buffer to store calculated value for variables, to avoid calculating the same variable multiple times.
        # ususally this is not a problem, but when random process is included, values from multiple calculations can be different.
        self.visited = dict()

        # Setp 1 initialise the state of the stocks if this is the first run

        if self.is_initialised == False:
            self.__built_in_variables['TIME'].append(self.initial_time) # initialization of stocks might require TIME
            self.init_stocks()
        else:
            self.update_stocks(self.dt)
        
        # Step 2
        from tqdm import tnrange
        for _ in tnrange(total_steps):
            self.update_states()
            self.current_step += 1  # update current_step counter
            self.current_time += self.dt
            self.__built_in_variables['TIME'].append(self.current_time)
            self.__name_values[self.current_time] = deepcopy(self.__time_slice_values)

            self.visited = dict()

            self.update_stocks(self.dt)
        
        # Step 3

        self.update_states()
        
    def init_stocks(self):
        self.__name_values[self.current_time] = deepcopy(self.__time_slice_values)
        for element in self.get_all_certain_type(['stock']):
            print('Initializing stock: {}'.format(element))
            # for ix in self.sfd.nodes[element]['equation'].sub_index:
            for ix in self.sfd.nodes[element]['equation'].keys():
                equation = self.sfd.nodes[element]['equation'][ix]
                # print('EQU', equation)

                # if the stock is a conveyor, extract its equation
                is_conveyor = False
                if type(equation) is Conveyor:
                    is_conveyor = True
                    conveyor = equation
                    equation = equation.equation

                value = self.calculate_experiment(equation, ix)
                
                self.__name_values[self.current_time][element][ix] = value

                if is_conveyor:
                    length = int(self.calculate_experiment(conveyor.length_time_units, ix) / self.dt)
                    # search for leaking flow
                    leak_fraction = None
                    for e in self.sfd.nodes:
                        if self.sfd.nodes[e]['element_type'] == 'flow':
                            if self.sfd.nodes[e]['leak']:
                                if self.sfd.nodes[e]['flow_from'] == element:
                                    leak_fraction_eqn = self.sfd.nodes[e]['equation'][ix]
                                    leak_fraction = self.calculate_experiment(leak_fraction_eqn, ix)
                    conveyor.initialize(length, value, leak_fraction)

        # set is_initialised flag to True
        print('All stocks initialised.')
        self.is_initialised = True

    def update_stocks(self, dt):
        # have a dictionary for flows and their values in this default_dt, to be added to or subtracted from stocks
        flows_dt = dict()

        # find all flows in the model
        for element in self.get_all_certain_type('flow'):  # loop through all flows in this SFD,
            flows_dt[element] = dict()  # make a position for it in the dict of flows_dt, initializing it with 0


        # calculate flows
        for flow in flows_dt.keys():
            for ix in self.sfd.nodes[flow]['equation'].keys():
                flows_dt[flow][ix] = dt * self.__name_values[self.current_time-self.dt][flow][ix]

        # calculating changes in stocks
        # have a dictionary of affected stocks and their changes, since one flow could affect 2 stocks.
        affected_stocks = dict()
        for flow in flows_dt.keys():
            successors = list(self.sfd.successors(flow))  # successors of a flow into a list
            # print('Successors of {}: '.format(flow), successors)

            for successor in successors:
                if self.sfd.nodes[successor]['element_type'] == 'stock':  # flow may also affect elements other than stock
                    
                    if successor not in affected_stocks.keys():
                        affected_stocks[successor] = dict() # a dict for subscripts
                    
                    in_out_factor = 1  # initialize
                    if self.sfd.nodes[flow]['flow_from'] == successor:  # if flow influences this stock negatively
                        in_out_factor = -1
                    elif self.sfd.nodes[flow]['flow_to'] == successor:  # if flow influences this stock positively
                        in_out_factor = 1
                    else:
                        print("Engine: Strange! {} seems to influence {} but not found in graph's attributes.".format(flow, successor))
                    
                    # because the connection between a flow and a stock does not vary across subscripts, we only consider indexer from here
                    for ix in self.sfd.nodes[flow]['equation'].keys():
                        if self.sfd.nodes[flow]['flow_from'] == successor and type(self.sfd.nodes[successor]['equation'][ix]) is Conveyor:
                            pass
                        elif ix in affected_stocks[successor].keys():  # this stock may have been added by other flows
                            affected_stocks[successor][ix] += flows_dt[flow][ix] * in_out_factor
                        else:
                            affected_stocks[successor][ix] = flows_dt[flow][ix] * in_out_factor

        # updating affected stocks values
        for stock, subscripted_delta_values in affected_stocks.items():
            for ix in self.__name_values[self.current_time][stock].keys():
                delta_value = subscripted_delta_values[ix]
                if type(self.sfd.nodes[stock]['equation'][ix]) is Conveyor:
                    self.sfd.nodes[stock]['equation'][ix].inflow(delta_value)
                    new_value = self.sfd.nodes[stock]['equation'][ix].level()
                else:
                    current_value = self.__name_values[self.current_time-self.dt][stock][ix]
                    # print(delta_value, type(delta_value), current_value, type(current_value))
                    new_value = delta_value + current_value
                if self.sfd.nodes[stock]['non_negative'] is True and new_value < 0:# if the stock in non-negative and the new value is below 0
                    # print("Stock {} is non-negative but its value is going below 0.".format(stock))
                    self.__name_values[self.current_time][stock][ix] = 0
                    # outflows from this stock should also be 0
                    
                else:
                    self.__name_values[self.current_time][stock][ix] = new_value

        # for those stocks not affected, extend its 'values' by its current value
        for node in self.sfd.nodes:
            if self.sfd.nodes[node]['element_type'] == 'stock':
                if node not in affected_stocks.keys():
                    for ix in self.__name_values[self.current_time][node].keys():
                        self.__name_values[self.current_time][node][ix] = self.__name_values[self.current_time-self.dt][node][ix]
    
    def update_states(self):
        # print('us0')
        """
        Core function for simulation. Calculating all flows and adjust stocks accordingly based on recursion.
        """

        # have a dictionary for flows and their values in this default_dt, to be added to or subtracted from stocks
        flows = dict()

        # find all flows in the model
        for element in self.get_all_certain_type('flow'):  # loop through all elements in this SFD,
            flows[element] = dict()  # make a position for it in the dict of flows, initializing it with 0

        leaking_flows = list() # tier 1
        convey_outflows = list() # tier 2
        non_leaking_flows = list() # tier 3
        # calculate flows
        
        # prioritise leaking flows
        for flow in flows.keys():
            if self.sfd.nodes[flow]['leak']:
                leaking_flows.append(flow)
            else:
                flow_from = self.sfd.nodes[flow]['flow_from']
                if flow_from is not None:
                    for ix in self.sfd.nodes[flow_from]['equation'].keys():
                        flow_from_equation = self.sfd.nodes[flow_from]['equation'][ix]
                        if type(flow_from_equation) is Conveyor:
                            convey_outflows.append(flow)
                        else:
                            non_leaking_flows.append(flow)
                        break
                else:
                    non_leaking_flows.append(flow)

        # tier 1
        for flow in leaking_flows:
            for ix in self.sfd.nodes[flow]['equation'].keys():
                
                # print('f leak:', flow)
                flow_leak_fraction_eqn = self.sfd.nodes[flow]['equation'][ix]
                flow_leak_fraction = self.calculate_experiment(flow_leak_fraction_eqn, ix)
                # print('f leak frac:', flow_leak_fraction)
                v = self.sfd.nodes[self.sfd.nodes[flow]['flow_from']]['equation'][ix].leak_linear(flow_leak_fraction) / self.dt
                # print('f leak v:', flow, v)
                
                flows[flow][ix] = v
                # as this flow is not calculated through calculate(), we manually register its value to __name_values
                self.__name_values[self.current_time][flow][ix] = v
            
            self.visited[flow] = flows[flow]  # save calculated flow to buffer
        
        # tier 2
        for flow in convey_outflows:
            if type(self.sfd.nodes[flow]['equation']) is dict:
                for ix in self.sfd.nodes[flow]['equation'].keys():
                    flow_from = self.sfd.nodes[flow]['flow_from']
                    # print('f out:', flow)
                    v = self.sfd.nodes[flow_from]['equation'][ix].outflow() / self.dt # the conveyor outputs the value of this DT but we need a flow value for the whole time unit (DT = 1)
                    # print('f out v:', v)
                    flows[flow][ix] = v
                    # as this flow is not calculated through calculate(), we manually register its value to __name_values
                    self.__name_values[self.current_time][flow][ix] = v
            else:
                flow_from = self.sfd.nodes[flow]['flow_from']
                # print('f out:', flow)
                v = self.sfd.nodes[flow_from]['equation'].outflow() / self.dt # the conveyor outputs the value of this DT but we need a flow value for the whole time unit (DT = 1)
                # print('f out v:', v)
                flows[flow] = v
                # as this flow is not calculated through calculate(), we manually register its value to __name_values
                self.__name_values[self.current_time][flow] = v  
            self.visited[flow] = flows[flow]  # save calculated flow to buffer

        # tier 3
        for flow in non_leaking_flows:
            # print('fff3', flow, self.sfd.nodes[flow])
            for ix in self.sfd.nodes[flow]['equation'].keys():
                v = self.calculate_experiment(flow, ix)
                # print('fff3.1', flow, ix, v)
                flows[flow][ix] = v # TODO: Flow behaviour needs more attention. (1) flow value should not exceed the amount remaining in the stock/dt (2) when multiple flows compete for one stock, the early-added gets higher priority
        
            self.visited[flow] = flows[flow]  # save calculated flow to buffer

        # print('All flows default_dt:', flows_dt)

        # calculate all not visited variables and parameters in this model, in order to update their value list
        # because all flows and stocks should be visited sooner or later, only V and P are considered.
        
        for element in self.get_all_certain_type(['auxiliary']):
            if element not in self.visited.keys():
                for ix in self.sfd.nodes[element]['equation'].keys():
                    # if ix not in self.visited[element].keys():
                    self.calculate_experiment(element, ix)

    @staticmethod
    def process_expression(expression):
        expression_1 = expression
        while '[' in expression_1:
            expression_1_a, expression_1_b = expression_1.split('[', 1)
            expression_1_b, expression_1_c = expression_1_b.split(']', 1)
            expression_1_b = expression_1_b.replace(' ', '').replace(',', '__cmm__')
            expression_1 = expression_1_a + '__ele1__' + expression_1_b + '__ele2__' + expression_1_c
        expression = expression_1
        # print('processed equation:', expression)
        return expression
    
    @staticmethod
    def process_subscript(subscript):
        subscript = subscript.replace(',', '__cmm__').replace(' ', '')
        return subscript

    
    def calculate_experiment(self, expression, subscript):
        self.n_indentation += 1
        # print(self.n_indentation*'\t' + 'processing expression:', expression)

        # check if the expression contains '[]' - cross reference of arrays
        if type(expression) is str and '[' in expression:
            # print('process a')
            expression = self.process_expression(expression)
            # print('process expression with subscripts:', expression)
            # expression = expression.replace('[', '__ele1__').replace(']', '__ele2__').replace(',', '__cmm__')

        # check if the expression is abc__ele1__subscript__ele2__
        if type(expression) is str and '__ele1__' in expression:
            # print('process b')
            expression, subscript = expression.split('__ele1__', 1)
            subscript, remaining = subscript.split('__ele2__', 1)
            # print('process b', expression, subscript, remaining)

        # check if this value has been calculated and stored in buffer
        # when initialising stock values, there's no 'self.visited' yet
        if expression in self.visited.keys():
            # print('calc a')
            # print(self.visited[expression])
            if subscript in self.visited[expression].keys():
                # print(self.visited[expression][subscript])
                self.n_indentation -= 1
                return self.visited[expression][subscript]
            else:
                pass
        
        # check if this is a system variable like TIME
        if expression in self.__built_in_variables.keys():
            # print('calc b')
            self.n_indentation -= 1
            return self.__built_in_variables[expression][-1]

        elif type(expression) in [int, float, np.int64]:
            # print('calc d')
            self.n_indentation -= 1
            return expression
            
        # check if this value has been calculated
        # if expression in self.__name_values[self.current_time].keys(): # name might be equation such as 'a > b'
        # if expression in self.sfd.nodes and self.__name_values[self.current_time][expression][subscript] is not None:
        if expression in self.sfd.nodes and subscript in self.__name_values[self.current_time][expression].keys() and self.__name_values[self.current_time][expression][subscript] is not None:
            #  and self.__name_values[self.current_time][expression][subscript] is not None: # case 1: subscripted var
                # print('calc c', expression, subscript)
                # print('calc c', self.__name_values[self.current_time][expression])
                self.n_indentation -= 1
                return self.__name_values[self.current_time][expression][subscript]

        # Goal --> Gap, gap is subscripted but goal is not; access to goal's value through gap's subscript (calc c) fails
        if  expression in self.sfd.nodes and 'nosubscript' in self.__name_values[self.current_time][expression].keys() and self.__name_values[self.current_time][expression]['nosubscript'] is not None:
                # print('calc e', expression, subscript)
                # print('calc e', self.__name_values[self.current_time][expression])
                self.n_indentation -= 1
                return self.__name_values[self.current_time][expression]['nosubscript']

        elif expression in self.sfd.nodes and self.sfd.nodes[expression]['element_type'] == 'stock':
            # print('calc g')
            # print(self.__name_values.keys())
            if not self.is_initialised:
                stock_eqn = self.sfd.nodes[expression]['equation'][subscript]
                if type(stock_eqn) is Conveyor:
                    value = self.calculate_experiment(stock_eqn.equation, subscript)
                else:
                    value = self.calculate_experiment(stock_eqn, subscript)
            else:
                value = self.__name_values[self.current_time][expression][subscript]
            self.n_indentation -= 1
            return value
        
        # expression is the name of a leaking flow
        elif expression in self.sfd.nodes and self.sfd.nodes[expression]['leak']:
            if not self.is_initialised: # it's currently in the process of init_stocks, no leaks have been calculated
                self.n_indentation -= 1
                return 0 # leaks are currently all 0

        else:  # calculation is needed
            # print(self.n_indentation*'\t' + 'calc h:', expression, subscript)
            if expression in self.sfd.nodes: # careful - outflows from conveyors have eqn 0 but should not be dealt with here
                # print(self.n_indentation*'\t' + 'calc h1:', self.sfd.nodes[expression]['equation'])
                if subscript in self.sfd.nodes[expression]['equation'].keys():
                    # print(self.n_indentation*'\t' + 'calc h1.1')
                    equation = self.sfd.nodes[expression]['equation'][subscript]
                else:
                    # print(self.n_indentation*'\t' + 'calc h1.2')
                    equation = self.sfd.nodes[expression]['equation']['nosubscript']
            else:
                equation = expression

            # replace abc[subscript] -> abc__ele1__subscript__ele2__
            if type(equation) is str:
                equation = self.process_expression(equation)
            
            # pre-process expression to match Python syntax
            # replace '=' with '=='
            if type(equation) is str:
                reg = "(?<!(<|>|=))=(?!(<|>|=))"
                equation = re.sub(reg, "==", equation)
            
            value = None

            while type(value) not in [int, float, np.int64, bool]: # if value has not become a number (taking care of the numpy data types)
                
                # check if the variable is a graph function which has an additonal layer that transforms the equation outcome
                if type(equation) is GraphFunc:
                    # print('calc h0')
                    input = self.calculate_experiment(equation.eqn, subscript)
                    equation = equation(input)
                
                # check if this is a conditional statement
                elif type(equation) is str and len(equation) > 2 and equation[:2] == 'IF':
                    # print('calc h1')
                    con_if = equation[2:].split('THEN')[0]
                    # print('con_if:', con_if)
                    con_then = equation[2:].split('THEN')[1].split('ELSE')[0]
                    # print(self.n_indentation*'\t'+'con_then:', con_then)
                    con_else = equation[2:].split('THEN')[1].split('ELSE')[1]
                    # print('con_else:', con_else)
                    if con_if is None:
                        raise Exception("Condition IF cannot be None.")
                    elif con_then is None and con_else is None:
                        raise Exception("Condition THEN and ELSE cannot both be None")
                    else:
                        con_eval = self.calculate_experiment(con_if, subscript)
                        # print('con_eval', con_eval)
                        if type(con_eval) is not bool:
                            raise TypeError("Condition IF must be True of False")
                        elif con_eval:
                            # print(self.n_indentation*'\t'+'here')
                            con_outcome = self.calculate_experiment(con_then, subscript)
                        else:
                            con_outcome = self.calculate_experiment(con_else, subscript)
                    con_outcome = str(con_outcome)
                    # print(self.n_indentation*'\t'+'con_outcome:', con_outcome)
                    con_statement = 'IF'+con_if+'THEN'+con_then+'ELSE'+con_else
                    # print('Constat', con_statement)
                    equation = equation.replace(con_statement, con_outcome)
                    # print('equation after con', equation)
                
                elif type(equation) is str and len(equation) > 3 and 'AND' in equation:
                    # print('calc h2.1')
                    and_0, and_1 = equation.split('AND')
                    # print('and_0', and_0)
                    # print('and_1', and_1)
                    and_0_eval = self.calculate_experiment(and_0, subscript)
                    if type(and_0_eval) is not bool:
                        raise TypeError
                    and_1_eval = self.calculate_experiment(and_1, subscript)
                    if type(and_1_eval) is not bool:
                        raise TypeError
                    if and_0_eval and and_1_eval:
                        and_outcome = 'True'
                    else:
                        and_outcome = 'False'
                    equation = and_outcome

                elif type(equation) is str and len(equation) > 2 and 'OR' in equation:
                    # print('calc h2.2')
                    and_0, and_1 = equation.split('OR')
                    # print('and_0', and_0)
                    # print('and_1', and_1)
                    and_0_eval = self.calculate_experiment(and_0, subscript)
                    if type(and_0_eval) is not bool:
                        raise TypeError
                    and_1_eval = self.calculate_experiment(and_1, subscript)
                    if type(and_1_eval) is not bool:
                        raise TypeError
                    if and_0_eval or and_1_eval:
                        equation = 'True'
                    else:
                        equation = 'False'

                # check if there are time-related functions in the equation
                elif type(equation) is str and re.findall(r"(\w+)[(].+[)]", str(equation)):
                    func_names = re.findall(r"(\w+)[(].+[)]", str(equation))
                    # print('calc h3')
                    for func_name in func_names:
                        # print(0, equation)
                        # print('calc h3 0', func_name)
                        if func_name in self.time_related_functions.keys():
                            func_args = re.findall(r"\w+[(](.+)[)]", str(equation))
                            # print('1',func_args)
                            func_args_split = func_args[0].split(",")
                            # print('2',func_names[0], func_args_split)

                            # pass args to the corresponding time-related function
                            func_args_full = [subscript] + func_args_split
                            func_value = self.time_related_functions[func_names[0]](*func_args_full)
                            
                            # replace the init() parts in the equation with their values
                            func_value_str = str(func_value)
                            func_str = func_name+'('+func_args[0]+')'
                            equation = equation.replace(func_str, func_value_str) # in case init() is a part of an equation, substitue init() with its value
                            # print('3', equation)

                '''
                Until here, all we want to have is an modified equation suitable for evaluation.
                '''

                try:
                    # print(self.n_indentation*'\t' + 'eval:', equation)
                    value = eval(str(equation), self.custom_functions)
                    # print(self.n_indentation*'\t'+'Evaluated {} = {}'.format(equation, value))

                except NameError as e:
                    s = e.args[0]
                    p = s.split("'")[1]
                    # print(self.n_indentation*'\t'+'NameError {} for equation: {}'.format(p,equation))
                    val = self.calculate_experiment(p, subscript)
                    # print(self.n_indentation*'\t'+'Calculated value for {}: {}'.format(p, val))
                    val_str = str(val)
                    reg = '(?<!_)'+p+'(?!_)' # negative lookahead/behind to makesure p is not _p/p_/_p_
                    equation = re.sub(reg, val_str, equation)
                
                except ZeroDivisionError as e:
                    raise e
            
            if expression in self.sfd.nodes:

                # Noted 21/8/2022: This is not OK. Positive check of non-negative stocks should be performed when updating the stock, not calculating flows. -10->[0]-10-> is legitimate but not pass here.
                # non-negative control of flows
                if self.sfd.nodes[expression]['element_type'] == 'flow':
                    if self.sfd.nodes[expression]['non_negative']:
                        if value < 0:
                            # print('Flow {} is non-negative but is going under 0 to {}.'.format(expression, value))
                            value = 0
                
                # leak flows have eqn but it's not for value, but for leak fraction. Check before register
                if self.sfd.nodes[expression]['leak']:
                    pass
                else:
                    self.__name_values[self.current_time][expression][subscript] = value

                try:  # when initialising stock values, there's no 'self.visited' yet
                    # print('calc updating vidited', expression, subscript, value)
                    if expression in self.sfd.nodes: # only register those that are variable names in the model
                        if expression not in self.visited.keys():
                            self.visited[expression] = dict()
                        if subscript not in self.visited[expression].keys():
                            self.visited[expression][subscript] = value
                except AttributeError:
                    pass
            
            self.n_indentation -= 1
            return value


    def clear_last_run(self):
        """
        Clear values for all nodes
        :return:
        """
        self.current_step = 1
        self.current_time = self.initial_time

        for name in self.__name_values.keys():
            self.__name_values[name] = dict()
        
        self.__built_in_variables['TIME'] = list()
        
        # reset delay_registers
        self.delay_registers = dict()

        # reset is_initialised flag
        self.is_initialised = False

    # Add elements on a stock-and-flow level (work with model file handlers)
    def add_stock(self, name=None, equation=None, x=0, y=0, non_negative=False):
        """
        :param name: name of the stock
        :param equation: initial value
        :param x: x
        :param y: y
        :return: uid of the stock
        """
        if equation is not None:
            # equation = self.text_to_equation(equation)
            uid = self.__add_element(name, element_type='stock', x=x, y=y, equation=equation, non_negative=non_negative)
            # # the initial value of a stock should be added to its simulation value DataFrame
            # for ix in self.__name_values[name].sub_index:
            #     self.__name_values[name][ix].append(equation)
        else:
            raise TypeError("Equation should not be None.")
        # print('Engine: added stock:', name, 'to graph.')
        return uid

    def add_flow(self, name=None, equation=None, x=0, y=0, points=None, flow_from=None, flow_to=None, leak=None, non_negative=False):
        if name is None:
            name = self.__name_manager.get_new_name(element_type='flow')
        if equation is not None:
            uid = self.__add_element(name,
                                    element_type='flow',
                                    flow_from=flow_from,
                                    flow_to=flow_to,
                                    x=x,
                                    y=y,
                                    equation=equation,
                                    points=points,
                                    leak=leak,
                                    non_negative=non_negative)

        self.connect_stock_flow(name, flow_from=flow_from, flow_to=flow_to)
        # print('Engine: added flow:', name, 'to graph.')
        return uid

    def connect_stock_flow(self, flow_name, flow_from=None, flow_to=None):
        """
        Connect stock and flow.
        :param flow_name: The flow's name
        :param flow_from: The stock this flow coming from
        :param flow_to: The stock this flow going into
        :return:
        """
        # If the flow influences a stock, create the causal link
        # we assume the connection between a flow and stock holds across all subscripts
        # for ix in self.sfd.nodes[flow_name]['equation'].sub_index:
        if flow_from is not None:  # Just set up
            self.sfd.nodes[flow_name]['flow_from'] = flow_from
            # self.__add_dependency(flow_name, flow_from, subscript=ix, display=False, polarity='negative')
            self.__add_dependency(flow_name, flow_from, display=False, polarity='negative')
        if flow_to is not None:  # Just set up
            self.sfd.nodes[flow_name]['flow_to'] = flow_to
            # self.__add_dependency(flow_name, flow_to, subscript=ix, display=False, polarity='positive')
            self.__add_dependency(flow_name, flow_to, display=False, polarity='positive')

    def disconnect_stock_flow(self, flow_name, stock_name):
        """
        Disconnect stock and flow
        :param flow_name: The flow's name
        :param stock_name: The stock this flow no longer connected to
        :return:
        """
        if self.sfd.nodes[flow_name]['flow_from'] == stock_name:
            self.sfd.remove_edge(flow_name, stock_name)
            self.sfd.nodes[flow_name]['flow_from'] = None
        if self.sfd.nodes[flow_name]['flow_to'] == stock_name:
            self.sfd.remove_edge(flow_name, stock_name)
            self.sfd.nodes[flow_name]['flow_to'] = None

    def add_aux(self, name=None, equation=None, x=0, y=0):
        
        if equation is not None:
            uid = self.__add_element(
                name,
                element_type='auxiliary',
                x=x,
                y=y,
                equation=equation
            )
        else:
            raise TypeError("Equation should not be None.")

        return uid

    def get_element_equation(self, name):
        if self.sfd.nodes[name]['element_type'] == 'stock':
            # if the node is a stock
            return self.sfd.nodes[name]['value'][0]  # just return its first value (initial).
        elif self.sfd.nodes[name]['function'] is None:
            # if the node does not have a function and not a stock, then it's constant
            return self.sfd.nodes[name]['value'][0]  # use its latest value
        else:  # it's not a constant value but a function  #
            return self.sfd.nodes[name]['function']

    def replace_element_equation(self, name, new_equation, subscripts=None):
        """
        Replace the equation of a variable.
        :param name: The name of the variable
        :param new_equation: The new equation
        :return:
        """
        # step 1: if not stock, remove all incoming connectors into this variable (node)
        #         if stock, just reset it's (initial) value
        # step 2: replace the equation of this variable in the graph representation
        # step 3: confirm connectors based on the new equation (only when the new equation is a function not a number
        # print("Engine: Replacing equation of {}".format(name))
        
        if new_equation is not None:
            new_equation_parsed = self.text_to_equation(new_equation)
        else:
            raise Exception("New equation for replacing could not be None.")
        
        # step 1:
        if self.sfd.nodes[name]['element_type'] != 'stock':
            to_remove = list()
            for u, v in self.sfd.in_edges(name):
                # print("In_edge found:", u, v)
                to_remove.append((u, v))
            self.sfd.remove_edges_from(to_remove)
            if len(to_remove) > 0:
                print("Engine: Edges removed.")
                pass

        # step 2:
        # retrieve or generate indexer
        if subscripts is None:
            subscripts = self.sfd.nodes[name]['equation'].keys()
        else:
            if type(subscripts) is not dict:
                raise TypeError("Subscripts should be described using a dictironary.")
            subscript_names = list()
            subscript_elements = list()
            for name, elements in subscripts.items():
                if type(elements) is not list:
                    raise TypeError("Subscripts elements should be described using a list.")
                subscript_names.append(name)
                subscript_elements.append(elements)
            subscripts = pd.MultiIndex.from_product(subscript_elements, subscript_names)
        
        for ix in subscripts:
            self.sfd.nodes[name]['equation'][ix] = new_equation
        
            # step 3:
            if type(new_equation_parsed[0]) not in [int, float]:
                # If new equation (parsed list) does not starts with a number, 
                # it's not a constant value, but a function
                if type(new_equation_parsed) is not str:
                    self.__add_function_dependencies(name, new_equation_parsed, ix)
                    # print("Engine: New edges created.")

    def remove_element(self, name):
        """
        Delete an element
        :param name:
        :return:
        """
        self.sfd.remove_node(name)
        print("Engine: {} is removed from the graph.".format(name))

    def remove_connector(self, from_element, to_element):
        self.sfd.remove_edge(from_element, to_element)

    # Reset a structure
    def reset_a_structure(self):
        self.sfd.clear()

    # Return a behavior
    def get_element_simulation_result(self, name, subscripts=None):
        if subscripts is None:
            subscripts = self.__name_values[name].keys()
        result = self.__name_values[name]
        result_dict = dict()
        for ix in subscripts:
            result_dict[ix] = result[ix]  # return a list of values; not a pandas DataFrame
        if len(result_dict) == 1:
            result_dict = next(iter(result_dict.values()))
        return result_dict

    # Return all simulation results as a pandas DataFrame
    def export_simulation_result(self):
        full_result = dict()
        for time, values in self.__name_values.items():
            # time_result = pd.DataFrame(index=None)
            time_result = dict()
            time_result[self.time_units] = time
            for node in self.sfd.nodes:
                
                # process variable name to match Stella standard
                node_output = node.replace('_', ' ')
                if node_output[0].isdigit():
                    node_output = '"{}"'.format(node_output)
                
                for ix in self.__name_values[time][node].keys():
                    if ix == 'nosubscript':
                        time_result[node_output] = values[node]['nosubscript']
                    else:
                        time_result[node_output+'[{}]'.format(ix.replace('_', ' '))] = values[node][ix]
            full_result[time] = time_result
        df_full_result = pd.DataFrame.from_dict(full_result).transpose()
        df_full_result.set_index(self.time_units, inplace=True)
        return df_full_result

    # Draw results
    def display_results(self, variables=None, dpi=100, rtn=False):
        if variables is None:
            variables = list(self.sfd.nodes)

        figure_0 = plt.figure(
            figsize=(8, 6),
            facecolor='whitesmoke',
            edgecolor='grey',
            dpi=dpi
        )

        plt.xlabel('Steps {} (Time: {} / Dt: {})'.format(
            int(self.simulation_time / self.dt),
            self.simulation_time,
            self.dt))
        plt.ylabel('Behavior')
        y_axis_minimum = 0
        y_axis_maximum = 0
        for name in variables:
            if self.sfd.nodes[name]['external'] is True:
                values = self.data_feeder.buffers_list[name]
            elif self.sfd.nodes[name]['value'] is not None:  # otherwise, dont's plot
                values = self.sfd.nodes[name]['value']
            else:
                continue  # no value found for this variable
            # print("Engine: getting min/max for", name)
            # set the range of axis based on this element's behavior
            # 0 -> end of period (time), 0 -> 100 (y range)

            name_minimum = min(values)
            name_maximum = max(values)
            if name_minimum == name_maximum:
                name_minimum *= 2
                name_maximum *= 2
                # print('Engine: Centered this straight line')

            if name_minimum < y_axis_minimum:
                y_axis_minimum = name_minimum

            if name_maximum > y_axis_maximum:
                y_axis_maximum = name_maximum

            # print("Engine: Y range: ", y_axis_minimum, '-', y_axis_maximum)
            plt.axis([0, self.simulation_time / self.dt, y_axis_minimum, y_axis_maximum])
            # print("Engine: Time series of {}:".format(name))
            # for i in range(len(values)):
            #     print("Engine: {0} at DT {1} : {2:8.4f}".format(name, i+1, values[i]))
            plt.plot(values, label=name)
        plt.legend()
        if rtn:  # if called from external, return the figure without show it.
            return figure_0
        else:  # otherwise, show the figure.
            plt.show()

    # Draw CLD with auto generated layout
    def draw_cld_with_auto_gen_layout(self, rtn=False):
        figure_3 = plt.figure(num='cld_auto_layout')
        subplot1 = figure_3.add_subplot(111)

        pp = pprint.PrettyPrinter(indent=4)

        # generate edge colors from polarities
        edge_attrs_polarity = nx.get_edge_attributes(self.sfd, 'polarity')
        custom_edge_colors = list()
        for edge, attr in edge_attrs_polarity.items():
            # print(attr)
            color = 'dodgerblue'  # black
            if attr == 'negative':
                color = 'k'  # black
            custom_edge_colors.append(color)

        # TODO: Develop the layout algorithm for CLDs

        # get all loops
        loop_gen = nx.simple_cycles(self.sfd)

        loops = dict()
        loop_n = 1
        longest_loop = 1
        longest_length = 1

        for loop in loop_gen:
            print(loop)
            loop_length = len(loop)

            # find (one of) the longest loop(s)
            if loop_length >= longest_length:
                longest_loop = loop_n
                longest_length = loop_length

            loops[loop_n] = {
                'variables': loop,
                'length': loop_length,
            }
            loop_n += 1

        print("Longest loop is {},length: {}, vars: {}".format(
            longest_loop, longest_length,
            str(loops[longest_loop]['variables'])
            )
        )

        # get positions for all nodes
        pos = nx.get_node_attributes(self.sfd, 'pos')

        # assign random positions to nodes
        for key in pos.keys():
            upper_range = 1
            lower_range = -1
            pos[key] = [random.uniform(lower_range, upper_range),
                        random.uniform(lower_range, upper_range)]
        pp.pprint(pos)

        # iteration parameters
        prior_edge_length = 0.1
        prior_hooke_law_k = 0.1

        def calculate_edge_length(edge):
            length = math.sqrt((pos[edge[0]][0] - pos[edge[1]][0]) ** 2 + (pos[edge[0]][1] - pos[edge[1]][1]) ** 2)
            return length

        def calculate_edge_trend(edge):
            if self.sfd[edge[0]][edge[1]]['length'] > prior_edge_length:  # stretched
                return -1
            elif self.sfd[edge[0]][edge[1]]['length'] == prior_edge_length:
                return 0
            elif self.sfd[edge[0]][edge[1]]['length'] < prior_edge_length:  # compressed
                return 1
            else:
                raise Exception

        def calculate_edge_move(edge):
            move_distance = abs(self.sfd[edge[0]][edge[1]]['length'] - prior_edge_length) * prior_hooke_law_k
            return move_distance

        # iteration
        for i in range(1001):
            subplot1.clear()
            for edge in self.sfd.edges:
                # print(edge)
                self.sfd[edge[0]][edge[1]]['length'] = calculate_edge_length(edge)
                self.sfd[edge[0]][edge[1]]['trend'] = calculate_edge_trend(edge)
                # print(self.sfd[edge[0]][edge[1]]['length'])
                # print(self.sfd[edge[0]][edge[1]]['trend'])

            for node in self.sfd.nodes:
                # print("for Node ", node)
                combined_distance = np.array([0.0, 0.0])
                for edge in self.sfd.edges(node):
                    # print("    for Edge ", edge)

                    direction = np.array(
                        [
                            pos[edge[0]][0] - pos[edge[1]][0],
                            pos[edge[0]][1] - pos[edge[1]][1]
                        ]
                    ) * self.sfd[edge[0]][edge[1]]['trend']

                    distance = direction * calculate_edge_move(edge)
                    # print('   ', distance)

                    combined_distance += distance

                # print("    Combined distance ", combined_distance)
                pos[node] += combined_distance

            # node_pos_adjustment_by_loop_center():
            """
            # the ORIGINAL object here is to decide the rad for one edge.
            # by default rad is set to 0.
            # one edge may appear in multiple loops.
            # therefore we take the cumulative rad.

            # Update:
            # Because networkx and matplotlib do not support specifying specifying rad for each edge,
            # we choose to use cross product to move the node, instead of change the rad
            """
            # TODO: re-write a node-edge based function for drawing CLD.
            for loop_n, loop in loops.items():
                # calculate the arithmetic center for all nodes
                x_sum = 0.0
                y_sum = 0.0
                for node in loop['variables']:
                    x_sum += pos[node][0]
                    y_sum += pos[node][1]
                center = (x_sum / loop['length'], y_sum / loop['length'])
                # print(center)

                # assign rad for all edges
                loop_edges = list(zip(loop['variables'], loop['variables'][1:] + [loop['variables'][0]]))
                for edge in loop_edges:
                    edge_vector = np.array(  # from end to head
                        [
                            pos[edge[1]][0] - pos[edge[0]][0],
                            pos[edge[1]][1] - pos[edge[0]][1]
                        ]
                    )
                    center_vector = np.array(  # from end to center
                        [
                            center[0] - pos[edge[0]][0],
                            center[1] - pos[edge[0]][1]
                        ]
                    )
                    cross_product = np.cross(edge_vector, center_vector)
                    # print('cp', cross_product)

                    head_vector = np.array(  # from head to center
                        [
                            center[0] - pos[edge[1]][0],
                            center[1] - pos[edge[1]][1]
                        ]
                    )

                    if cross_product >= 0:  # move head toward the center
                        pos[edge[1]] += head_vector * 0.1

            if i % 100 == 0:
                print("Iteration step:", i, "/1000")
                if i % 500 == 0:

                    nx.draw_networkx(
                        self.sfd,
                        pos=pos,
                        connectionstyle='arc3, rad=-0.3',
                        # connectionstyle=custom_edge_connectionstyle,
                        node_color='gold',
                        edge_color=custom_edge_colors,
                        arrowsize=20,
                        font_size=9
                    )
                    plt.axis('off')  # turn off axis for structure display
                    plt.show()

    # Draw graphs with customized labels and colored connectors
    def draw_graphs_with_function_value_polarity(self, rtn=False):
        self.figure2 = plt.figure(num='cld')

        plt.clf()
        # generate node labels
        node_attrs_function = nx.get_node_attributes(self.sfd, 'function')
        node_attrs_value = nx.get_node_attributes(self.sfd, 'value')
        custom_node_labels = dict()
        for node, attr in node_attrs_function.items():
            # when element only has a value but no function
            if attr is None:
                attr = node_attrs_value[node][0]
            # when the element has a function
            else:
                attr = self.equation_to_text(attr)
            custom_node_labels[node] = "{}={}".format(node, attr)

        # generate edge polarities
        edge_attrs_polarity = nx.get_edge_attributes(self.sfd, 'polarity')
        custom_edge_colors = list()
        for edge, attr in edge_attrs_polarity.items():
            color = 'k'  # black
            if attr == 'negative':
                color = 'b'  # blue
            custom_edge_colors.append(color)

        # generate node positions
        pos = nx.get_node_attributes(self.sfd, 'pos')

        nx.draw_networkx(
            G=self.sfd,
            labels=custom_node_labels,
            font_size=10,
            node_color='skyblue',
            edge_color=custom_edge_colors,
            pos=pos,
            ax=plt.gca()
        )

        plt.gca().invert_yaxis()
        plt.axis('off')  # turn off axis for structure display

        if rtn:
            return self.figure2
        else:
            plt.show()