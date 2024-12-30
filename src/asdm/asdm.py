import networkx as nx
import numpy as np
import re
from itertools import product
from pprint import pprint
from scipy import stats
from scipy.interpolate import interp1d
from copy import deepcopy
import matplotlib.pyplot as plt

class Node:
    def __init__(self, node_id, operator=None, value=None, operands=None, subscripts=None):
        self.node_id = node_id
        self.operator = operator
        self.value = value
        self.subscripts = subscripts if subscripts is not None else []
        self.operands = operands if operands is not None else []

    def __str__(self):
        if self.operands:
            return f'{self.operator}({", ".join(str(operand) for operand in self.operands)})' 
        else: 
            if self.subscripts:
                return f'{self.operator}({self.value}{self.subscripts})'
            else:
                return f'{self.operator}({self.value})'

class Parser:
    def __init__(self):
        
        self.numbers = {
            'NUMBER': r'(?<![a-zA-Z0-9)])[-]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
        }
        self.special_symbols = {
            'COMMA': r',',
            'LPAREN': r'\(',
            'RPAREN': r'\)',
            'LSPAREN': r'\[',
            'RSPAREN': r'\]',
        }

        self.logic_operators ={
            'NGT': r'\<\=',
            'NLT': r'\>\=',
            'GT': r'\>',
            'LT': r'\<',
            'EQS': r'\=',
            'AND': r'AND',
            'OR': r'OR',
            'NOT': r'NOT',
            'CONIF': r'IF',
            'CONTHEN': r'THEN',
            'CONELSE': r'ELSE',
            }

        self.arithmetic_operators = {
            'PLUS': r'\+',
            'MINUS': r'\-',
            'TIMES': r'\*',
            'FLOORDIVIDE': r'\/\/',
            'DIVIDE': r'\/',
            'MOD': r'MOD(?=\s)', # there are spaces surronding MOD, but the front space is strip()-ed
            'EXP': r'\^',
        }

        self.functions = { # use lookahead (?=\() to ensure only match INIT( not INITIAL
            'MIN': r'MIN(?=\()',
            'MAX': r'MAX(?=\()',
            'SAFEDIV': r'SAFEDIV(?=\()',
            'RBINOM': r'RBINOM(?=\()',
            'INIT': r'INIT(?=\()',
            'DELAY': r'DELAY(?=\()',
            'DELAY1': r'DELAY1(?=\()',
            'DELAY3': r'DELAY3(?=\()',
            'SMTH1': r'SMTH1(?=\()',
            'SMTH3': r'SMTH3(?=\()',
            'STEP': r'STEP(?=\()',
            'HISTORY': r'HISTORY(?=\()',
            'LOOKUP': r'LOOKUP(?=\()',
            'SUM': r'SUM(?=\()',
            'PULSE': r'PULSE(?=\()',
            'INT': r'INT(?=\()',
        }

        self.names = {
            'ABSOLUTENAME': r'"[\s\S]*?"',
            'NAME': r'[a-zA-Z0-9_£$\?]*', # add support for £ and $ in variable names
        }

        self.HEAD = "PARSER"

        self.node_id = 0
        self.tokens = []
        self.current_index = 0

    def tokenise(self, s):
        tokens = []
        while len(s) > 0:
            # print(self.HEAD, 'Tokenising:', s, 'len:', len(s))
            for type_name, type_regex in (
                self.numbers | \
                self.special_symbols | \
                self.logic_operators | \
                self.arithmetic_operators | \
                self.functions | \
                self.names
                ).items():
                m = re.match(pattern=type_regex, string=s)
                if m:
                    token = m[0]
                    if token[0] == "\"" and token[-1] == "\"": # strip quotation marks from matched string
                        token = token[1:-1]
                    if type_name == 'ABSOLUTENAME':
                        type_name = 'NAME'
                    if type_name == 'NUMBER' and token[0] == '-': # fix the -1: negative 1 vs minus 1 problem
                        if not(len(tokens) == 0 or tokens[-1][0] == ['LPAREN', '(']):
                            tokens.append(['MINUS', '-'])
                            tokens.append(['NUMBER', token[1:]])
                            s = s[m.span()[1]:].strip()
                        else:
                            tokens.append([type_name, token])
                            s = s[m.span()[1]:].strip()
                    else:
                        if type_name in self.functions:
                            tokens.append(['FUNC', token])
                        else:
                            tokens.append([type_name, token])
                        s = s[m.span()[1]:].strip()
                    break
        return tokens
    
    def parse(self, expression, plot=False, verbose=False):
        self.verbose = verbose

        if self.verbose:
            print(f"Starting parse of expression: {expression}")
        self.tokens = self.tokenise(expression)
        if verbose:
            print(f"Tokens: {self.tokens}")
        
        ast = self.parse_statement()
        if self.current_index != len(self.tokens):
            raise ValueError("Unexpected end of parsing")
        if self.verbose:
            print("Completed parse")
            print("AST:", ast)
        
        # create ast graph
        ast_graph = nx.DiGraph()
        node_labels = {}

        def add_nodes_and_edges(current_node, parent=None):
            ast_graph.add_node(
                current_node.node_id, 
                operator=current_node.operator, 
                value=current_node.value,
                subscripts=current_node.subscripts,
                operands=[operand.node_id for operand in current_node.operands]
                )
            if parent is not None:
                ast_graph.add_edge(
                    parent.node_id, current_node.node_id
                    )
                

            label_node_id = str(current_node.node_id)
            label_node_op = 'operator:\n' + str(current_node.operator)
            if len(current_node.operands) > 0:
                label_node_operands = 'operands:\n' + str([operand.node_id for operand in current_node.operands])
            elif current_node.subscripts:
                label_node_operands = str(current_node.value)+str(current_node.subscripts)
            elif current_node.value:
                label_node_operands = str(current_node.value)
            else:
                label_node_operands = ''
            node_labels[current_node.node_id] = label_node_id+'\n'+ label_node_op+'\n'+ label_node_operands
            for operand in current_node.operands:
                add_nodes_and_edges(operand, current_node)

        add_nodes_and_edges(ast)

        ast_graph.add_node('root')
        ast_graph.add_edge('root', ast.node_id)
        node_labels['root'] = 'root'

        if self.verbose:
            print("AST_graph", ast_graph.nodes.data(True))

        if plot:
            pos = nx.nx_agraph.graphviz_layout(ast_graph, prog="dot")
            plt.figure(figsize=(12, 8))
            nx.draw(ast_graph, pos, labels=node_labels, with_labels=True, node_size=500, node_color="lightblue", font_size=9, font_weight="bold", arrows=True)
            plt.title("AST Visualization")
            plt.show()
        
        # reset parser state
        self.node_id = 0
        self.tokens = []
        self.current_index = 0

        return ast_graph
    
    def parse_statement(self):
        """Parse a statement. The statement could be an IF-THEN-ELSE statement or an expression."""
        if self.tokens[self.current_index][0] == 'CONIF':
            self.current_index += 1
            condition = self.parse_statement()
            if self.tokens[self.current_index][0] == 'CONTHEN':
                self.current_index += 1
                then_branch = self.parse_statement()
                if self.tokens[self.current_index][0] == 'CONELSE':
                    self.current_index += 1
                    else_branch = self.parse_statement()
                    self.node_id += 1
                    return Node(node_id=self.node_id, operator='CON', operands=[condition, then_branch, else_branch])
                else:
                    raise ValueError(f"Expected ELSE, got {self.tokens[self.current_index]}")
            else:
                raise ValueError(f"Expected THEN, got {self.tokens[self.current_index]}")
        return self.parse_expression()

    def parse_expression(self):
        """Parse an expression."""
        if self.verbose:
            print('parse_expression   ', self.tokens[self.current_index:])
        return self.parse_or_expression()

    def parse_or_expression(self):
        """Parse an or expression."""
        if self.verbose:
            print('parse_or_expr      ', self.tokens[self.current_index:])
        nodes = [self.parse_and_expression()]
        while self.current_index < len(self.tokens) and self.tokens[self.current_index][0] == 'OR':
            op = self.tokens[self.current_index]
            self.current_index += 1
            left = nodes.pop()
            right = self.parse_and_expression()
            self.node_id += 1
            nodes.append(Node(node_id=self.node_id, operator=op[0], operands=[left, right]))
        return nodes[0]

    def parse_and_expression(self):
        """Parse an and expression."""
        if self.verbose:
            print('parse_and_expr     ', self.tokens[self.current_index:])
        nodes = [self.parse_not_expression()]
        while self.current_index < len(self.tokens) and self.tokens[self.current_index][0] == 'AND':
            op = self.tokens[self.current_index]
            self.current_index += 1
            left = nodes.pop()
            right = self.parse_not_expression()
            self.node_id += 1
            nodes.append(Node(node_id=self.node_id, operator=op[0], operands=[left, right]))
        return nodes[0]
    
    def parse_not_expression(self):
        """Parse a NOT expression."""
        if self.verbose:
            print('parse_not_expr     ', self.tokens[self.current_index:])
        if self.tokens[self.current_index][0] == 'NOT':
            self.current_index += 1
            self.node_id += 1
            return Node(node_id=self.node_id, operator='NOT', operands=[self.parse_statement()])
        return self.parse_compare_expression()
    
    def parse_compare_expression(self):
        """Parse a comparison expression."""
        if self.verbose:
            print('parse_compare_expr ', self.tokens[self.current_index:])
        node = self.parse_arith_expression()
        if self.current_index < len(self.tokens) and self.tokens[self.current_index][0] in ['GT', 'LT', 'EQS', 'NGT', 'NLT']:
            op = self.tokens[self.current_index]
            self.current_index += 1
            left = node
            right = self.parse_arith_expression()
            self.node_id += 1
            return Node(node_id=self.node_id, operator=op[0], operands=[left, right])
        return node

    def parse_arith_expression(self):
        """Parse an expression for '+' and '-' with lower precedence."""
        if self.verbose:
            print('parse_arith_expr   ', self.tokens[self.current_index:])
        nodes = [self.parse_mod()]
        while self.current_index < len(self.tokens) and self.tokens[self.current_index][0] in ['PLUS', 'MINUS']:
            op = self.tokens[self.current_index]
            self.current_index += 1
            left = nodes.pop()
            right = self.parse_mod()
            self.node_id += 1
            nodes.append(Node(node_id=self.node_id, operator=op[0], operands=[left, right]))
        return nodes[0]
    
    def parse_mod(self):
        """Parse a mod operation."""
        if self.verbose:
            print('parse_mod          ', self.tokens[self.current_index:])
        nodes = [self.parse_term()]
        while self.current_index < len(self.tokens) and self.tokens[self.current_index][0] == 'MOD':
            op = self.tokens[self.current_index]
            self.current_index += 1
            left = nodes.pop()
            right = self.parse_term()
            self.node_id += 1
            nodes.append(Node(node_id=self.node_id, operator=op[0], operands=[left, right]))
        return nodes[0]

    def parse_term(self):
        """Parse a term for '*' and '/' with higher precedence."""
        if self.verbose:
            print('parse_term         ', self.tokens[self.current_index:])
        nodes = [self.parse_exponent()]
        while self.current_index < len(self.tokens) and self.tokens[self.current_index][0] in ['TIMES', 'DIVIDE', 'FLOORDIVIDE']:
            op = self.tokens[self.current_index]
            self.current_index += 1
            left = nodes.pop()
            right = self.parse_exponent()
            self.node_id += 1
            nodes.append(Node(node_id=self.node_id, operator=op[0], operands=[left, right]))
        return nodes[0]
    
    def parse_exponent(self):
        """Parse an EXP (^) operation."""
        if self.verbose:
            print('parse_exponent     ', self.tokens[self.current_index:])
        nodes = [self.parse_factor()]
        while self.current_index < len(self.tokens) and self.tokens[self.current_index][0] == 'EXP':
            op = self.tokens[self.current_index]
            self.current_index += 1
            left = nodes.pop()
            right = self.parse_factor()
            self.node_id += 1
            nodes.append(Node(node_id=self.node_id, operator=op[0], operands=[left, right]))
        return nodes[0]

    def parse_factor(self):
        """Parse a factor which could be a number, a variable, a function call, or an expression in parentheses."""
        if self.verbose:
            print('parse_factor       ', self.tokens[self.current_index:])
        token = self.tokens[self.current_index]
        if token[0] == 'LPAREN':
            self.current_index += 1
            node = self.parse_statement()
            self.current_index += 1  # Skipping the closing ')'
            return node
        elif token[0] == 'FUNC':
            return self.parse_function_call()
        elif token[0] == 'NAME':
            node = self.parse_variable()
            return node
        elif token[0] == 'NUMBER':
            self.current_index += 1
            self.node_id += 1
            return Node(node_id=self.node_id, operator='IS', value=token[1])
        raise ValueError(f"Unexpected token: {token}")

    def parse_function_call(self):
        """Parse a function call."""
        if self.verbose:
            print('parse_function_call', self.tokens[self.current_index:])
        func_name = self.tokens[self.current_index]
        self.current_index += 2  # Skipping the function name and the opening '('
        args = []
        while self.tokens[self.current_index][0] != 'RPAREN':
            args.append(self.parse_expression())
            if self.tokens[self.current_index][0] == 'COMMA':
                self.current_index += 1  # Skipping the comma
        self.current_index += 1  # Skipping the closing ')'
        self.node_id += 1
        return Node(node_id=self.node_id, operator=func_name[1], operands=args)
    
    def parse_variable(self):
        """Parse a variable. The variable may be subscripted."""
        if self.verbose:
            print('parse_variable     ', self.tokens[self.current_index:])
        var_name = self.tokens[self.current_index][1]
        self.current_index += 1
        if self.current_index < len(self.tokens) and self.tokens[self.current_index][0] == 'LSPAREN':
            subscripts = []
            self.current_index += 1 # Skipping the opening '['
            while self.tokens[self.current_index][0] != 'RSPAREN':
                if self.tokens[self.current_index][0] != 'COMMA':
                    if self.tokens[self.current_index][0] == 'NAME':
                        subscripts.append(self.tokens[self.current_index][1])
                        self.current_index += 1
                    elif self.tokens[self.current_index][0] == 'NUMBER':
                        subscripts.append(str(self.tokens[self.current_index][1]))
                        self.current_index += 1
                    else:
                        raise ValueError(f"Unexpected token for subscript: {self.tokens[self.current_index]}")
                else:
                    self.current_index += 1 # Skipping the comma
            self.current_index += 1 # Skipping the closing ']'
            self.node_id += 1
            return Node(node_id=self.node_id, operator='SPAREN', value=var_name, subscripts=subscripts)
        self.node_id += 1
        return Node(node_id=self.node_id, operator='EQUALS', value=var_name)

class Solver(object):
    def __init__(self, sim_specs=None, dimension_elements=None, name_space=None, graph_functions=None):
        
        self.sim_specs = sim_specs # current_time, initial_time, dt, simulation_time, time_units, running,
        self.dimension_elements = dimension_elements
        self.name_space = name_space
        self.graph_functions = graph_functions

        ### Functions ###

        def integer(a):
            return int(a)

        def logic_and(a, b):
            return (a and b)
        
        def logic_or(a, b):
            return (a or b)
        
        def logic_not(a):
            return (not a)
        
        def greater_than(a, b):
            if a > b:
                return True
            elif a <= b:
                return False
            else:
                raise Exception()

        def less_than(a, b):
            if a < b:
                return True
            elif a >= b:
                return False
            else:
                raise Exception()

        def no_greater_than(a, b):
            if a <= b:
                return True
            elif a > b:
                return False
            else:
                raise Exception()

        def no_less_than(a, b):
            if a >= b:
                return True
            elif a < b:
                return False
            else:
                raise Exception()

        def equals(a, b):
            if a == b:
                return True
            elif a != b:
                return False
            else:
                raise Exception

        def plus(a, b):
            # print('\t'*self.id_level+self.HEAD, 'plus', a, type(a), b, type(b))
            try:
                return a + b
            except TypeError as e:
                if type(a) is dict and type(b) is dict:
                    o = dict()
                    for k in a:
                        o[k] = a[k] + b[k]
                    return o
                else:
                    raise e

        def minus(a, b):
            # print('\t'*self.id_level+self.HEAD, 'minus', a,type(a), b, type(b))
            try:
                return a - b
            except TypeError as e:
                if type(a) is dict and type(b) is dict:
                    o = dict()
                    for k in a:
                        o[k] = a[k] - b[k]
                    return o
                elif type(a) is dict and type(b) in [int, float]:
                    o = dict()
                    for k in a:
                        o[k] = a[k] - b
                    return o
                elif type(a) in [int, float] and type(b) is dict:
                    o = dict()
                    for k in b:
                        o[k] = a - b[k]
                    return o
                else:
                    raise e

        def times(a, b):
            try:
                return a * b
            except TypeError as e:
                if type(a) is dict and type(b) is dict:
                    o = dict()
                    for k in a:
                        o[k] = a[k] * b[k]
                    return o
                elif type(a) is dict and type(b) in [int, float, np.float64]:
                    o = dict()
                    for k in a:
                        o[k] = a[k] * b
                    return o
                elif type(a) in [int, float, np.float64] and type(b) is dict:
                    o = dict()
                    for k in b:
                        o[k] = a * b[k]
                    return o
                else:
                    raise e

        def divide(a, b):
            try:
                return a / b
            except TypeError as e:
                if type(a) is dict and type(b) is dict:
                    # print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a/b', a, b)
                    o = dict()
                    for k in a:
                        o[k] = a[k] / b[k]
                    return o
                elif type(a) is dict and type(b) in [int, float, np.float64]:
                    o = dict()
                    for k in a:
                        o[k] = a[k] / b
                    return o
                elif type(a) in [int, float, np.float64] and type(b) is dict:
                    o = dict()
                    for k in b:
                        o[k] = a / b[k]
                    return o
                else:
                    raise e
        
        def floor_divide(a, b):
            try:
                return a // b
            except TypeError as e:
                if type(a) is dict and type(b) is dict:
                    # print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a//b', a, b)
                    o = dict()
                    for k in a:
                        o[k] = a[k] // b[k]
                    return o
                elif type(a) is dict and type(b) in [int, float, np.float64]:
                    o = dict()
                    for k in a:
                        o[k] = a[k] // b
                    return o
                elif type(a) in [int, float, np.float64] and type(b) is dict:
                    o = dict()
                    for k in b:
                        o[k] = a // b[k]
                    return o
                else:
                    raise e
        
        def safe_div(a, b, c=0):
            if b == 0:
                return c
            else:
                return a / b

        def mod(a, b):
            try:
                return a % b
            except TypeError as e:
                if type(a) is dict and type(b) is dict:
                    # print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a % b', a, b)
                    o = dict()
                    for k in a:
                        o[k] = a[k] % b[k]
                    return o
                elif type(a) is dict and type(b) in [int, float, np.float64]:
                    o = dict()
                    for k in a:
                        o[k] = a[k] % b
                    return o
                elif type(a) in [int, float, np.float64] and type(b) is dict:
                    o = dict()
                    for k in b:
                        o[k] = a % b[k]
                    return o
                else:
                    raise e
                
        def exp(a, b):
            return a ** b

        def con(a, b, c):
            if a:
                return b
            else:
                return c

        def step(stp, time):
            # print('step:', stp, time)
            if sim_specs['current_time'] >= time:
                # print('step out:', stp)
                return stp
            else:
                # print('step out:', 0)
                return 0
            
        def pulse(volume, first_pulse=None, interval=None):
            if first_pulse is None:
                    first_pulse = sim_specs['initial_time']
            if interval is None:
                if sim_specs['current_time'] >= first_pulse: # pulse for all dt after fist pulse
                    return volume / sim_specs['dt']
                else:
                    return 0
            elif interval == 0 or interval > sim_specs['simulation_time']: # only one pulse
                if sim_specs['current_time'] == first_pulse:
                    return volume / sim_specs['dt']
                else:
                    return 0
            else:
                if (sim_specs['current_time'] >= first_pulse) and (sim_specs['current_time'] - first_pulse) % interval == 0: # pulse every interval
                    return volume / sim_specs['dt']
                else:
                    return 0
            
        def rbinom(n, p):
            s = stats.binom.rvs(int(n), p, size=1)[0]
            return float(s) # TODO: something is wrong here - the dimension of s goes high like [[[[30]]]] if not float()ed.
        
        ### Function mapping ###

        self.built_in_functions = {
            'AND':      logic_and,
            'OR':       logic_or,
            'NOT':      logic_not,
            'GT':       greater_than,
            'LT':       less_than,
            'NGT':      no_greater_than,
            'NLT':      no_less_than,
            'EQS':      equals,
            'PLUS':     plus,
            'MINUS':    minus,
            'TIMES':    times,
            'DIVIDE':   divide,
            'FLOORDIVIDE': floor_divide,
            'MIN':      min,
            'MAX':      max,
            'SAFEDIV':  safe_div,
            'CON':      con,
            'STEP':     step,
            'MOD':      mod,
            'RBINOM':   rbinom,
            'PULSE':    pulse,
            'EXP':      exp,
            'INT':      integer,
        }

        self.time_related_functions = [
            'INIT',
            'DELAY',
            'DELAY1',
            'DELAY3',
            'HISTORY',
            'SMTH1',
            'SMTH3',
        ]

        self.array_related_functions = [ # they take variable name as argument, not its value
            'SUM',
        ]

        self.lookup_functions = [
            'LOOKUP'
        ]

        self.custom_functions = {}
        self.time_expr_register = {}
        
        self.id_level = 0

        self.HEAD = "SOLVER"

    def calculate_node(self, parsed_equation, node_id='root', subscript=None, verbose=False, var_name=''):        
        if verbose:
            print('\n'+'\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'processing node {} on subscript {}:'.format(node_id, subscript))

        self.id_level += 1
        
        if type(parsed_equation) is dict:  
            raise Exception('Parsed equation should not be a dict. var:', var_name)

        # if verbose:
        #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'type of parsed_equation: graph')
        if node_id == 'root':
            node_id = list(parsed_equation.successors('root'))[0]
        node = parsed_equation.nodes[node_id]
        if verbose:
            print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'node:', node_id, node)
        node_operator = node['operator']
        node_value = node['value']
        node_subscripts = node['subscripts']
        node_operands = node['operands']
        if node_operator == 'IS':
            # if verbose:
            #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'oprt1')
            #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'o1')
            value = np.float64(node_value)
            if verbose:
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v2 IS:', value)
        elif node_operator == 'EQUALS':
            if verbose:
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'operator v3', node_operator)
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'operator v3', node_value)
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'operator v3', node_subscripts)
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'operands v3', node_operands)
            
            node_var = node_value
            if subscript:
                value = self.name_space[node_var]
                if type(value) is dict:
                    value = value[subscript]
                    if verbose:
                        print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v3.1.1 EQUALS: subscript present, variable subscripted', value)
                else:
                    if verbose:
                        print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v3.1.2 EQUALS: subscript present, variable not subscripted', value)
            else:
                value = self.name_space[node_var]
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v3.2 EQUALS: subscript not present', value)
            
            if verbose:
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v3 EQUALS:', value)
        
        elif node_operator == 'SPAREN': # TODO this part is too dynamic, therefore can be slow.
            var_name = node_value
            if verbose:
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a1', subscript)
            if node_subscripts is None: # only var_name; no subscript is specified
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a1.1')
                # this could be 
                # (1) this variable (var_name) is not subscripted therefore the only value of it should be used;
                # (2) this variable (var_name) is subscripted in the same way as the variable using it (a contextual info is needed and provided in the arg subscript)
                if subscript:
                    if verbose:
                        print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a1.1.1')
                    value = self.name_space[var_name][subscript] 
                else:
                    if verbose:
                        print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a1.1.2')
                    value = self.name_space[var_name]
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v4.1 Sparen without sub:', value)
            else: # there are explicitly specified subscripts in oprands like a[b]
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a1.2', 'subscript from node definition:', node_subscripts[:], 'subscript from context:', subscript)
                # prioritise subscript from node definition
                try:
                    subscript_from_definition = tuple(node_subscripts[:]) # use tuple to make it hashable
                    value = self.name_space[var_name][subscript_from_definition]
                    if verbose:
                        print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a1.2.1')
                except KeyError as e: # subscript in operands looks like a[Dimension_1, Element_1], inference needed
                    if verbose:
                        print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a1.2.2')
                    if subscript: # there's subscript in context
                        subscript_from_definition = node_subscripts[:]
                        subscript_from_operands_with_replacement = list()
                        for i in range(len(subscript_from_definition)):
                            if subscript_from_definition[i] in self.dimension_elements.keys(): # it's sth like Dimension_1
                                # if verbose:
                                #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a1.2.2.1')
                                subscript_from_operands_with_replacement.append(subscript[i]) # take the element from context subscript in the same position to replace Dimension_1
                            else: # it's sth like Element_1
                                # if verbose:
                                #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a1.2.2.2')
                                subscript_from_operands_with_replacement.append(subscript_from_definition[i]) # add to list directly
                        subscript_from_operands_with_replacement = tuple(subscript_from_operands_with_replacement)
                        if verbose:
                            print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a1.2.2', subscript_from_operands_with_replacement)
                        value = self.name_space[var_name][subscript_from_operands_with_replacement] # try if subscript is Element_1
                    else: # there's no subscript in context
                        raise e
                        
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v4.2 Sparen with sub:', value)
        
        # elif operator == 'PAREN':
            # value = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[0][2], subscript=subscript, verbose=verbose)
            # if verbose:
                # print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v6 Paren:', value)

        elif node_operator in self.built_in_functions.keys(): # plus, minus, con, etc.
            if verbose:
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'operator v7 Built-in operator:', node_operator, node_operands)
            func_name = node_operator
            function = self.built_in_functions[func_name]
            oprds = []
            for operand in node_operands:
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v7.1', 'operand', operand)
                v = self.calculate_node(parsed_equation=parsed_equation, node_id=operand, subscript=subscript, verbose=verbose)
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v7.2', 'value', v, subscript)
                oprds.append(v)
            if verbose:
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v7.3', 'operands', oprds)
            value = function(*oprds)
            if verbose:
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v7 Built-in operation:', value)
        
        elif node_operator in self.custom_functions.keys(): # graph functions
            if verbose:
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'custom func operator', node_operator)
            func_name = node_operator
            function = self.custom_functions[func_name]
            oprds = []
            for operand in node_operands:
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'operand', operand)
                v = self.calculate_node(parsed_equation=parsed_equation, node_id=operand, subscript=subscript, verbose=verbose)
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'value', v)
                oprds.append(v)
            if verbose:
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'operands', oprds)
            value = function(*oprds)
            if verbose:
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v8 GraphFunc:', value)

        elif node_operator in self.time_related_functions: # init, delay, etc
            if verbose:
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'time-related func. operator:', node_operator, 'operands:', node_operands)
            func_name = node_operator
            if func_name == 'INIT':
                if tuple([parsed_equation, node_id, node_operands[0]]) in self.time_expr_register.keys():
                    value = self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])]
                else:
                    value = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[0], subscript=subscript, verbose=verbose)
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])] = value
            elif func_name == 'DELAY':
                # expr value
                expr_value = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[0], subscript=subscript, verbose=verbose)
                if tuple([parsed_equation, node_id, node_operands[0]]) in self.time_expr_register.keys():
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])].append(expr_value)
                else:
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])] = [expr_value]
                
                # init value
                if len(node_operands) == 2: # there's no initial value specified -> use the delayed expr's initial value
                    init_value = self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][0]
                elif len(node_operands) == 3: # there's an initial value specified
                    init_value = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[2], subscript=subscript, verbose=verbose)
                else:
                    raise Exception("Invalid initial value for DELAY in operands {}".format(node_operands))

                # delay time
                delay_time = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[1], subscript=subscript, verbose=verbose)
                if delay_time > (self.sim_specs['current_time'] - self.sim_specs['initial_time']): # (- initial_time) because simulation might not start from time 0
                    value = init_value
                else:
                    delay_steps = delay_time / self.sim_specs['dt']
                    value = self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][-int(delay_steps+1)]
            elif func_name == 'DELAY1':
                # args values
                order = 1
                expr_value = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[0], subscript=subscript, verbose=verbose)
                delay_time = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[1], subscript=subscript, verbose=verbose)

                if len(node_operands) == 3:
                    init_value = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[2], subscript=subscript, verbose=verbose)
                elif len(node_operands) == 2:
                    init_value = expr_value
                else:
                    raise Exception('Invalid number of args for DELAY1.')
                
                # register
                if tuple([parsed_equation, node_id, node_operands[0]]) not in self.time_expr_register.keys():
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])] = list()
                    for i in range(order):
                        self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])].append(delay_time/order*init_value)
                # outflows
                outflows = list()
                for i in range(order):
                    outflows.append(self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][i]/(delay_time/order) * self.sim_specs['dt'])
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][i] -= outflows[i]
                # inflows
                self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][0] += expr_value * self.sim_specs['dt']
                for i in range(1, order):
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][i] += outflows[i-1]

                return outflows[-1] / self.sim_specs['dt']

            elif func_name == 'DELAY3':
                # arg values
                order = 3
                expr_value = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[0], subscript=subscript, verbose=verbose)
                delay_time = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[1], subscript=subscript, verbose=verbose)
                if len(node_operands) == 3:
                    init_value = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[2], subscript=subscript, verbose=verbose)
                elif len(node_operands) == 2:
                    init_value = expr_value
                else:
                    raise Exception('Invalid number of args for SMTH3.')
                
                # register
                if tuple([parsed_equation, node_id, node_operands[0]]) not in self.time_expr_register.keys():
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])] = list()
                    for i in range(order):
                        self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])].append(delay_time/order*init_value)
                # outflows
                outflows = list()
                for i in range(order):
                    outflows.append(self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][i]/(delay_time/order) * self.sim_specs['dt'])
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][i] -= outflows[i]
                # inflows
                self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][0] += expr_value * self.sim_specs['dt']
                for i in range(1, order):
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][i] += outflows[i-1]

                return outflows[-1] / self.sim_specs['dt']

            elif func_name == 'HISTORY':
                # expr value
                expr_value = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[0], subscript=subscript, verbose=verbose)
                if tuple([parsed_equation, node_id, node_operands[0]]) in self.time_expr_register.keys():
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])].append(expr_value)
                else:
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])] = [expr_value]
                
                # historical time
                historical_time = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[1], subscript=subscript, verbose=verbose)
                if historical_time > self.sim_specs['current_time'] or historical_time < self.sim_specs['initial_time']:
                    value = 0
                else:
                    historical_steps = (historical_time - self.sim_specs['initial_time']) / self.sim_specs['dt']
                    value = self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][int(historical_steps)]
            
            elif func_name == 'SMTH1':
                # arg values
                order = 1
                expr_value = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[0], subscript=subscript, verbose=verbose)
                if type(expr_value) is dict:
                    if subscript is not None:
                        expr_value = expr_value[subscript]
                    else:
                        raise Exception('Invalid subscript.')
                smth_time = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[1], subscript=subscript, verbose=verbose)
                if len(node_operands) == 3:
                    init_value = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[2], subscript=subscript, verbose=verbose)
                elif len(node_operands) == 2:
                    init_value = expr_value
                else:
                    raise Exception('Invalid number of args for SMTH1.')
                
                # register
                if tuple([parsed_equation, node_id, node_operands[0]]) not in self.time_expr_register.keys():
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])] = list()
                    for i in range(order):
                        self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])].append(smth_time/order*init_value)
                # outflows
                outflows = list()
                for i in range(order):
                    outflows.append(self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][i]/(smth_time/order) * self.sim_specs['dt'])
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][i] -= outflows[i]
                # inflows
                self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][0] += expr_value * self.sim_specs['dt']
                for i in range(1, order):
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][i] += outflows[i-1]

                return outflows[-1] / self.sim_specs['dt']

            elif func_name == 'SMTH3':
                # arg values
                order = 3
                expr_value = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[0], subscript=subscript, verbose=verbose)
                if type(expr_value) is dict:
                    if subscript is not None:
                        expr_value = expr_value[subscript]
                    else:
                        raise Exception('Invalid subscript.')
                smth_time = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[1], subscript=subscript, verbose=verbose)
                if len(node_operands) == 3:
                    init_value = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[2], subscript=subscript, verbose=verbose)
                elif len(node_operands) == 2:
                    init_value = expr_value
                else:
                    raise Exception('Invalid number of args for SMTH3.')
                
                # register
                if tuple([parsed_equation, node_id, node_operands[0]]) not in self.time_expr_register.keys():
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])] = list()
                    for i in range(order):
                        self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])].append(smth_time/order*init_value)
                # outflows
                outflows = list()
                for i in range(order):
                    outflows.append(self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][i]/(smth_time/order) * self.sim_specs['dt'])
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][i] -= outflows[i]
                # inflows
                self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][0] += expr_value * self.sim_specs['dt']
                for i in range(1, order):
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][i] += outflows[i-1]

                return outflows[-1] / self.sim_specs['dt']

            else:
                raise Exception('Unknown time-related operator {}'.format(node_operator))
            if verbose:
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v9 Time-related Func:', value)
        
        elif node_operator in self.array_related_functions: # Array-RELATED
            if verbose:
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'array-related func. operator:', node_operator, 'operands:', node_operands)
            func_name = node_operator
            if func_name == 'SUM':
                arrayed_var_name = parsed_equation.nodes[node_operands[0]]['value']
                sum_array = 0
                for _, sub_val in self.name_space[arrayed_var_name].items():
                    sum_array += sub_val
                value = sum_array
            if verbose:
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v10 Array-related Func:', value)
        
        elif node_operator in self.lookup_functions: # LOOKUP
            if verbose:
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'Lookup func. operator:', node_operator, 'operands:', node_operands)
            func_name = node_operator
            if func_name == 'LOOKUP':
                look_up_func_node_id = node_operands[0]
                look_up_func_name = parsed_equation.nodes[look_up_func_node_id]['value']
                look_up_func = self.graph_functions[look_up_func_name]
                input_value = self.calculate_node(parsed_equation=parsed_equation, node_id=node_operands[1], subscript=subscript, verbose=verbose)
                value = look_up_func(input_value)
            else:
                raise Exception('Unknown Lookup function {}'.format(node_operator))
        
        else:
            raise Exception('Unknown operator {}'.format(node_operator))
        
        self.id_level -= 1
        
        if verbose:
            print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'value for node {} on subscript {}'.format(node_id, subscript), value)

        return value


class GraphFunc(object):
    def __init__(self, out_of_bound_type, yscale, ypts, xscale=None, xpts=None):
        self.out_of_bound_type = out_of_bound_type
        self.yscale = yscale
        self.xscale = xscale
        self.xpts = xpts
        self.ypts = ypts
        self.eqn = None
        self.initialise()
    
    def initialise(self):
        if self.xpts is None:
            self.xpts = np.linspace(self.xscale[0], self.xscale[1], num=len(self.ypts))
        self.interp_func = interp1d(self.xpts, self.ypts, kind='linear')
        self.interp_func_above = interp1d(self.xpts[-2:], self.ypts[-2:], kind='linear', fill_value='extrapolate')
        self.interp_func_below = interp1d(self.xpts[:2], self.ypts[:2], kind='linear', fill_value='extrapolate')

    def __call__(self, input):
        # input out of xscale treatment:
        if self.out_of_bound_type is None: # default to continuous
            input = max(input, self.xpts[0])
            input = min(input, self.xpts[-1])
            output = float(self.interp_func(input)) # the output (like array([1.])) needs to be converted to float to avoid dimension explosion
            return output
        elif self.out_of_bound_type == 'extrapolate':
            if input < self.xpts[0]:
                output = float(self.interp_func_below(input))
            elif input > self.xpts[-1]:
                output = float(self.interp_func_above(input))
            else:
                output = float(self.interp_func(input))
            return output
        elif self.out_of_bound_type == 'discrete':
            if input < self.xpts[0]:
                return self.ypts[0]
            elif input > self.xpts[-1]:
                return self.ypts[-1]
            else:
                for i, xpt in enumerate(self.xpts):
                    if input < xpt:
                        return self.ypts[i-1]
                return self.ypts[-1]
        else:
            raise Exception('Unknown out_of_bound_type {}'.format(self.out_of_bound_type))
    
    def overwrite_xpts(self, xpts):
        # if len(self.xpts) != len(xpts):
            # print("Warning: new set of x points have a different length to the old set.")
        self.xpts = xpts
        
    def overwrite_xscale(self, xscale):
        self.xscale = xscale
        self.xpts = None # to auto-infer self.xpts from self.xscale, self.xpts must set to None

    def overwrite_ypts(self, ypts):
        # if len(self.ypts) != len(ypts):
            # print("Warning: new set of y points have a different length to the old set.")
        self.ypts = ypts


class Conveyor(object):
    def __init__(self, length, eqn):
        self.length_time_units = length
        self.equation = eqn
        self.length_steps = None # to be decided at runtime
        self.total = 0 # to be decided when initialising stocks
        self.slats = list() # pipe [new, ..., old]
        self.is_initialized = False
        self.leaks = list()
        self.leak_fraction = 0

    def initialize(self, length, value, leak_fraction=None):
        self.total = value
        self.length_steps = length
        if leak_fraction is None or leak_fraction == 0:
            for _ in range(self.length_steps):
                self.slats.append(self.total/self.length_steps)
                self.leaks.append(0)
        else:
            self.leak_fraction = leak_fraction
            n_leak = 0
            for i in range(self.length_steps):
                n_leak += i+1
            # print('Conveyor N total leaks:', n_leak)
            self.output = self.total / (self.length_steps + (n_leak * self.leak_fraction) / ((1-self.leak_fraction)*self.length_steps))
            # print('Conveyor Output:', output)
            leak = self.output * (self.leak_fraction/((1-self.leak_fraction)*self.length_steps))
            # print('Conveyor Leak:', leak)
            # generate slats
            for i in range(self.length_steps):
                self.slats.append(self.output + (i+1)*leak)
                self.leaks.append(leak)
            self.slats.reverse()
        # print('Conveyor Initialised:', self.conveyor, '\n')
        self.is_initialized = True

    def level(self):
        return self.total

    # order of execution:
    # 1 Leak from every slat
    #   to do this we need to know the leak for every slat
    # 2 Pop the last slat
    # 3 Input as the first slat

    def leak_linear(self):
        for i in range(self.length_steps):
            self.slats[i] = self.slats[i] - self.leaks[i]
        
        total_leaked = sum(self.leaks)
        self.total -= total_leaked
        return total_leaked
    
    def outflow(self):
        output = self.slats.pop()
        self.total -= output
        self.leaks.pop()
        return output

    def inflow(self, value):
        self.total += value
        self.slats = [value] + self.slats
        self.leaks = [value* self.leak_fraction/self.length_steps]+self.leaks


class Stock(object):
    def __init__(self):
        self.initialised = False


class DataFeeder(object):
    def __init__(self, data, from_time=0, data_dt=1, interpolate=False):
        """
        data: a list

        """
        self.interpolate = interpolate
        self.data_dt = data_dt
        self.from_time =from_time
        self.time_data = dict()
        time = self.from_time
        for d in data:
            self.time_data[time] = d
            time += self.data_dt
        # print(self.time_data)
        self.last_success_time = None

    def __call__(self, current_time): # make a datafeeder callable
        try:
            d = self.time_data[current_time]
            self.last_success_time = current_time
        except KeyError:
            if current_time < self.from_time:
                raise Exception("Current time < external data starting time.")
            elif current_time > list(self.time_data.keys())[-1]:
                raise Exception("Current time > external data ending time.")
            else:
                if self.interpolate:
                    d_0 = self.time_data[self.last_success_time]
                    d_1 = self.time_data[self.last_success_time + self.data_dt]
                    interp_func_2pts = interp1d(
                        [self.last_success_time, self.last_success_time + self.data_dt],
                        [d_0, d_1]
                        )
                    d = interp_func_2pts(current_time)
                else:
                    d = self.time_data[self.last_success_time]
        return(np.float64(d))


class sdmodel(object):
    # equations
    def __init__(self, from_xmile=None):
        # Debug
        self.HEAD = 'ENGINE'
        self.debug_level = 0

        # sim_specs
        self.sim_specs = {
            'initial_time': 0,
            'current_time': 0,
            'dt': 0.25,
            'simulation_time': 13,
            'time_units' :'Weeks',
            'running': False,
        }

        # dimensions
        self.var_dimensions = dict()
        self.dimension_elements = dict()
        
        # stocks
        self.stocks = dict()
        self.stock_equations = dict()
        self.stock_equations_parsed = dict()
        self.stock_non_negative = dict()

        # discrete variables
        self.conveyors = dict()
        
        # flow
        self.flow_positivity = dict()
        self.flow_equations = dict()
        self.flow_equations_parsed = dict()

        # connections
        self.stock_flows = dict()
        self.flow_stocks = dict()
        self.leak_conveyors = dict()
        self.outflow_conveyors = dict()
        
        # aux
        self.aux_equations = dict()
        self.aux_equations_parsed = dict()

        # graph_functions
        self.graph_functions = dict()
        self.graph_functions_renamed = dict()

        # variable_values
        self.name_space = dict()
        self.stock_shadow_values = dict() # temporary device to store in/out flows' effect on stocks.
        self.time_slice = dict()
        self.full_result = dict()
        self.full_result_flattened = dict()

        # env variables
        self.env_variables = {
            'TIME': 0,
            'DT': 0.25
        }

        # parser
        self.parser = Parser()

        # solver
        self.solver = Solver(
            sim_specs=self.sim_specs,
            dimension_elements=self.dimension_elements,
            name_space=self.name_space,
            graph_functions=self.graph_functions,
            )

        # custom functions
        self.custom_functions = {}

        # If the model is based on an XMILE file
        if from_xmile is not None:
            # print(self.HEAD, 'Reading XMILE model from {}'.format(from_xmile))
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
                self.env_variables['DT'] =sim_dt
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
                    gf = var.find('gf')
                    out_of_bound_type = gf.get('type')
                    if gf.find('xscale'):
                        xscale = [
                            float(gf.find('xscale').get('min')),
                            float(gf.find('xscale').get('max'))
                        ]
                    else:
                        xscale = None
                    
                    if gf.find('xpts'):
                        xpts = [float(t) for t in gf.find('xpts').text.split(',')]
                    else:
                        xpts = None
                    
                    if xscale is None and xpts is None:
                        raise Exception("GraphFunc: xscale and xpts cannot both be None.")

                    yscale = [
                        float(gf.find('yscale').get('min')),
                        float(gf.find('yscale').get('max'))
                    ]
                    ypts = [float(t) for t in gf.find('ypts').text.split(',')]

                    equation = GraphFunc(out_of_bound_type=out_of_bound_type, yscale=yscale, ypts=ypts, xscale=xscale, xpts=xpts)
                    return equation

                # create var subscripted equation
                def subscripted_equation(var):
                    if var.find('dimensions'):
                        self.var_dimensions[self.name_handler(var.get('name'))] = list()
                        var_dimensions = var.find('dimensions').findAll('dim')
                        # print('Found dimensions {}:'.format(var), var_dimensions)

                        var_dims = dict()
                        for dimension in var_dimensions:
                            dim_name = dimension.get('name')
                            self.var_dimensions[self.name_handler(var.get('name'))].append(dim_name)
                            var_dims[dim_name] = dims[dim_name]
                        
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
                                raise Exception('No meaningful definition found for variable {}'.format(self.name_handler(var.get('name'))))
                            
                            # fetch lists of elements and generate elements trings
                            element_combinations = product(*list(var_dims.values()))

                            for ect in element_combinations:
                                var_subscripted_eqn[ect] =equation
                        return(var_subscripted_eqn)
                    else:
                        self.var_dimensions[self.name_handler(var.get('name'))] = None
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
                    name = self.name_handler(stock.get('name'))
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
                        name, 
                        equation=subscripted_equation(stock), 
                        non_negative=non_negative,
                        is_conveyor=is_conveyor,
                        in_flows=[f.text for f in inflows],
                        out_flows=[f.text for f in outflows],
                        )
                    
                # create auxiliaries
                for auxiliary in auxiliaries:
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

        self.name_space.update(self.env_variables)

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
        if type(equation) in [int, float, np.int_, np.float64]:
            equation = str(equation)
        self.stock_equations[name] = equation
        self.stock_non_negative[name] = non_negative
        connections = dict()
        if len(in_flows) != 0:
            connections['in'] = in_flows
        if len(out_flows) != 0:
            connections['out'] = out_flows
        self.stock_flows[name] = connections

        for in_flow in in_flows:
            if in_flow not in self.flow_stocks:
                self.flow_stocks[in_flow] = dict()
            self.flow_stocks[in_flow]['to'] = name
        for out_flow in out_flows:
            if out_flow not in self.flow_stocks:
                self.flow_stocks[out_flow] = dict()
            self.flow_stocks[out_flow]['from'] = name
        
        self.stocks[name] = Stock()
    
    def add_flow(self, name, equation, leak=None, non_negative=False):
        if type(equation) in [int, float, np.int_, np.float64]:
            equation = str(equation)
        self.flow_positivity[name] = non_negative
        if leak:
            self.leak_conveyors[name] = None # to be filled when parsing the conveyor
        self.flow_equations[name] = equation
    
    def add_aux(self, name, equation):
        if type(equation) in [int, float, np.int_, np.float64]:
            equation = str(equation)
        self.aux_equations[name] = equation

    def format_new_equation(self, new_equation):
        if type(new_equation) is str:
            pass
        elif type(new_equation) in [int, float, np.int_, np.float64]:
            new_equation = str(new_equation)
        elif type(new_equation) is DataFeeder:
            pass
        elif type(new_equation) is dict:
            pass
        else:
            raise Exception('Unsupported new equation {} type {}'.format(new_equation, type(new_equation)))
        return new_equation

    def replace_element_equation(self, name, new_equation):
        new_equation = self.format_new_equation(new_equation)
        
        if name in self.stock_equations:
            if type(new_equation) is dict:
                if type(self.stock_equations[name]) is not dict: # if the old equation is not subscripted
                    self.stock_equations[name] = new_equation # replace the whole equation
                    for k_new, v_new in self.stock_equations[name].items():
                        self.stock_equations[name][k_new] = self.format_new_equation(v_new)
                else:
                    for k_new, v_new in new_equation.items():
                        if k_new in self.stock_equations[name]:
                            self.stock_equations[name][k_new] = self.format_new_equation(v_new)
            else:
                self.stock_equations[name] = new_equation
        elif name in self.flow_equations:
            if type(new_equation) is dict:
                if type(self.flow_equations[name]) is not dict: # if the old equation is not subscripted
                    self.flow_equations[name] = new_equation # replace the whole equation
                    for k_new, v_new in self.flow_equations[name].items():
                        self.flow_equations[name][k_new] = self.format_new_equation(v_new)
                else:
                    for k_new, v_new in new_equation.items():
                        if k_new in self.flow_equations[name]:
                            self.flow_equations[name][k_new] = self.format_new_equation(v_new)
            else:
                self.flow_equations[name] = new_equation
        elif name in self.aux_equations:
            if type(new_equation) is dict:
                if type(self.aux_equations[name]) is not dict: # if the old equation is not subscripted
                    self.aux_equations[name] = new_equation # replace the whole equation
                    for k_new, v_new in self.aux_equations[name].items():
                        self.aux_equations[name][k_new] = self.format_new_equation(v_new)
                else:
                    for k_new, v_new in new_equation.items():
                        if k_new in self.aux_equations[name]:
                            self.aux_equations[name][k_new] = self.format_new_equation(v_new)
            else:
                self.aux_equations[name] = new_equation
        else:
            raise Exception('Unable to find {} in the current model'.format(name))
    
    # def overwrite_stock_value(self, name, new_value):
    #     print(self.name_space)
    #     if name not in self.name_space:
    #         if name in self.stock_equations:
    #             raise Exception("Stock {} exists, but able to find in the current model's namespace. Not simulated?".format(name))
    #         else:
    #             raise Exception('Unable to find Stock {} in the current model.'.format(name))
        
    #     if type(self.name_space[name]) is dict and type(new_value) in [int, float, np.int_, np.float64]:
    #         raise Exception('Unable to overwrite arrayed stock value of {} with {} of type {}'.format(name, new_value, type(new_value)))
        
    #     self.name_space[name] = new_value
        
    def overwrite_graph_function_points(self, name, new_xpts=None, new_xscale=None, new_ypts=None):
        if new_xpts is None and new_xscale is None and new_ypts is None:
            raise Exception("Inputs cannot all be None.")

        if name in self.stock_equations:
            graph_func_equation = self.stock_equations[name]
        elif name in self.flow_equations:
            graph_func_equation = self.flow_equations[name]
        elif name in self.aux_equations:
            graph_func_equation = self.aux_equations[name]
        else:
            raise Exception('Unable to find {} in the current model'.format(name))
        
        if new_xpts is not None:
            # print('Old xpts:', graph_func_equation.xpts)
            graph_func_equation.overwrite_xpts(new_xpts)
            # print('New xpts:', graph_func_equation.xpts)
        
        if new_xscale is not None:
            # print('Old xscale:', graph_func_equation.xscale)
            graph_func_equation.overwrite_xscale(new_xscale)
            # print('New xscale:', graph_func_equation.xscale)
        
        if new_ypts is not None:
            # print('Old ypts:', graph_func_equation.ypts)
            graph_func_equation.overwrite_ypts(new_ypts)
            # print('New ypts:', graph_func_equation.ypts)
        
        graph_func_equation.initialise()

    def parse_equation(self, var, equation):
        if type(equation) is GraphFunc:
            gfunc_name = 'GFUNC{}'.format(len(self.graph_functions_renamed))
            self.graph_functions_renamed[gfunc_name] = equation # just for length ... for now
            self.graph_functions[var] = equation
            self.parser.functions.update({gfunc_name:gfunc_name+r"(?=\()"}) # make name var also a function name and add it to the parser
            self.solver.custom_functions.update({gfunc_name:equation})
            equation = gfunc_name+'('+ equation.eqn + ')'  # make equation into form like var(eqn), 
                                            # where eqn is the euqaiton whose outcome is the input to GraphFunc var()
                                            # this is also how Vensim handles GraphFunc
            parsed_equation = self.parser.parse(equation)
            return parsed_equation
        
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

            return [ # using list to store [len_eqn, val_eqn]. Don't use {'len':xxx, 'val':xxx} to avoid confusion with subscripted equation.
                parsed_equation_len, 
                parsed_equation_val
                ]

        elif type(equation) is DataFeeder:
            return equation

        elif type(equation) in [str, int, float, np.int_, np.float64]:
            parsed_equation = self.parser.parse(equation)
            return parsed_equation

        else:
            raise Exception('Unsupported equation {} type {}'.format(equation, type(equation)))
    
    def batch_parse(self, equations, parsed_equations, verbose=False):
        for var, equation in equations.items():
            if verbose:
                print(self.HEAD, "Parsing:", var)
                print(self.HEAD, "    Eqn:", equation)
            
            if type(equation) is dict:
                parsed_equations[var] = dict()
                for k, ks in equation.items():
                    parsed_equations[var][k] = self.parse_equation(var=var, equation=ks)
            
            else:
                parsed_equations[var] = self.parse_equation(var=var, equation=equation)
            

    def parse(self, verbose=False):
        # string equation -> calculation tree

        self.batch_parse(self.stock_equations, self.stock_equations_parsed, verbose=verbose)
        self.batch_parse(self.flow_equations, self.flow_equations_parsed, verbose=verbose)
        self.batch_parse(self.aux_equations, self.aux_equations_parsed, verbose=verbose)
        
    def is_dependent(self, var1, var2):
        # determine if var2 depends on var1, i.e., var1 --> var2 or var1 appears in var2's equation
        
        def is_dependent_sub(var1, parsed_equation_var2, dependent=False):
            leafs = [x for x in parsed_equation_var2.nodes() if parsed_equation_var2.out_degree(x)==0]
            for leaf in leafs:
                if parsed_equation_var2.nodes[leaf]['operator'][0] in ['EQUALS', 'SPAREN']:
                    operands = parsed_equation_var2.nodes[leaf]['operands']
                    if operands[0][0] == 'NUMBER': # if 'NUMBER' then pass, as numbers (e.g. 100) do not have a node
                        pass
                    elif operands[0][0] == 'NAME': # this refers to a variable like 'a'
                        var_dependent = operands[0][1]
                        if var_dependent == var1:
                            dependent = True
                            break
                    elif operands[0][0] == 'FUNC': # this refers to a subscripted variable like 'a[ele1]'
                        # need to find that 'SPAREN' node
                        var_dependent_node_id = operands[0][2]
                        var_dependent = parsed_equation_var2.nodes[var_dependent_node_id]['operands'][0][1]
                        if var_dependent == var1:
                            dependent = True
                            break
            return dependent
        
        parsed_equation_var2 = (self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)[var2]
        if type(parsed_equation_var2) is dict:
            for _, sub_eqn in parsed_equation_var2.items():
                dependent = is_dependent_sub(var1, sub_eqn)
                if dependent:
                    return True
            return False
        else:
            return is_dependent_sub(var1, parsed_equation_var2)

    def calculate_dependents(self, parsed_equation, verbose=False):
        leafs = [x for x in parsed_equation.nodes() if parsed_equation.out_degree(x)==0]
        for leaf in leafs:
            # print('i4.1')
            if parsed_equation.nodes[leaf]['operator'] in ['EQUALS', 'SPAREN']:
                # operands = parsed_equation.nodes[leaf]['operands']
                var_dependent = parsed_equation.nodes[leaf]['value']
                self.calculate_variable(var=var_dependent, verbose=verbose)
    
    def calculate_variable(self, var, subscript=None, verbose=False, leak_frac=False, conveyor_init=False, conveyor_len=False):
        # print("\nEngine Calculating: {:<15} on subscript {}".format(var, subscript), '\n', 'name_space:', self.name_space, '\n', 'flow_effects', self.stock_shadow_values)
        # print("Engine Calculating: {:<15} on subscript {}".format(var, subscript))
        # debug
        if var in self.env_variables.keys():
            return
        
        if subscript is not None:
            parsed_equation = (self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)[var][subscript]
        else:
            parsed_equation = (self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)[var]
        
        # DataFeeder - external data
        if type(parsed_equation) is DataFeeder:
            if var not in self.name_space:
                self.name_space[var] = parsed_equation(self.sim_specs['current_time'])

        # A: var is a Conveyor
        if var in self.conveyors:
            # print('Calculating Conveyor {}'.format(var))
            if not (conveyor_init or conveyor_len):
                if not self.conveyors[var]['conveyor'].is_initialized:
                    # print('Initialising {}'.format(var))
                    # when initialising, equation of the conveyor needs to be evaluated, setting flag conveyor_len to True 
                    self.calculate_variable(var=var, subscript=subscript, verbose=verbose, conveyor_len=True)
                    conveyor_length = self.conveyors[var]['len']
                    length_steps = int(conveyor_length/self.sim_specs['dt'])
                    
                    # when initialising, equation of the conveyor needs to be evaluated, setting flag conveyor_init to True 
                    self.calculate_variable(var=var, subscript=subscript, verbose=verbose, conveyor_init=True)
                    conveyor_init_value = self.conveyors[var]['val']
                    
                    leak_flows = self.conveyors[var]['leakflow']
                    if len(leak_flows) == 0:
                        leak_fraction = 0
                    else:
                        for leak_flow in leak_flows.keys():
                            self.calculate_variable(var=leak_flow, subscript=subscript, verbose=verbose, leak_frac=True)
                            leak_fraction = self.conveyors[var]['leakflow'][leak_flow] # TODO multiple leakflows
                    self.conveyors[var]['conveyor'].initialize(length_steps, conveyor_init_value, leak_fraction)
                    
                    # put initialised conveyor value to name_space
                    value = self.conveyors[var]['conveyor'].level()
                    self.name_space[var] = value
                
                if var not in self.stock_shadow_values:
                    # print("Updatting {} and its outflows".format(var))
                    # print("    Name space1:", self.name_space)
                    # leak
                    for leak_flow, leak_fraction in self.conveyors[var]['leakflow'].items():
                        if leak_flow not in self.name_space: 
                            # print('    Calculating leakflow {} for {}'.format(leak_flow, var))
                            leaked_value = self.conveyors[var]['conveyor'].leak_linear()
                            self.name_space[leak_flow] = leaked_value / self.sim_specs['dt'] # TODO: we should also consider when leak flows are subscripted
                    # out
                    for outputflow in self.conveyors[var]['outputflow']:
                        if outputflow not in self.name_space:
                            # print('    Calculating outflow {} for {}'.format(outputflow, var))
                            outflow_value = self.conveyors[var]['conveyor'].outflow()
                            self.name_space[outputflow] = outflow_value / self.sim_specs['dt']
                    # print("    Name space2:", self.name_space)
                    self.stock_shadow_values[var] = self.conveyors[var]['conveyor'].level()

            elif conveyor_len:
                # print('Calculating LEN for {}'.format(var))
                # it is the intitial value of the conveyoer
                parsed_equation = self.stock_equations_parsed[var][0]
                self.calculate_dependents(parsed_equation=parsed_equation, verbose=verbose)
                self.conveyors[var]['len'] = self.solver.calculate_node(parsed_equation=parsed_equation, verbose=verbose, var_name=var)
            
            elif conveyor_init:
                # print('Calculating INIT VAL for {}'.format(var))
                # it is the intitial value of the conveyoer
                parsed_equation = self.stock_equations_parsed[var][1]
                self.calculate_dependents(parsed_equation=parsed_equation, verbose=verbose)
                self.conveyors[var]['val'] = self.solver.calculate_node(parsed_equation=parsed_equation, verbose=verbose, var_name=var)
        
        # B: var is a normal stock
        elif var not in self.conveyors and var in self.stocks:
            if not self.stocks[var].initialised:
                # print('Stock {} not initialised'.format(var))
                if type(parsed_equation) is dict:
                    for sub, sub_parsed_equation in parsed_equation.items():
                        self.calculate_dependents(parsed_equation=sub_parsed_equation, verbose=verbose)
                        value = self.solver.calculate_node(parsed_equation=sub_parsed_equation, subscript=sub, verbose=verbose, var_name=var)
                        if var not in self.name_space:
                            self.name_space[var] = dict()
                        self.name_space[var][sub] = value
                elif var in self.var_dimensions and self.var_dimensions[var] is not None: # The variable is subscripted but all elements uses the same equation
                    self.calculate_dependents(parsed_equation=parsed_equation, verbose=verbose)
                    for sub in self.dimension_elements[self.var_dimensions[var]]:
                        value = self.solver.calculate_node(parsed_equation=parsed_equation, subscript=sub, verbose=verbose, var_name=var)
                        if var not in self.name_space:
                            self.name_space[var] = dict()
                        self.name_space[var][sub] = value
                else: # The variable is not subscripted
                    self.calculate_dependents(parsed_equation=parsed_equation, verbose=verbose)
                    value = self.solver.calculate_node(parsed_equation=parsed_equation, subscript=subscript, verbose=verbose, var_name=var)
                    self.name_space[var] = value
                self.stocks[var].initialised = True
                
            # if the stock's shadow value has not been calculated for this dt:
            if var not in self.stock_shadow_values:
                # load stock's value from last dt from name_space
                self.stock_shadow_values[var] = deepcopy(self.name_space[var])
                if var in self.stock_flows: # some stocks are not connected to any flow
                    if 'in' in self.stock_flows[var]:
                        in_flows = self.stock_flows[var]['in']
                        for in_flow in in_flows:
                            if in_flow not in self.name_space:
                                self.calculate_variable(var=in_flow, subscript=subscript, verbose=verbose)
                            if var in self.stock_non_negative:
                                if type(self.name_space[in_flow]) is not dict:
                                    if self.stock_shadow_values[var] + self.name_space[in_flow] * self.sim_specs['dt'] < 0:
                                        self.name_space[in_flow] = self.stock_shadow_values[var] * -1 / self.sim_specs['dt']
                                        self.stock_shadow_values[var] = 0
                                    else:
                                        self.stock_shadow_values[var] += self.name_space[in_flow] * self.sim_specs['dt']
                                else:
                                    for sub, subval in self.name_space[in_flow].items():
                                        if self.stock_shadow_values[var][sub] + subval * self.sim_specs['dt'] < 0:
                                            self.name_space[in_flow][sub] = self.stock_shadow_values[var][sub] * -1 / self.sim_specs['dt']
                                            self.stock_shadow_values[var][sub] = 0
                                        else:
                                            self.stock_shadow_values[var][sub] += subval * self.sim_specs['dt']
                                
                            else:
                                if type(self.name_space[in_flow]) is not dict:
                                    self.stock_shadow_values[var] += self.name_space[in_flow] * self.sim_specs['dt']
                                else:
                                    for sub, subval in self.name_space[in_flow].items():
                                        self.stock_shadow_values[var][sub] += subval * self.sim_specs['dt']
                    if 'out' in self.stock_flows[var]:
                        out_flows = self.stock_flows[var]['out']
                        
                        # outflow prioritisation
                        # rule 1: first added first
                        # rule 2: dependents ranked higher
                        if len(out_flows) > 1:
                            for i in range(len(out_flows)-1, 0, -1):
                                for j in range(i):
                                    if self.is_dependent(out_flows[j+1], out_flows[j]):
                                        temp = out_flows[j+1]
                                        out_flows[j+1] = out_flows[j]
                                        out_flows[j] = temp

                        for out_flow in out_flows:
                            if out_flow not in self.name_space:
                                self.calculate_variable(var=out_flow, subscript=subscript, verbose=verbose)
                            if var in self.stock_non_negative:
                                if type(self.name_space[out_flow]) is not dict:
                                    if self.stock_shadow_values[var] - self.name_space[out_flow] * self.sim_specs['dt'] < 0:
                                        self.name_space[out_flow] = self.stock_shadow_values[var] / self.sim_specs['dt']
                                        self.stock_shadow_values[var] = 0
                                    else:
                                        self.stock_shadow_values[var] -= self.name_space[out_flow] * self.sim_specs['dt']
                                else:
                                    for sub, subval in self.name_space[out_flow].items():
                                        if self.stock_shadow_values[var][sub] - subval * self.sim_specs['dt'] < 0:
                                            self.name_space[out_flow][sub] = self.stock_shadow_values[var][sub] / self.sim_specs['dt']
                                            self.stock_shadow_values[var][sub] = 0
                                        else:
                                            self.stock_shadow_values[var][sub] -= subval * self.sim_specs['dt']
                            else:
                                if type(self.name_space[out_flow]) is not dict:
                                    self.stock_shadow_values[var] -= self.name_space[out_flow] * self.sim_specs['dt']
                                else:
                                    for sub, subval in self.name_space[out_flow].items():
                                        self.stock_shadow_values[var][sub] -= self.name_space[out_flow][sub] * self.sim_specs['dt']
                else: # for those stocks without flows connected:
                    pass
            
            # if the stock's shadow value has been updated, do nothing as the real value is already in name_space
            else:
                pass
        
        # C: var is a flow
        elif var in self.flow_equations:
            # var is a leakflow. In this case the conveyor needs to be initialised
            if var in self.leak_conveyors:
                if not leak_frac:
                    # if mode is not 'leak_frac', something other than the conveyor is requiring the leak_flow; 
                    # then it is the real value of the leak flow that is requested.
                    # then conveyor needs to be calculated. Otherwise it is the conveyor that requires it 
                    if var not in self.name_space: # the leak_flow is not calculated, which means the conveyor has not been initialised
                        self.calculate_variable(var=self.leak_conveyors[var], subscript=subscript)
                else:
                    # it is the value of the leak_fraction (a percentage) that is requested.    
                    # leak_fraction is calculated using leakflow's equation. 
                    parsed_equation = self.flow_equations_parsed[var]
                    self.calculate_dependents(parsed_equation=parsed_equation, verbose=verbose)
                    self.conveyors[self.leak_conveyors[var]]['leakflow'][var] = self.solver.calculate_node(parsed_equation=parsed_equation, verbose=verbose, var_name=var)

            elif var in self.outflow_conveyors:
                # requiring an outflow's value triggers the calculation of its connected conveyor
                if var not in self.name_space: # the outflow is not calculated, which means the conveyor has not been initialised
                    self.calculate_variable(var=self.outflow_conveyors[var], subscript=subscript)
                    
            elif var in self.flow_equations: # var is a normal flow
                if var not in self.name_space:
                    if type(parsed_equation) is dict:
                        for sub, sub_parsed_equaton in parsed_equation.items():
                            self.calculate_dependents(parsed_equation=sub_parsed_equaton, verbose=verbose)
                            value = self.solver.calculate_node(parsed_equation=sub_parsed_equaton, subscript=sub, verbose=verbose, var_name=var)
                            
                            # control flow positivity by itself
                            if self.flow_positivity[var] is True:
                                if value < 0:
                                    value = 0
                            if var not in self.name_space:
                                self.name_space[var] = dict()
                            self.name_space[var][sub] = value
                    elif var in self.var_dimensions and self.var_dimensions[var] is not None: # The variable is subscripted but all elements uses the same equation
                        self.calculate_dependents(parsed_equation=parsed_equation, verbose=verbose)
                        for sub in self.dimension_elements[self.var_dimensions[var]]:
                            value = self.solver.calculate_node(parsed_equation=parsed_equation, subscript=sub, verbose=verbose, var_name=var)
                            # control flow positivity by itself
                            if self.flow_positivity[var] is True:
                                if value < 0:
                                    value = 0
                            if var not in self.name_space:
                                self.name_space[var] = dict()
                            self.name_space[var][sub] = value
                    else:
                        self.calculate_dependents(parsed_equation=parsed_equation, verbose=verbose)
                        value = self.solver.calculate_node(parsed_equation=parsed_equation, subscript=subscript, verbose=verbose, var_name=var)

                        # control flow positivity by itself
                        if self.flow_positivity[var] is True:
                            if value < 0:
                                value = 0
                        
                        self.name_space[var] = value
                else:
                    pass
        
        # D: var is an auxiliary
        elif var in self.aux_equations:
            if var not in self.name_space:
                if type(parsed_equation) is dict:
                    for sub, sub_parsed_equation in parsed_equation.items():
                        self.calculate_dependents(parsed_equation=sub_parsed_equation, verbose=verbose)
                        value = self.solver.calculate_node(parsed_equation=sub_parsed_equation, subscript=sub, verbose=verbose, var_name=var)
                        if var not in self.name_space:
                            self.name_space[var] = dict()
                        self.name_space[var][sub] = value
                elif var in self.var_dimensions and self.var_dimensions[var] is not None: # The variable is subscripted but all elements uses the same equation
                    self.calculate_dependents(parsed_equation=parsed_equation, verbose=verbose)
                    for sub in self.dimension_elements[self.var_dimensions[var]]:
                        value = self.solver.calculate_node(parsed_equation=parsed_equation, subscript=sub, verbose=verbose, var_name=var)
                        if var not in self.name_space:
                            self.name_space[var] = dict()
                        self.name_space[var][sub] = value
                else:
                    self.calculate_dependents(parsed_equation=parsed_equation, verbose=verbose)
                    value = self.solver.calculate_node(parsed_equation=parsed_equation, subscript=subscript, verbose=verbose, var_name=var)
                    self.name_space[var] = value
                        
            else:
                pass
        
        else:
            raise Exception("Undefined var: {}".format(var))

    def calculate_variables(self, verbose=False):
        for var in (self.stock_equations_parsed | self.aux_equations_parsed | self.flow_equations_parsed).keys():
            self.calculate_variable(var=var, verbose=verbose)
    
    def update_conveyors(self):
        for conveyor_name, conveyor in self.conveyors.items(): # Stock is a Conveyor
            total_flow_effect = 0
            connections = self.stock_flows[conveyor_name]
            for direction, flows in connections.items():
                if direction == 'in':
                    for flow in flows:
                        total_flow_effect += self.name_space[flow]

            # in
            conveyor['conveyor'].inflow(total_flow_effect * self.sim_specs['dt'])
            self.stock_shadow_values[conveyor_name] = conveyor['conveyor'].level()
            
    def simulate(self, time=None, dt=None, dynamic=True, verbose=False, debug_against=False):
        if debug_against is not None:
            if debug_against is True:
                import pandas as pd
                self.df_debug_against = pd.read_csv('stella.csv')
            elif debug_against is False:
                pass
            else:
                self.df_debug_against = pd.read_csv(debug_against)

        self.parse(verbose=verbose)

        if time is None:
            time = self.sim_specs['simulation_time']
        if dt is None:
            dt = self.sim_specs['dt']
        steps = int(time/dt)

        if self.sim_specs['running']: # Continue simulation
            steps -= 1

        def step(debug=False):
            if verbose:
                # print('--time {} --'.format(self.sim_specs['current_time']))
                print('--step {} start--'.format(s))
                # print('\n--step {} start--\n'.format(s), self.name_space)
            self.calculate_variables(verbose=verbose)
            
            # snapshot current name space
            current_snapshot = deepcopy(self.name_space)
            current_snapshot[self.sim_specs['time_units']] = current_snapshot['TIME']
            current_snapshot.pop('TIME')
            
            # debug
            if debug is not False:
                all_var_validated = True
                invalid_vars = list()
                for var, val in current_snapshot.items():
                    if type(val) is dict:
                        try:
                            for sub, subval in val.items():
                                series_key = self.var_name_to_csv_entry(var, sub)
                                validated = abs(subval - debug[series_key]) <= 0.00001
                                if not validated:
                                    all_var_validated = False
                                    if var not in invalid_vars:
                                        invalid_vars.append(var)
                        except KeyError as e:
                            print(debug.keys())
                            raise e
                    else:
                        try:
                            series_key = self.var_name_to_csv_entry(var)
                            validated = abs(val - debug[series_key]) <= 0.00001
                            if not validated:
                                all_var_validated = False
                                invalid_vars.append(var)
                        except KeyError as e:
                            print(debug.keys())
                            raise e
                if not all_var_validated:
                    print('\n')
                    for invalid_var in invalid_vars:
                        print('Result of {} at Time [{}], Step [{}] invalid'.format(invalid_var, self.sim_specs['current_time'], self.current_step))
                        asdm_result = self.name_space[invalid_var]
                        print('asdm   result:', asdm_result)
                        if type(asdm_result) is dict:
                            for sub, subval in asdm_result.items():
                                print('Stella result:', debug[self.var_name_to_csv_entry(invalid_var+'[{}]'.format(', '.join(sub)))])
                        else:
                            print('Stella result:', debug[self.var_name_to_csv_entry(invalid_var)])
                        print('\n')
                        # self.trace_error(var_with_error=var)
                    raise Exception()
            
            self.time_slice[self.sim_specs['current_time']] = current_snapshot
            if verbose:
                print('\n--step {} finish--\n'.format(s))
                # print('\n--step {} finish--\n'.format(s), self.name_space, self.stock_shadow_values)
            self.update_conveyors()
            
            # prepare name_space for next step
            self.sim_specs['current_time'] += dt
            self.name_space.clear()
            for stock, stock_value in self.stock_shadow_values.items():
                self.name_space[stock] = stock_value
            self.stock_shadow_values.clear()
            
            self.name_space['TIME'] = self.sim_specs['current_time']
            self.name_space['DT'] = self.sim_specs['dt']
        
        self.current_step = 0
        for s in range(steps+1):
            if debug_against:
                step(debug=self.df_debug_against.iloc[self.current_step])
            else:
                step()
            self.current_step += 1
        
        self.sim_specs['running'] = True

    def trace_error(self, var_with_error, sub=None):
        self.debug_level += 1

        print(self.debug_level*'\t', 'Tracing error on {} ...'.format(var_with_error))
        print(self.debug_level*'\t', 'asdm value    :', self.name_space[var_with_error])
        print(self.debug_level*'\t', 'Expected value:', self.df_debug_against.iloc[self.current_step][self.var_name_to_csv_entry(var_with_error)])
        
        if sub is not None:
            parsed_equation = (self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)[var_with_error][sub]
        else:
            parsed_equation = (self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)[var_with_error]
        
        leafs = [x for x in parsed_equation.nodes() if parsed_equation.out_degree(x)==0]
        print(self.debug_level*'\t', 'Dependencies of {}:'.format(var_with_error))
        
        for leaf in leafs:
            # print(self.debug_level*'\t', parsed_equation.nodes[leaf])
            if parsed_equation.nodes[leaf]['operator'][0] in ['EQUALS', 'SPAREN']:
                operands = parsed_equation.nodes[leaf]['operands']
                if operands[0][0] == 'NUMBER':
                    pass
                elif operands[0][0] == 'NAME': # this refers to a variable like 'a'
                    var_dependent = operands[0][1]
                    print(self.debug_level*'\t', '-- Dependent:', var_dependent)
                    print(self.debug_level*'\t', '   asdm value    :', self.name_space[var_dependent])
                    print(self.debug_level*'\t', '   Expected value:', self.df_debug_against.iloc[self.current_step][self.var_name_to_csv_entry(var_dependent)])
        
                elif operands[0][0] == 'FUNC': # this refers to a subscripted variable like 'a[ele1]'
                    # need to find that 'SPAREN' node
                    var_dependent_node_id = operands[0][2]
                    var_dependent = parsed_equation.nodes[var_dependent_node_id]['operands'][0][1]
                    print(self.debug_level*'\t', '-- Dependent:', var_dependent)
                    print(self.debug_level*'\t', '   asdm value    :', self.name_space[var_dependent])
                    print(self.debug_level*'\t', '   Expected value:', self.df_debug_against.iloc[self.current_step][self.var_name_to_csv_entry(var_dependent)])
        
        if var_with_error in self.flow_stocks:
            connected_stocks = self.flow_stocks[var_with_error]
            for direction, connected_stock in connected_stocks.items():
                print(self.debug_level*'\t', '-- Connected stock: {:<4} {}'.format(direction, connected_stock))
                print(self.debug_level*'\t', '   asdm value    :', self.name_space[connected_stock])
                print(self.debug_level*'\t', '   Expected value:', self.df_debug_against.iloc[self.current_step][self.var_name_to_csv_entry(connected_stock)])
        
        print()
        self.debug_level -= 1

    def var_name_to_csv_entry(self, var, sub=None):
        if sub is None:
            series_key = var.replace('_', ' ')
        else:
            series_key = "{}[{}]".format(var, ', '.join(sub)).replace('_', ' ')
        
        if series_key[0].isdigit() or series_key[-1] == ')': # 1 day -> "1 day", a(b)-> "a(b)"
            series_key = '\"'+ series_key + '\"'
        return series_key
        
    def clear_last_run(self):
        self.sim_specs['current_time'] = self.sim_specs['initial_time']
        self.name_space = dict()
        self.name_space.update(self.env_variables)
        self.stock_shadow_values = dict()
        self.time_slice = dict()
        for stock_name, stock in self.stocks.items():
            stock.initialised = False

        self.stock_equations_parsed = dict()
        self.flow_equations_parsed = dict()
        self.aux_equations_parsed = dict()

        self.graph_functions = dict()
        self.graph_functions_renamed = dict()

        self.full_result = dict()
        self.full_result_flattened = dict()

        self.solver = Solver(
            sim_specs=self.sim_specs,
            dimension_elements=self.dimension_elements,
            name_space=self.name_space,
            graph_functions=self.graph_functions,
            )

        self.custom_functions = dict()

        self.sim_specs['running'] = False

    def summary(self):
        print('\nSummary:\n')
        # print('------------- Definitions -------------')
        # pprint(self.stock_equations | self.flow_equations | self.aux_equations)
        # print('')
        print('-------------  Sim specs  -------------')
        pprint(self.sim_specs)
        print('')
        print('-------------   Runtime   -------------')
        pprint(self.name_space)
        print('')
    
    def get_element_simulation_result(self, name, subscript=None):
        if not subscript:
            if type((self.stock_equations | self.flow_equations | self.aux_equations)[name]) is dict:
                result = dict()
                for sub in (self.stock_equations | self.flow_equations | self.aux_equations)[name].keys():
                    result[sub] = list()
                for time, slice in self.time_slice.items():
                    for sub, value in slice[name].items():
                        result[sub].append(value)
                return result
            else:
                result = list()
                for time, slice in self.time_slice.items():
                    result.append(slice[name])
                return result

        else:
            result= list()
            for time, slice in self.time_slice.items():
                result.append(slice[name][subscript])
            return result
            
    def export_simulation_result(self, flatten=False, format='dict', to_csv=False, dt=False):
        self.full_result = dict()
        self.full_result_df = None
        
        # generate full_result
        if dt:
            self.full_result['TIME'] = list()
        for time, slice in self.time_slice.items():
            if dt:
                self.full_result['TIME'].append(time)
            for var, value in slice.items():
                if var == 'DT':
                    continue
                if type(value) is dict:
                    for sub, subvalue in value.items():
                        try:
                            # self.full_result[var+'[{}]'.format(', '.join(sub))].append(subvalue)
                            self.full_result[var][sub].append(subvalue)
                        except:
                            try:
                                self.full_result[var][sub] = [subvalue]
                            except:
                                self.full_result[var] = dict()
                                self.full_result[var][sub] = [subvalue]
                else:
                    try:
                        self.full_result[var].append(value)
                    except:
                        self.full_result[var] = [value]
        
        # flatten the full_result
        self.full_result_flattened = dict()
        for var, result in self.full_result.items():
            if type(result) is dict:
                for sub, subresult in result.items():
                    self.full_result_flattened[var+'[{}]'.format(', '.join(sub))] = subresult
            else:
                self.full_result_flattened[var] = result
        
        if format == 'dict':
            if flatten:
                return self.full_result_flattened
            else:
                return self.full_result
        elif format == 'df':
            import pandas as pd
            self.full_result_df = pd.DataFrame.from_dict(self.full_result_flattened)
            self.full_result_df.drop([self.sim_specs['time_units']],axis=1, inplace=True)
            
            if to_csv:
                if type(to_csv) is not str:
                    self.full_result_df.to_csv('asdm.csv', index=False)
                else:
                    self.full_result_df.to_csv(to_csv, index=False)
                
            return self.full_result_df
    
    def display_results(self, variables=None):
        if type(variables) is list and len(variables) == 0:
            variables = list((self.stock_equations | self.flow_equations | self.aux_equations).keys())

        if type(variables) is str:
            variables = [variables]

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        
        if len(self.full_result) == 0:
            self.full_result = self.export_simulation_result()

        for var in variables:
            result = self.full_result[var]
            if type(result) is list:
                ax.plot(result, label='{}'.format(var))
            else:
                for sub, subresult in self.full_result[var].items():
                    ax.plot(subresult, label='{}[{}]'.format(var, ', '.join(sub)))
        
        ax.legend()
        plt.show()

    def get_dependent_graph(self, var, graph=None):
        if graph is None:
            graph = nx.DiGraph()

        # parse equations of the model into ast
        self.parse()
        all_equations = (self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)
        if var in self.env_variables: # like 'TIME'
            return graph
        parsed_equation = all_equations[var]
        dependent_variables = list()
        if type(parsed_equation) is not dict:
            leafs = [x for x in parsed_equation.nodes() if parsed_equation.out_degree(x)==0]
            for leaf in leafs:
                if parsed_equation.nodes[leaf]['operator'] == 'EQUALS':
                    dependent_variables.append(parsed_equation.nodes[leaf]['value'])
        else:
            for _, sub_eqn in parsed_equation.items():
                leafs = [x for x in sub_eqn.nodes() if sub_eqn.out_degree(x)==0]
                for leaf in leafs:
                    if sub_eqn.nodes[leaf]['operator'] == 'EQUALS':
                        dependent_variable = sub_eqn.nodes[leaf]['value']
                        if dependent_variable not in dependent_variables: # remove duplicates
                            dependent_variables.append(dependent_variable)
        
        # print(dependent_variables)
        if len(dependent_variables) == 0:
            return graph
        else:
            for dependent_var in dependent_variables:
                graph.add_edge(dependent_var, var)
                self.get_dependent_graph(dependent_var, graph)
            return graph

    def generate_dependent_graph(self, vars=None, show=False, loop=False):
        if vars is None:
            vars = list(self.flow_equations.keys())
        elif type(vars) is str:
            vars = [vars]
        
        dg = nx.DiGraph()

        for var in vars:
            dg_var = self.get_dependent_graph(var)
            dg = nx.compose(dg, dg_var)

        # create flow-to-stock edges if loop=True
        if loop:
            for flow in self.flow_stocks.keys():
                if flow in vars:
                    for stock in self.flow_stocks[flow].values():
                        dg.add_edge(flow, stock)
        
        if not show:
            return dg
        else:
            import matplotlib.pyplot as plt
            from networkx.drawing.nx_agraph import graphviz_layout
            pos = graphviz_layout(dg, prog='dot')
            # pos = nx.spring_layout(dg)
            nx.draw(
                dg, 
                pos, 
                with_labels=True, 
                node_size=300, 
                node_color="skyblue", 
                node_shape="s", 
                alpha=1, 
                linewidths=5
                )
            plt.show()