import networkx as nx
import numpy as np
import re
from itertools import product
from pprint import pprint
from scipy import stats
from scipy.interpolate import interp1d
from copy import deepcopy
import matplotlib.pyplot as plt
import logging

logger_parser = logging.getLogger('asdm.parser')
logger_solver = logging.getLogger('asdm.solver')
logger_graph_function = logging.getLogger('asdm.graph_function')
logger_conveyor = logging.getLogger('asdm.conveyor')
logger_data_feeder = logging.getLogger('asdm.data_feeder')
logger_sdmodel = logging.getLogger('asdm.simulator')

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: | %(name)s | %(message)s"
)

logger_parser.setLevel(logging.INFO)
logger_solver.setLevel(logging.INFO)
logger_graph_function.setLevel(logging.INFO)
logger_conveyor.setLevel(logging.INFO)
logger_data_feeder.setLevel(logging.INFO)
logger_sdmodel.setLevel(logging.INFO)

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
        self.logger = logger_parser
        
        self.numbers = {
            'NUMBER': r'(?<![a-zA-Z0-9)])[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
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

        self.functions = { # use lookahead (?=\s*\() to ensure only match INIT( or INIT  ( not INITIAL
            'MIN': r'MIN(?=\s*\()',
            'MAX': r'MAX(?=\s*\()',
            'SAFEDIV': r'SAFEDIV(?=\s*\()',
            'RBINOM': r'RBINOM(?=\s*\()',
            'INIT': r'INIT(?=\s*\()',
            'DELAY': r'DELAY(?=\s*\()',
            'DELAY1': r'DELAY1(?=\s*\()',
            'DELAY3': r'DELAY3(?=\s*\()',
            'SMTH1': r'SMTH1(?=\s*\()',
            'SMTH3': r'SMTH3(?=\s*\()',
            'STEP': r'STEP(?=\s*\()',
            'HISTORY': r'HISTORY(?=\s*\()',
            'LOOKUP': r'LOOKUP(?=\s*\()',
            'SUM': r'SUM(?=\s*\()',
            'PULSE': r'PULSE(?=\s*\()',
            'INT': r'INT(?=\s*\()',
            'LOG10': r'LOG10(?=\s*\()',
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
            self.logger.debug(f"Tokenising: {s} len: {len(s)}")
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

                    if token in self.dimension_elements.keys():
                        tokens.append(['DIMENSION', token])
                        s = s[m.span()[1]:].strip()
                    else:
                        if type_name in self.functions:
                            tokens.append(['FUNC', token])
                        else:
                            tokens.append([type_name, token])
                        s = s[m.span()[1]:].strip()
                    break
        
        return tokens
    
    def parse(self, expression, plot=False):
        # remove \n in the expression
        expression = expression.replace('\n', ' ')

        self.logger.debug("")
        self.logger.debug(f"Starting parse of expression: {expression}")
        self.tokens = self.tokenise(expression)
        self.logger.debug(f"Tokens: {self.tokens}")
        
        ast = self.parse_statement()
        if self.current_index != len(self.tokens):
            raise ValueError("Unexpected end of parsing of expression {} at index {} of tokens {}".format(expression, self.current_index, self.tokens))
        self.logger.debug("Completed parse")
        self.logger.debug(f"AST: {ast}")
        
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

        self.logger.debug(f"AST_graph {ast_graph.nodes.data(True)}")

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
        self.logger.debug("")
        self.logger.debug(f"parse_statement   {self.tokens[self.current_index:]}")
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
        self.logger.debug(f"parse_expression   {self.tokens[self.current_index:]}")
        return self.parse_or_expression()

    def parse_or_expression(self):
        """Parse an or expression."""
        self.logger.debug(f"parse_or_expr      {self.tokens[self.current_index:]}")
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
        self.logger.debug(f"parse_and_expr     {self.tokens[self.current_index:]}")
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
        self.logger.debug(f"parse_not_expr     {self.tokens[self.current_index:]}")
        if self.tokens[self.current_index][0] == 'NOT':
            self.current_index += 1
            self.node_id += 1
            return Node(node_id=self.node_id, operator='NOT', operands=[self.parse_statement()])
        return self.parse_compare_expression()
    
    def parse_compare_expression(self):
        """Parse a comparison expression."""
        self.logger.debug(f"parse_compare_expr {self.tokens[self.current_index:]}")
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
        self.logger.debug(f"parse_arith_expr   {self.tokens[self.current_index:]}")
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
        self.logger.debug(f"parse_mod          {self.tokens[self.current_index:]}")
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
        self.logger.debug(f"parse_term         {self.tokens[self.current_index:]} ")
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
        self.logger.debug(f"parse_exponent     {self.tokens[self.current_index:]}")
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
        self.logger.debug(f"parse_factor       {self.tokens[self.current_index:]}")
        token = self.tokens[self.current_index]
        
        # Handle unary minus (negation)
        if token[0] == 'MINUS':
            self.current_index += 1
            operand = self.parse_factor()  # Recursively parse the operand
            self.node_id += 1
            return Node(node_id=self.node_id, operator='UNARY_MINUS', operands=[operand])
        elif token[0] == 'LPAREN':
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
        self.logger.debug(f"parse_function_call{self.tokens[self.current_index:]}")
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
        self.logger.debug(f"parse_variable     {self.tokens[self.current_index:]}")
        var_name = self.tokens[self.current_index][1]
        self.current_index += 1
        if self.current_index < len(self.tokens) and self.tokens[self.current_index][0] == 'LSPAREN':
            subscripts = []
            subscripts_in_token = []
            subscript_in_token = []
            self.current_index += 1 # Skipping the opening '['
            # split subscript tokens by commas
            while self.tokens[self.current_index][0] != 'RSPAREN':
                # in runtime, referring to other element in the same dimension (e.g. another age group) can be done by
                # something like "Age-1", where Age is the dimension name.
                if self.tokens[self.current_index][0] != 'COMMA':
                    subscript_in_token.append(self.tokens[self.current_index]) # collect this token into the current subscript
                else:
                    subscripts_in_token.append(subscript_in_token)
                    subscript_in_token = []
                self.current_index += 1
            subscripts_in_token.append(subscript_in_token) # add the last subscript
            subscripts = subscripts_in_token
            self.current_index += 1 # Skipping the closing ']'
            self.node_id += 1
            return Node(node_id=self.node_id, operator='SPAREN', value=var_name, subscripts=subscripts)
        self.node_id += 1
        return Node(node_id=self.node_id, operator='EQUALS', value=var_name)

class Solver(object):
    def __init__(self, sim_specs=None, dimension_elements=None, var_dimensions=None, name_space=None, graph_functions=None):
        self.logger = logger_solver

        self.sim_specs = sim_specs # current_time, initial_time, dt, simulation_time, time_units
        self.dimension_elements = dimension_elements
        self.var_dimensions = var_dimensions
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

        def unary_minus(a):
            """Unary minus operator (negation)"""
            try:
                return -a
            except TypeError as e:
                if type(a) is dict:
                    o = dict()
                    for k in a:
                        o[k] = -a[k]
                    return o
                else:
                    raise e

        def unary_plus(a):
            """Unary plus operator"""
            try:
                return +a
            except TypeError as e:
                if type(a) is dict:
                    o = dict()
                    for k in a:
                        o[k] = +a[k]
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
            """ Safely divide a by b, handling scalars and dictionaries, with logging for division by zero. """
            
            def safe_div(x, y, key=None):
                """ Helper function to safely divide x by y and log warnings if y is zero. """
                if y == 0:
                    msg = f"Warning: Divide by zero encountered in divide({x}, {y}), returning 0"
                    if key is not None:
                        msg += f" for subscript '{key}'"
                    print(msg)
                    return 0
                return x / y

            # Scalar / Scalar
            if isinstance(a, (int, float, np.float64)) and isinstance(b, (int, float, np.float64)):
                return safe_div(a, b)

            # Dictionary / Dictionary
            if isinstance(a, dict) and isinstance(b, dict):
                return {k: safe_div(a[k], b.get(k, 1), k) for k in a}

            # Dictionary / Scalar
            if isinstance(a, dict) and isinstance(b, (int, float, np.float64)):
                return {k: safe_div(a[k], b, k) for k in a}

            # Scalar / Dictionary
            if isinstance(a, (int, float, np.float64)) and isinstance(b, dict):
                return {k: safe_div(a, b[k], k) for k in b}

            # Unsupported types
            self.logger.error(f"TypeError in divide(): Unsupported types {type(a)} and {type(b)}")
            raise TypeError(f"Unsupported types for division: {type(a)}, {type(b)}")
        
        def floor_divide(a, b):
            try:
                return a // b
            except TypeError as e:
                if type(a) is dict and type(b) is dict:
                    # self.logger.debug('    '*self.id_level+'[ '+var_name+' ] ', 'a//b', a, b)
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
                    # self.logger.debug('    '*self.id_level+'[ '+var_name+' ] ', 'a % b', a, b)
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
            # self.logger.debug('step:', stp, time)
            if sim_specs['current_time'] >= time:
                # self.logger.debug('step out:', stp)
                return stp
            else:
                # self.logger.debug('step out:', 0)
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
        
        def log10(a):
            return np.log10(a)
        
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
            'UNARY_MINUS': unary_minus,
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
            'LOG10':    log10,
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

    def calculate_node(self, var_name, parsed_equation, node_id='root', subscript=None):        
        self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v0 processing node {node_id}:")

        self.id_level += 1
        
        if type(parsed_equation) is dict:  
            raise Exception('Parsed equation should not be a dict. var:', var_name)

        if node_id == 'root':
            node_id = list(parsed_equation.successors('root'))[0]
        node = parsed_equation.nodes[node_id]
        self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] node: {node_id} {node}")
        node_operator = node['operator']
        node_value = node['value']
        node_subscripts_in_token = node['subscripts']
        node_operands = node['operands']
        if node_operator == 'IS':
            value = np.float64(node_value)
            self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v2 IS: {value}")
        elif node_operator == 'EQUALS':
            self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] operator v3 node_operator {node_operator}")
            self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] operator v3 node_value {node_value}")
            self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] operator v3 node_subscripts_in_token {node_subscripts_in_token}")
            self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] operands v3 node_operands {node_operands}")
            
            # Case 1: node_value is a variable name, e.g. "Population"
            if node_value in self.name_space.keys():
                node_var = node_value
                if subscript:
                    value = self.name_space[node_var]
                    if type(value) is dict:
                        value = value[subscript]
                        self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v3.1.1 EQUALS: subscript present, variable subscripted {value}")
                    else:
                        self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v3.1.2 EQUALS: subscript present, variable not subscripted {value}")
                else:
                    value = self.name_space[node_var]
                    self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v3.2 EQUALS: subscript not present {value}")
                
                self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v3 EQUALS: {value}")

            # Case 2: node_value is a dimension name, e.g. "Age"
            elif node_value in self.var_dimensions[var_name]: # only consider the dimension of the variable we are calculating
                # In this case, evaluate something like "Age=1" to determine if the current element is the one we are looking for.
                # Our job here is to return the order of the element (we are currently calculating) in the dimension.
                if subscript is not None:
                    self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v3.3 EQUALS: subscript present {subscript}")
                    dimension_order = list(self.var_dimensions[var_name]).index(node_value) # get the index of the dimension name in var_dimensions
                    self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v3.3.1 EQUALS: dimension {node_value} within {self.var_dimensions[var_name]} order {dimension_order}")
                    try:
                        element_order = self.dimension_elements[node_value].index(subscript[dimension_order])
                        self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v3.3.2 EQUALS: element {subscript[dimension_order]} within {self.var_dimensions[var_name][dimension_order]} order {element_order}")
                    except ValueError:
                        self.logger.error(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v3.3.2 EQUALS: element {subscript[dimension_order]} not found within dimension: elements {node_value}: {list(self.dimension_elements[node_value])}")
                        raise

                    value = element_order + 1 # +1 because the order starts from 0, but we want to return 1, 2, 3, etc.
                    self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v3.3 EQUALS: value {value}")
                else:
                    self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v3.4 EQUALS: subscript not present")
                    raise Exception(f'Subscript is not provided for dimension {node_value}. var: {var_name}')
            # Raise Exception('Dimension name should not be used as a variable name. var:', node_value)
            else:
                self.logger.error(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v3 EQUALS: name {node_value} is not defined in the name space or dimension elements.")
                raise Exception(f'Name {node_value} is not defined in the name space or dimension elements. var: {var_name}')

        elif node_operator == 'SPAREN': # TODO this part is too dynamic, therefore can be slow.
            var_name = node_value
            self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] a1 context subscript {subscript}")
            if node_subscripts_in_token is None: # only var_name; no subscript is specified
                self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] a1.1")
                # this could be 
                # (1) this variable (var_name) is not subscripted therefore the only value of it should be used;
                # (2) this variable (var_name) is subscripted in the same way as the variable using it (a contextual info is needed and provided in the arg subscript)
                if subscript:
                    self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] a1.1.1")
                    value = self.name_space[var_name][subscript] 
                else:
                    self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] a1.1.2")
                    value = self.name_space[var_name]
                self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v4.1 Sparen without sub: {value}")
            else: # there are explicitly specified subscripts in oprands like a[b]
                # print('subscripts from node definition', node_subscripts_in_token)
                # print('subscripts from context', subscript)
                # After allowing "Dimention-1" in the subscript, we need to ad-hoc construct the right subscripts to use for retrieving variable value
                # TODO: may need to consider more cases here.
                node_subscripts = []
                for subscript_in_token in node_subscripts_in_token:
                    # Case 1: just a subscript in the context, e.g. a[Element_1]
                    if len(subscript_in_token) == 1 and subscript_in_token[0][0] in ['NAME', 'NUMBER']:
                        node_subscripts.append(subscript_in_token[0][1]) # it's a dimension name, e.g. Dimension-1
                        self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] a1.2.1 subscript in token: {subscript_in_token[0][1]}")
                    # Case 2: subscript has in-line referencing to another element in the same dimension, e.g. a[Dimension-1]
                    elif len(subscript_in_token) == 3:
                        # step 1: find out the dimension name
                        dimension_name = subscript_in_token[0][1]
                        self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] a1.2.2.1 dimension name: {dimension_name}")
                        elements = self.dimension_elements[dimension_name]
                        # step 2: find out the offset direction
                        offset_operator = subscript_in_token[1][0]
                        self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] a1.2.2.2 offset operator: {offset_operator}")
                        # step 3: find out the offset amount
                        offset_amount = subscript_in_token[2][1]
                        self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] a1.2.2.3 offset amount: {offset_amount}")
                        
                        # We need to read from the context subscript the current element in the relevant dimension
                        # in case of wrong order in context subscripts, we check every one of them if they belong to elements
                        for context_element in subscript:
                            if context_element in elements:
                                ind_context_element = elements.index(context_element)
                                self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] a1.2.2.4 context element index: {ind_context_element}")
                                if offset_operator == 'MINUS':
                                    ind_new_element = ind_context_element - int(offset_amount)
                                elif offset_operator == 'PLUS':
                                    ind_new_element = ind_context_element + int(offset_amount)
                                else:
                                    raise Exception(f"Invalid offset operator {offset_operator} in subscript {subscript_in_token}.")
                                self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] a1.2.2.5 new element index: {ind_new_element}")
                                new_element = elements[ind_new_element]
                                self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] a1.2.2.6 new element: {new_element}")
                                node_subscripts.append(new_element)
                                break
                    else:
                        raise Exception(f"Invalid length of subscript tokens {subscript_in_token}: {len(subscript_in_token)}, should be 1 or 3.")

                self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] a1.3 subscript from node definition: {node_subscripts[:]} subscript from context: {subscript}")
                # prioritise subscript from node definition
                try:
                    subscript_from_definition = tuple(node_subscripts[:]) # use tuple to make it hashable
                    value = self.name_space[var_name][subscript_from_definition]
                    self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] a1.3.1 value{str(value)}")
                except KeyError as e: # subscript in operands looks like a[Dimension_1, Element_1], inference needed
                    self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] a1.3.2 subscript in operands contains dimension name(s)")
                    if subscript: # there's subscript in context
                        subscript_from_definition = node_subscripts[:] # definition is what user put in equation, should take higher priority
                        subscript_from_definition_with_replacement = list()
                        subscript_from_context_index = 0
                        for i in range(len(subscript_from_definition)):
                            if subscript_from_definition[i] in self.var_dimensions[var_name]: # it's sth like Dimension_1 - needed to be replaced by the contextual element as it's not specified
                                dimension_from_definition = subscript_from_definition[i]
                                # now need to find out which element in the context subscript corresponds to this dimension
                                while subscript_from_context_index < len(subscript) and subscript[subscript_from_context_index] not in self.dimension_elements[dimension_from_definition]:
                                    subscript_from_context_index += 1
                                self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] a1.3.2.1 replace {dimension_from_definition} with {subscript[subscript_from_context_index]} from context subscript {subscript}")
                                subscript_from_definition_with_replacement.append(subscript[subscript_from_context_index]) # take the element from context subscript in the same position to replace Dimension_1
                                subscript_from_context_index += 1
                            else: # it's sth like Element_1 - specified by the user, should take priority
                                self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] a1.3.2.2 keep {subscript_from_definition[i]} as is, since it is not in dimension names of this model")
                                subscript_from_definition_with_replacement.append(subscript_from_definition[i]) # add to list directly
                        subscript_from_definition_with_replacement = tuple(subscript_from_definition_with_replacement)
                        self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] a1.3.2.2 {subscript_from_definition_with_replacement}")
                        value = self.name_space[var_name][subscript_from_definition_with_replacement] # try if subscript is Element_1
                    else: # there's no subscript in context
                        raise e
                        
                self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v4.2 SPAREN with sub: {value}")
        
        elif node_operator in self.built_in_functions.keys(): # plus, minus, con, etc.
            self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] operator v7 Built-in operator: {node_operator}, {node_operands}")
            func_name = node_operator
            function = self.built_in_functions[func_name]
            oprds = []
            for operand in node_operands:
                self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v7.1 operand {operand}")
                v = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=operand, subscript=subscript)
                self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v7.2 value {v} {subscript}")
                oprds.append(v)
            self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v7.3 operands {oprds}")
            value = function(*oprds)
            self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v7 Built-in operator {node_operator}: {value}")
        
        elif node_operator in self.custom_functions.keys(): # graph functions
            self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] custom func operator {node_operator}")
            func_name = node_operator
            function = self.custom_functions[func_name]
            oprds = []
            for operand in node_operands:
                self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] operand {operand}")
                v = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=operand, subscript=subscript)
                self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] value {v}")
                oprds.append(v)
            self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] operands {oprds}")
            value = function(*oprds)
            self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v8 GraphFunc: {value}")

        elif node_operator in self.time_related_functions: # init, delay, etc
            self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] time-related func. operator: {node_operator} operands {node_operands}")
            func_name = node_operator
            if func_name == 'INIT':
                if tuple([parsed_equation, node_id, node_operands[0]]) in self.time_expr_register.keys():
                    value = self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])]
                else:
                    value = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[0], subscript=subscript)
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])] = value
            elif func_name == 'DELAY':
                # expr value
                expr_value = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[0], subscript=subscript)
                if tuple([parsed_equation, node_id, node_operands[0]]) in self.time_expr_register.keys():
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])].append(expr_value)
                else:
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])] = [expr_value]
                
                # init value
                if len(node_operands) == 2: # there's no initial value specified -> use the delayed expr's initial value
                    init_value = self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][0]
                elif len(node_operands) == 3: # there's an initial value specified
                    init_value = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[2], subscript=subscript)
                else:
                    raise Exception("Invalid initial value for DELAY in operands {}".format(node_operands))

                # delay time
                delay_time = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[1], subscript=subscript)
                if delay_time > (self.sim_specs['current_time'] - self.sim_specs['initial_time']): # (- initial_time) because simulation might not start from time 0
                    value = init_value
                else:
                    delay_steps = delay_time / self.sim_specs['dt']
                    value = self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][-int(delay_steps+1)]
            elif func_name == 'DELAY1':
                # args values
                order = 1
                expr_value = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[0], subscript=subscript)
                delay_time = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[1], subscript=subscript)

                if len(node_operands) == 3:
                    init_value = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[2], subscript=subscript)
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
                expr_value = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[0], subscript=subscript)
                delay_time = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[1], subscript=subscript)
                if len(node_operands) == 3:
                    init_value = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[2], subscript=subscript)
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
                expr_value = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[0], subscript=subscript)
                if tuple([parsed_equation, node_id, node_operands[0]]) in self.time_expr_register.keys():
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])].append(expr_value)
                else:
                    self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])] = [expr_value]
                
                # historical time
                historical_time = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[1], subscript=subscript)
                if historical_time > self.sim_specs['current_time'] or historical_time < self.sim_specs['initial_time']:
                    value = 0
                else:
                    historical_steps = (historical_time - self.sim_specs['initial_time']) / self.sim_specs['dt']
                    value = self.time_expr_register[tuple([parsed_equation, node_id, node_operands[0]])][int(historical_steps)]
            
            elif func_name == 'SMTH1':
                # arg values
                order = 1
                expr_value = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[0], subscript=subscript)
                if type(expr_value) is dict:
                    if subscript is not None:
                        expr_value = expr_value[subscript]
                    else:
                        raise Exception('Invalid subscript.')
                smth_time = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[1], subscript=subscript)
                if len(node_operands) == 3:
                    init_value = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[2], subscript=subscript)
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
                expr_value = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[0], subscript=subscript)
                if type(expr_value) is dict:
                    if subscript is not None:
                        expr_value = expr_value[subscript]
                    else:
                        raise Exception('Invalid subscript.')
                smth_time = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[1], subscript=subscript)
                if len(node_operands) == 3:
                    init_value = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[2], subscript=subscript)
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
            self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v9 Time-related Func: {value}")
        
        elif node_operator in self.array_related_functions: # Array-RELATED
            self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] array-related func. operator: {node_operator} operands: {node_operands}")
            func_name = node_operator
            if func_name == 'SUM':
                arrayed_var_name = parsed_equation.nodes[node_operands[0]]['value']
                sum_array = 0
                for _, sub_val in self.name_space[arrayed_var_name].items():
                    sum_array += sub_val
                value = sum_array
            self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v10 Array-related Func: {value}")
        
        elif node_operator in self.lookup_functions: # LOOKUP
            self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] Lookup func. operator: {node_operator} operands: {node_operands}")
            func_name = node_operator
            if func_name == 'LOOKUP':
                look_up_func_node_id = node_operands[0]
                look_up_func_name = parsed_equation.nodes[look_up_func_node_id]['value']
                look_up_func = self.graph_functions[look_up_func_name]
                input_value = self.calculate_node(var_name=var_name, parsed_equation=parsed_equation, node_id=node_operands[1], subscript=subscript)
                value = look_up_func(input_value)
            else:
                raise Exception('Unknown Lookup function {}'.format(node_operator))
        
        else:
            raise Exception('Unknown operator {}'.format(node_operator))
        
        self.id_level -= 1

        self.logger.debug(f"{'    '*self.id_level}[ {var_name}:{subscript} ] v0 value for node {node_id}: {value}")

        return value


class GraphFunc(object):
    def __init__(self, out_of_bound_type, yscale, ypts, xscale=None, xpts=None):
        self.logger = logger_graph_function

        self.out_of_bound_type = out_of_bound_type
        self.yscale = yscale
        self.xscale = xscale
        self.xpts = xpts
        self.ypts = ypts
        self.eqn = None
        self.initialize()
    
    def initialize(self):
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
            # self.logger.debug("Warning: new set of x points have a different length to the old set.")
        self.xpts = xpts
        
    def overwrite_xscale(self, xscale):
        self.xscale = xscale
        self.xpts = None # to auto-infer self.xpts from self.xscale, self.xpts must set to None

    def overwrite_ypts(self, ypts):
        # if len(self.ypts) != len(ypts):
            # self.logger.debug("Warning: new set of y points have a different length to the old set.")
        self.ypts = ypts


class Conveyor(object):
    def __init__(self, length, eqn):
        self.logger = logger_conveyor

        self.length_time_units = length
        self.equation = eqn
        self.length_steps = None # to be decided at runtime
        self.total = 0 # to be decided when initializing stocks
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
            # self.logger.debug('Conveyor N total leaks:', n_leak)
            self.output = self.total / (self.length_steps + (n_leak * self.leak_fraction) / ((1-self.leak_fraction)*self.length_steps))
            # self.logger.debug('Conveyor Output:', output)
            leak = self.output * (self.leak_fraction/((1-self.leak_fraction)*self.length_steps))
            # self.logger.debug('Conveyor Leak:', leak)
            # generate slats
            for i in range(self.length_steps):
                self.slats.append(self.output + (i+1)*leak)
                self.leaks.append(leak)
            self.slats.reverse()
        # self.logger.debug('Conveyor initialized:', self.conveyor, '\n')
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
        self.initialized = False


class DataFeeder(object):
    def __init__(self, data, from_time=0, data_dt=1, interpolate=False):
        """
        data: a list

        """
        self.logger = logger_data_feeder
        self.interpolate = interpolate
        self.data_dt = data_dt
        self.from_time =from_time
        self.time_data = dict()
        time = self.from_time
        for d in data:
            self.time_data[time] = d
            time += self.data_dt
        # self.logger.debug(self.time_data)
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
    def __init__(self, from_xmile=None, parser_debug_level='info', solver_debug_level='info', simulator_debug_level='info'):
        # Debug
        self.HEAD = 'ENGINE'
        self.debug_level_trace_error = 0
        self.logger = logger_sdmodel

        # model debug level
        if simulator_debug_level == 'debug':
            self.logger.setLevel(logging.DEBUG)
        elif simulator_debug_level == 'info':
            self.logger.setLevel(logging.INFO)
        elif simulator_debug_level == 'warning':
            self.logger.setLevel(logging.WARNING)
        elif simulator_debug_level == 'error':
            self.logger.setLevel(logging.ERROR)
        else:
            raise Exception('Unknown debug level {}'.format(simulator_debug_level))

        # sim_specs
        self.sim_specs = {
            'initial_time': 0,
            'current_time': 0,
            'dt': 0.25,
            'simulation_time': 13,
            'time_units' :'Weeks',
        }


        # dimensions
        self.var_dimensions = dict() # 'dim1':['ele1', 'ele2']
        self.dimension_elements = dict()
        
        # stocks
        self.stocks = dict()
        self.stock_equations = dict()
        self.stock_equations_parsed = dict()
        self.stock_non_negative = dict()
        self.stock_non_negative_temp_value = dict()
        self.stock_non_negative_out_flows = dict()

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

        # parser debug level
        if parser_debug_level == 'debug':
            self.parser.logger.setLevel(logging.DEBUG)
        elif parser_debug_level == 'info':
            self.parser.logger.setLevel(logging.INFO)
        elif parser_debug_level == 'warning':
            self.parser.logger.setLevel(logging.WARNING)
        elif parser_debug_level == 'error':
            self.parser.logger.setLevel(logging.ERROR)
        else:
            raise Exception('Unknown debug level {}'.format(parser_debug_level))

        # dependency graphs
        self.dg_init = nx.DiGraph()
        self.dg_iter = nx.DiGraph()
        self.ordered_vars_init = list()
        self.ordered_vars_iter = list()

        # solver
        self.solver = Solver(
            sim_specs=self.sim_specs,
            dimension_elements=self.dimension_elements,
            var_dimensions=self.var_dimensions,
            name_space=self.name_space,
            graph_functions=self.graph_functions,
        )

        # solver debug level
        if solver_debug_level == 'debug':
            self.solver.logger.setLevel(logging.DEBUG)
        elif solver_debug_level == 'info':
            self.solver.logger.setLevel(logging.INFO)
        elif solver_debug_level == 'warning':
            self.solver.logger.setLevel(logging.WARNING)
        elif solver_debug_level == 'error':
            self.solver.logger.setLevel(logging.ERROR)
        else:
            raise Exception('Unknown debug level {}'.format(solver_debug_level))

        # custom functions
        self.custom_functions = {}
        
        # state
        self.state = 'created'

        # If the model is based on an XMILE file
        if from_xmile is not None:
            # self.logger.debug(self.HEAD, 'Reading XMILE model from {}'.format(from_xmile))
            from pathlib import Path
            xmile_path = Path(from_xmile)
            if xmile_path.exists():
                with open(xmile_path, encoding='utf-8') as f:
                    xmile_content = f.read()
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
                    dimensions = subscripts_root.find_all('dim')

                    dims = dict()
                    for dimension in dimensions:
                        name = dimension.get('name')
                        try:
                            size = dimension.get('size')
                            dims[name] = [str(i) for i in range(1, int(size)+1)]
                        except:
                            elems = dimension.find_all('elem')
                            elem_names = list()
                            for elem in elems:
                                elem_names.append(elem.get('name'))
                            dims[name] = elem_names
                    self.dimension_elements.update(dims) # need to use update here to do the 'True' assignment
                except AttributeError:
                    pass
                
                # read variables
                variables_root = BeautifulSoup(xmile_content, 'xml').find('variables') # omit names in view
                stocks = variables_root.find_all('stock')
                flows = variables_root.find_all('flow')
                auxiliaries = variables_root.find_all('aux')
                
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
                        var_dimensions = var.find('dimensions').find_all('dim')
                        # self.logger.debug('Found dimensions {}:'.format(var), var_dimensions)

                        var_dims = dict()
                        for dimension in var_dimensions:
                            dim_name = dimension.get('name')
                            self.var_dimensions[self.name_handler(var.get('name'))].append(dim_name)
                            var_dims[dim_name] = dims[dim_name]
                        
                        var_subscripted_eqn = dict()
                        var_elements = var.find_all('element')
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
                        # self.logger.debug('nonnegstock', stock)
                        non_negative = True
                    
                    is_conveyor = False
                    if stock.find('conveyor'):
                        is_conveyor = True

                    inflows = stock.find_all('inflow')
                    outflows = stock.find_all('outflow')
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

                self.state = 'loaded'

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

        self.state = 'loaded'
    
    def add_flow(self, name, equation, leak=None, non_negative=False):
        if type(equation) in [int, float, np.int_, np.float64]:
            equation = str(equation)
        self.flow_positivity[name] = non_negative
        if leak:
            self.leak_conveyors[name] = None # to be filled when parsing the conveyor
        self.flow_equations[name] = equation

        self.state = 'loaded'
    
    def add_aux(self, name, equation):
        if type(equation) in [int, float, np.int_, np.float64]:
            equation = str(equation)
        self.aux_equations[name] = equation

        self.state = 'loaded'

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

        if self.state == 'loaded':
            pass
        elif self.state == 'simulated':
            self.state = 'changed'

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
            # self.logger.debug('Old xpts:', graph_func_equation.xpts)
            graph_func_equation.overwrite_xpts(new_xpts)
            # self.logger.debug('New xpts:', graph_func_equation.xpts)
        
        if new_xscale is not None:
            # self.logger.debug('Old xscale:', graph_func_equation.xscale)
            graph_func_equation.overwrite_xscale(new_xscale)
            # self.logger.debug('New xscale:', graph_func_equation.xscale)
        
        if new_ypts is not None:
            # self.logger.debug('Old ypts:', graph_func_equation.ypts)
            graph_func_equation.overwrite_ypts(new_ypts)
            # self.logger.debug('New ypts:', graph_func_equation.ypts)
        
        graph_func_equation.initialize()

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
    
    def batch_parse(self, equations, parsed_equations):
        # Debug logic: collect all equations that cannot be parsed and log them, then end the parsing process.
        unparsed_equations = list()
        counter_all_equations = 0
        counter_unparsed_variables = 0
        counter_all_variables = 0

        for var, equation in equations.items():
            # self.logger.debug("Parsing: {}".format(var))
            # self.logger.debug("    Eqn: {}".format(equation))
            
            if type(equation) is dict:
                un_parsed = False
                parsed_equations[var] = dict()
                for k, ks in equation.items():
                    try:
                        parsed_equations[var][k] = self.parse_equation(var=var, equation=ks)
                        counter_all_equations += 1
                    except Exception as e:
                        self.logger.error("Error parsing equation for variable {}: {}".format(var, e))
                        unparsed_equations.append(((var, k), ks, e))
                        counter_all_equations += 1
                        un_parsed = True
                if un_parsed:
                    counter_unparsed_variables += 1
            else:
                try:
                    parsed_equations[var] = self.parse_equation(var=var, equation=equation)
                    counter_all_equations += 1
                except Exception as e:
                    self.logger.error("Error parsing equation for variable {}: {}".format(var, e))
                    unparsed_equations.append((var, equation, e))
                    counter_all_equations += 1
                    counter_unparsed_variables += 1
                    exit()
            counter_all_variables += 1
        
        if len(unparsed_equations) > 0:
            self.logger.error(f"The following {len(unparsed_equations)} equations (out of {counter_all_equations}) could not be parsed:")
            self.logger.error("")
            for i in range(len(unparsed_equations)):
                var, eqn, error = unparsed_equations[i]
                self.logger.error("{} Variable: {}".format(i+1, var))
                self.logger.error("{} Equation: {}".format(i+1, eqn))
                self.logger.error("{} Error: {}".format(i+1, error))
                self.logger.error("")
            raise Exception(f"Parsing failed for {len(unparsed_equations)} equations out of {counter_all_equations} ({counter_unparsed_variables} variables out of {counter_all_variables}). See logs for details.")

    def parse(self):
        # string equation -> calculation tree

        self.batch_parse(self.stock_equations, self.stock_equations_parsed)
        self.batch_parse(self.flow_equations, self.flow_equations_parsed)
        self.batch_parse(self.aux_equations, self.aux_equations_parsed)

        self.state = 'parsed'

    def is_dependent(self, var1, var2):
        # determine if var2 depends directly on var1, i.e., var1 --> var2 or var1 appears in var2's equation
        
        def is_dependent_sub(var1, parsed_equation_var2, dependent=False):
            leafs = [x for x in parsed_equation_var2.nodes() if parsed_equation_var2.out_degree(x)==0]
            for leaf in leafs:
                dependent = False
                operator = parsed_equation_var2.nodes[leaf]['operator']
                if operator == 'EQUALS':
                    value = parsed_equation_var2.nodes[leaf]['value']
                    if value == var1:
                        dependent = True
                elif operator == 'SPAREN': # TODO: This branch needs further test
                    operands = parsed_equation_var2.nodes[leaf]['operands']
                    if operands[0][0] == 'FUNC': # this refers to a subscripted variable like 'a[ele1]'
                        # need to find that 'SPAREN' node
                        var_dependent_node_id = operands[0][2]
                        var_dependent = parsed_equation_var2.nodes[var_dependent_node_id]['operands'][0][1]
                        if var_dependent == var1:
                            dependent = True
                            break
                else:
                    pass
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

    def calculate_variable(self, var, dg, subscript=None, leak_frac=False, conveyor_init=False, conveyor_len=False):
        if leak_frac or conveyor_init or conveyor_len:
            self.logger.debug(f"    Calculating: {var:<15} on subscript {subscript}; flags leak_frac={leak_frac}, conveyor_init={conveyor_init}, conveyor_len={conveyor_len}")
        else:
            self.logger.debug(f"    Calculating: {var:<15} on subscript {subscript}")
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
            # self.logger.debug('Calculating Conveyor {}'.format(var))
            if not (conveyor_init or conveyor_len):
                if not self.conveyors[var]['conveyor'].is_initialized:
                    self.logger.debug(f"    Initializing conveyor {var}")
                    # when initializing, equation of the conveyor needs to be evaluated, using flag conveyor_len=True 
                    self.calculate_variable(var=var, dg=dg, subscript=subscript, conveyor_len=True)
                    conveyor_length = self.conveyors[var]['len']
                    length_steps = int(conveyor_length/self.sim_specs['dt'])
                    
                    # when initializing, equation of the conveyor needs to be evaluated, using flag conveyor_init=True 
                    self.calculate_variable(var=var, dg=dg, subscript=subscript, conveyor_init=True)
                    conveyor_init_value = self.conveyors[var]['val']
                    
                    leak_flows = self.conveyors[var]['leakflow']
                    if len(leak_flows) == 0:
                        leak_fraction = 0
                    else:
                        for leak_flow in leak_flows.keys():
                            self.calculate_variable(var=leak_flow, dg=dg, subscript=subscript, leak_frac=True)
                            leak_fraction = self.conveyors[var]['leakflow'][leak_flow] # TODO multiple leakflows
                    self.conveyors[var]['conveyor'].initialize(length_steps, conveyor_init_value, leak_fraction)
                    
                    # put initialized conveyor value to name_space
                    value = self.conveyors[var]['conveyor'].level()
                    self.name_space[var] = value

                    self.logger.debug(f"    Initialized conveyor {var}")
                
                if var not in self.stock_shadow_values:
                    # self.logger.debug("Updating {} and its outflows".format(var))
                    # self.logger.debug("    Name space1:", self.name_space)
                    # leak
                    for leak_flow, leak_fraction in self.conveyors[var]['leakflow'].items():
                        if leak_flow not in self.name_space: 
                            # self.logger.debug('    Calculating leakflow {} for {}'.format(leak_flow, var))
                            leaked_value = self.conveyors[var]['conveyor'].leak_linear()
                            self.name_space[leak_flow] = leaked_value / self.sim_specs['dt'] # TODO: we should also consider when leak flows are subscripted
                    # out
                    for outputflow in self.conveyors[var]['outputflow']:
                        if outputflow not in self.name_space:
                            # self.logger.debug('    Calculating outflow {} for {}'.format(outputflow, var))
                            outflow_value = self.conveyors[var]['conveyor'].outflow()
                            self.name_space[outputflow] = outflow_value / self.sim_specs['dt']
                    # self.logger.debug("    Name space2:", self.name_space)
                    self.stock_shadow_values[var] = self.conveyors[var]['conveyor'].level()

            elif conveyor_len:
                # self.logger.debug('Calculating LEN for {}'.format(var))
                # it is the intitial value of the conveyoer
                parsed_equation = self.stock_equations_parsed[var][0]
                self.conveyors[var]['len'] = self.solver.calculate_node(var_name=var, parsed_equation=parsed_equation)

            elif conveyor_init:
                # self.logger.debug('Calculating INIT VAL for {}'.format(var))
                # it is the intitial value of the conveyoer
                parsed_equation = self.stock_equations_parsed[var][1]
                self.conveyors[var]['val'] = self.solver.calculate_node(var_name=var, parsed_equation=parsed_equation)
        
        # B: var is a normal stock
        elif var not in self.conveyors and var in self.stocks:
            if not self.stocks[var].initialized:
                self.logger.debug(f"    Stock {var} not initialized")
                if type(parsed_equation) is dict:
                    for sub, sub_parsed_equation in parsed_equation.items():
                        sub_value = self.solver.calculate_node(var_name=var, parsed_equation=sub_parsed_equation, subscript=sub)
                        if var not in self.name_space:
                            self.name_space[var] = dict()
                        self.name_space[var][sub] = sub_value
                elif var in self.var_dimensions and self.var_dimensions[var] is not None: # The variable is subscripted but all elements uses the same equation
                    for sub in self.dimension_elements[self.var_dimensions[var]]:
                        sub_value = self.solver.calculate_node(var_name=var, parsed_equation=parsed_equation, subscript=sub)
                        if var not in self.name_space:
                            self.name_space[var] = dict()
                        self.name_space[var][sub] = sub_value
                else: # The variable is not subscripted
                    value = self.solver.calculate_node(var_name=var, parsed_equation=parsed_equation, subscript=subscript)
                    self.name_space[var] = value
                
                self.stocks[var].initialized = True
                self.stock_shadow_values[var] = deepcopy(self.name_space[var])
                if self.stock_non_negative[var] is True:
                    self.stock_non_negative_temp_value[var] = deepcopy(self.name_space[var])

                self.logger.debug(f"    Stock {var} initialized = {self.name_space[var]}")
            
            else:
                if self.stock_non_negative[var] is True:
                    self.logger.debug(f"    Stock {var} already initialized = {self.name_space[var]}, temp value: {self.stock_non_negative_temp_value[var]}")
                else:
                    self.logger.debug(f"    Stock {var} already initialized = {self.name_space[var]}")
        
        # C: var is a flow
        elif var in self.flow_equations:
            # var is a leakflow. In this case the conveyor needs to be initialized
            if var in self.leak_conveyors:
                if not leak_frac:
                    # if mode is not 'leak_frac', something other than the conveyor is requiring the leak_flow; 
                    # then it is the real value of the leak flow that is requested.
                    # then conveyor needs to be calculated. Otherwise it is the conveyor that requires it 
                    if var not in self.name_space: # the leak_flow is not calculated, which means the conveyor has not been initialized
                        self.calculate_variable(var=self.leak_conveyors[var], dg=dg, subscript=subscript)
                else:
                    # it is the value of the leak_fraction (a percentage) that is requested.    
                    # leak_fraction is calculated using leakflow's equation. 
                    parsed_equation = self.flow_equations_parsed[var]
                    self.conveyors[self.leak_conveyors[var]]['leakflow'][var] = self.solver.calculate_node(var_name=var, parsed_equation=parsed_equation)

            elif var in self.outflow_conveyors:
                # requiring an outflow's value triggers the calculation of its connected conveyor
                if var not in self.name_space: # the outflow is not calculated, which means the conveyor has not been initialized
                    self.calculate_variable(var=self.outflow_conveyors[var], dg=dg, subscript=subscript)

            elif var in self.flow_equations: # var is a normal flow
                if var not in self.name_space:
                    if type(parsed_equation) is dict:
                        for sub, sub_parsed_equation in parsed_equation.items():
                            sub_value = self.solver.calculate_node(var_name=var, parsed_equation=sub_parsed_equation, subscript=sub)
                            if var not in self.name_space:
                                self.name_space[var] = dict()
                            self.name_space[var][sub] = sub_value
                    elif var in self.var_dimensions and self.var_dimensions[var] is not None: # The variable is subscripted but all elements uses the same equation
                        for sub in self.dimension_elements[self.var_dimensions[var]]:
                            sub_value = self.solver.calculate_node(var_name=var, parsed_equation=parsed_equation, subscript=sub)
                            if var not in self.name_space:
                                self.name_space[var] = dict()
                            self.name_space[var][sub] = sub_value
                    else:
                        value = self.solver.calculate_node(var_name=var, parsed_equation=parsed_equation, subscript=subscript)
                        self.name_space[var] = value

                    # control flow positivity by itself
                    if self.flow_positivity[var] is True:
                        if type(self.name_space[var]) is dict:
                            for sub, sub_value in self.name_space[var].items():
                                if sub_value < 0:
                                    self.name_space[var][sub] = np.float64(0)
                                    self.logger.debug(f"    Flow {var}[{sub}] is negative, set to 0")
                        else:
                            if self.name_space[var] < 0:
                                self.name_space[var] = np.float64(0)
                                self.logger.debug('    '+"Flow {} is negative, set to 0".format(var))

                    # do not use 'value' from here on, use 'self.name_space[var]' instead
                    # check flow attributes for its constraints from non-negative stocks
                    flow_attributes = dg.nodes[var]
                    self.logger.debug('    '+'Checking attributes: {}'.format(flow_attributes))
                    
                    if 'considered_for_non_negative_stock' in flow_attributes:
                        if flow_attributes['considered_for_non_negative_stock'] is True:
                            flow_to_stock = self.flow_stocks[var]['to']
                            self.logger.debug('    '+f'----considering inflow {var} into non-negative stocks {flow_to_stock} whose temp value is {self.stock_non_negative_temp_value[flow_to_stock]}')
                            # this is an in_flow and this in_flow should be considered before constraining out_flows

                            # To prevent a negative inflow from making the stock negative, we need to constrain the inflow
                            # This only happens if the inflow is a biflow
                            if self.flow_positivity[var] is False:
                                if type(self.name_space[var]) is dict:
                                    for sub, sub_value in self.name_space[var].items():
                                        if self.stock_non_negative_temp_value[flow_to_stock][sub] + sub_value * self.sim_specs['dt'] < 0:
                                            self.name_space[var][sub] = self.stock_non_negative_temp_value[flow_to_stock][sub] / self.sim_specs['dt'] *-1 # this outcome is different from Stella, but it is more reasonable. See AwkwardStockFlow.stmx, stock10
                                            self.stock_non_negative_temp_value[flow_to_stock][sub] = np.float64(0)
                                        else:
                                            self.stock_non_negative_temp_value[flow_to_stock][sub] += sub_value * self.sim_specs['dt']
                                elif var in self.var_dimensions and self.var_dimensions[var] is not None: # The variable is subscripted but all elements uses the same equation
                                    for sub in self.stock_non_negative_temp_value[flow_to_stock]:
                                        if self.stock_non_negative_temp_value[flow_to_stock][sub] + self.name_space[var] * self.sim_specs['dt'] < 0:
                                            self.name_space[var] = self.stock_non_negative_temp_value[flow_to_stock][sub] / self.sim_specs['dt'] *-1
                                            self.stock_non_negative_temp_value[flow_to_stock][sub] = np.float64(0)
                                        else:
                                            self.stock_non_negative_temp_value[flow_to_stock][sub] += self.name_space[var] * self.sim_specs['dt']
                                else:
                                    if self.stock_non_negative_temp_value[flow_to_stock] + self.name_space[var] * self.sim_specs['dt'] < 0:
                                        self.name_space[var] = self.stock_non_negative_temp_value[flow_to_stock] / self.sim_specs['dt'] *-1 # this outcome is different from Stella, but it is more reasonable. See AwkwardStockFlow.stmx, stock10
                                        self.stock_non_negative_temp_value[flow_to_stock] = np.float64(0)
                                    else:
                                        self.stock_non_negative_temp_value[flow_to_stock] += self.name_space[var] * self.sim_specs['dt']

                    if 'out_from_non_negative_stock' in flow_attributes:
                        out_from_non_negative_stock = flow_attributes['out_from_non_negative_stock']
                        self.logger.debug('    '+f'----considering outflow {var} out from for non-negative stock {out_from_non_negative_stock} whose name_space value is {self.name_space[var]}')
                        
                        # constrain this out_flow
                        self.logger.debug('    '+f'----stock {out_from_non_negative_stock} temp value is {self.stock_non_negative_temp_value[out_from_non_negative_stock]}')
                        if type(self.name_space[var]) is dict:
                            for sub, sub_value in self.name_space[var].items():
                                if self.stock_non_negative_temp_value[out_from_non_negative_stock][sub] - sub_value * self.sim_specs['dt'] < 0:
                                    self.name_space[var][sub] = self.stock_non_negative_temp_value[out_from_non_negative_stock][sub] / self.sim_specs['dt']
                                    self.logger.debug('    '+f'----constraining flow {var} for non-negative stocks {out_from_non_negative_stock} to {self.name_space[var]}')
                                    self.stock_non_negative_temp_value[out_from_non_negative_stock][sub] = np.float64(0)
                                else:
                                    self.stock_non_negative_temp_value[out_from_non_negative_stock][sub] -= sub_value * self.sim_specs['dt']
                        elif var in self.var_dimensions and self.var_dimensions[var] is not None: # The variable is subscripted but all elements uses the same equation
                            for sub in self.stock_non_negative_temp_value[out_from_non_negative_stock]:
                                if self.stock_non_negative_temp_value[out_from_non_negative_stock][sub] - self.name_space[var] * self.sim_specs['dt'] < 0:
                                    self.name_space[var] = self.stock_non_negative_temp_value[out_from_non_negative_stock][sub] / self.sim_specs['dt']
                                    self.logger.debug('    '+f'----constraining flow {var} for non-negative stocks {out_from_non_negative_stock} to {self.name_space[var]}')
                                    self.stock_non_negative_temp_value[out_from_non_negative_stock][sub] = np.float64(0)
                                else:
                                    self.stock_non_negative_temp_value[out_from_non_negative_stock][sub] -= self.name_space[var] * self.sim_specs['dt']
                        else:                        
                            if self.stock_non_negative_temp_value[out_from_non_negative_stock] - self.name_space[var] * self.sim_specs['dt'] < 0:
                                self.name_space[var] = self.stock_non_negative_temp_value[out_from_non_negative_stock] / self.sim_specs['dt']
                                self.logger.debug('    '+f'----constraining flow {var} for non-negative stocks {out_from_non_negative_stock} to {self.name_space[var]}')
                                self.stock_non_negative_temp_value[out_from_non_negative_stock] = np.float64(0)
                            else:
                                self.stock_non_negative_temp_value[out_from_non_negative_stock] -= self.name_space[var] * self.sim_specs['dt']

                    self.logger.debug('    '+'Flow {} = {}'.format(var, self.name_space[var]))
                else:
                    self.logger.debug('    '+'Flow {} is already in name space.'.format(var))
                    # raise Warning('Flow {} is already in name space.'.format(var)) # this should not happen, just in case of any bugs as we switched from dynamic calculation to static calculation
        
        # D: var is an auxiliary
        elif var in self.aux_equations:
            if var not in self.name_space:
                if type(parsed_equation) is dict:
                    for sub, sub_parsed_equation in parsed_equation.items():
                        value = self.solver.calculate_node(var_name=var, parsed_equation=sub_parsed_equation, subscript=sub)
                        if var not in self.name_space:
                            self.name_space[var] = dict()
                        self.name_space[var][sub] = value
                elif var in self.var_dimensions and self.var_dimensions[var] is not None: # The variable is subscripted but all elements uses the same equation
                    for sub in self.dimension_elements[self.var_dimensions[var]]:
                        value = self.solver.calculate_node(var_name=var, parsed_equation=parsed_equation, subscript=sub)
                        if var not in self.name_space:
                            self.name_space[var] = dict()
                        self.name_space[var][sub] = value
                else:
                    value = self.solver.calculate_node(var_name=var, parsed_equation=parsed_equation, subscript=subscript)
                    self.name_space[var] = value
                self.logger.debug('    '+'Aux {} = {}'.format(var, value))
                        
            else:
                pass
        
        else:
            raise Exception("Undefined var: {}".format(var))

    def update_stocks(self):
        for stock, in_out_flows in self.stock_flows.items():
            if stock not in self.conveyors: # coneyors are updated separately
                if stock in self.stock_shadow_values:
                    self.logger.debug('updating stock {} shadow_value is {}'.format(stock, self.stock_shadow_values[stock]))
                else:
                    self.logger.debug('updating stock {} shadow_value not exist, name_space value is {}'.format(stock, self.name_space[stock]))
                
                if len(in_out_flows) != 0:
                    for direction, flows in in_out_flows.items():
                        if direction == 'in':
                            for flow in flows:
                                self.logger.debug('--inflow {} = {}'.format(flow, self.name_space[flow]))
                                if stock not in self.stock_shadow_values:
                                    self.stock_shadow_values[stock] = deepcopy(self.name_space[stock])
                                if type(self.stock_shadow_values[stock]) is dict:
                                    if type(self.name_space[flow]) is dict:
                                        for sub, sub_value in self.name_space[flow].items():
                                            self.stock_shadow_values[stock][sub] += sub_value * self.sim_specs['dt']
                                    else:
                                        for sub in self.stock_shadow_values[stock].keys():
                                            self.stock_shadow_values[stock][sub] += self.name_space[flow] * self.sim_specs['dt']
                                else:
                                    self.stock_shadow_values[stock] += self.name_space[flow] * self.sim_specs['dt']
                                self.logger.debug('----stock_shadow_value {} bcomes {}'.format(stock, self.stock_shadow_values[stock]))
                        elif direction == 'out':
                            for flow in flows:
                                self.logger.debug('--outflow {} = {}'.format(flow, self.name_space[flow]))
                                if stock not in self.stock_shadow_values:
                                    self.stock_shadow_values[stock] = deepcopy(self.name_space[stock])
                                if type(self.stock_shadow_values[stock]) is dict:
                                    if type(self.name_space[flow]) is dict:
                                        for sub, sub_value in self.name_space[flow].items():
                                            self.stock_shadow_values[stock][sub] -= sub_value * self.sim_specs['dt']
                                    else:
                                        for sub in self.stock_shadow_values[stock].keys():
                                            self.stock_shadow_values[stock][sub] -= self.name_space[flow] * self.sim_specs['dt']
                                else:
                                    self.stock_shadow_values[stock] -= self.name_space[flow] * self.sim_specs['dt']
                                self.logger.debug('----stock_shadow_value {} becomes {}'.format(stock, self.stock_shadow_values[stock]))
                else: # there are obsolete stocks that are not connected to any flows
                    self.logger.debug('stock {} is not connected to any flows'.format(stock))
                    self.stock_shadow_values[stock] = deepcopy(self.name_space[stock])
                    self.logger.debug('stock_shadow_value {} remains {}'.format(stock, self.stock_shadow_values[stock]))
            else:
                pass # conveyors are updated separately
    
    def update_conveyors(self):
        for conveyor_name, conveyor in self.conveyors.items(): # Stock is a Conveyor
            self.logger.debug('updating conveyor {}'.format(conveyor_name))
            total_flow_effect = 0
            connected_flows = self.stock_flows[conveyor_name]
            for direction, flows in connected_flows.items():
                if direction == 'in':
                    for flow in flows:
                        total_flow_effect += self.name_space[flow]

            # in
            conveyor['conveyor'].inflow(total_flow_effect * self.sim_specs['dt'])
            self.stock_shadow_values[conveyor_name] = conveyor['conveyor'].level()

    def simulate(self, time=None, dt=None):
        self.logger.debug('Simulation started with specs: {}'.format(self.sim_specs))
        self.logger.debug('Equations: {}'.format(self.stock_equations | self.flow_equations | self.aux_equations))
        
        if time is None:
            time = self.sim_specs['simulation_time']
        if dt is None:
            dt = self.sim_specs['dt']
        iterations = int(time/dt)

        if self.state in ['simulated', 'changed']:
            if self.state == 'changed':
                self.logger.debug('Equation changed after last simulation, re-parsing.')
                self.parse() # set state to 'parsed'
                self.generate_ordered_vars()

            self.logger.debug("")
            self.logger.debug("*** Resuming ***")
            self.logger.debug("")
            # self.name_space.clear()
            # # use last time slice as the initial values for the next simulation, do not do initialization again
            # # self.sim_specs['current_time'] -= self.sim_specs['dt'] # go back to the last time step
            # for stock in (self.stocks | self.conveyors):
            #     last_value = self.time_slice[self.sim_specs['current_time'] - self.sim_specs['dt']][stock]
            #     self.name_space[stock] = last_value
            
            # self.name_space['TIME'] = self.sim_specs['current_time']
            # self.name_space['DT'] = self.sim_specs['dt']

            self.logger.debug('Continuing simulation from time {} for {} iteration'.format(self.sim_specs['current_time'], iterations))
        
        elif self.state == 'loaded':
            # parse equations and order execution (compile)
            self.parse() # set state to 'parsed'
            self.generate_ordered_vars()
            # Initialization Phase
            self.logger.debug("")
            self.logger.debug("*** Initialization ***")
            self.logger.debug("")
            self.logger.debug("self.ordered_vars_init {}".format(self.ordered_vars_init))

            # Initialize self.stock_non_negative_temp_value
            for stock, is_non_negative in self.stock_non_negative.items():
                if is_non_negative:
                    self.stock_non_negative_temp_value[stock] = None

            for var in self.ordered_vars_init:
                self.calculate_variable(var=var, dg=self.dg_init)
        
            # Since it's just loaded, we need 2 iterations (if we were to simulate 1 DT)
            # The 1st iteration is to calculate the flows (and auxiliaries) based on the initialilized stocks (1st row of outcome) and update the stocks (2nd row of outcome)
            # The 2nd iteration is to calculate the flows (and auxiliaries) based on the updated stocks (2nd row of outcome) and update the stocks.
            # The updated stocks (in name_space) should be part of the 3rd row of outcome, but they are not saved (bus still in name_space) as we only simulate 1 DT.
            iterations += 1 

        # Iteration Phase
        self.logger.debug("")
        self.logger.debug("*** Iteration ***")
        self.logger.debug("")
        self.logger.debug("self.ordered_vars_iter {}".format(self.ordered_vars_iter))
        self.logger.debug("Current name_space: {}".format(self.name_space))

        # self.current_iteration = 0

        for s in range(iterations):
            self.logger.debug('--iteration {} start, current time {}--'.format(s, self.sim_specs['current_time']))
            # self.logger.debug('--time {} --'.format(self.sim_specs['current_time']))
            # self.logger.debug('\n--step {} start--\n'.format(s), self.name_space)
            
            # Iter step 1: calculate flows and auxiliaries they depend on
            for var in self.ordered_vars_iter:
                self.calculate_variable(var=var, dg=self.dg_iter)

            # Iter step 2: update stocks using flows and conveyors
            self.update_stocks() # update stock shadow values using flows
            self.update_conveyors() # update stock shadow values as well as conveyors 

            # Snapshot current name space
            current_snapshot = deepcopy(self.name_space)
            current_snapshot[self.sim_specs['time_units']] = current_snapshot['TIME']
            current_snapshot.pop('TIME')
            
            self.time_slice[self.sim_specs['current_time']] = current_snapshot

            self.logger.debug('--step {} finished--'.format(s)) 
            self.logger.debug('name_space {}'.format(self.name_space))
            self.logger.debug('shadow_val {}'.format(self.stock_shadow_values))

            
            # Iter step 3: update simulation time
            self.sim_specs['current_time'] += dt
            # self.current_iteration += 1

            # prepare name_space for next step
            self.logger.debug('--- prepared name_space for next step ---')
            self.logger.debug('clear name space')
            self.name_space.clear()
            self.logger.debug(f'name space: {self.name_space}')

            self.logger.debug('populate name_space using shadow values')
            # Here this shadow value is used directly as the stock value for the next time step
            # This is OK if the model equations are not changed 'dynamically' during the simulation
            # However if flow equations are changed, either in themselves or in their dependencies,
            # then the shadow value will be incorrect.
            for stock, stock_value in self.stock_shadow_values.items():
                self.name_space[stock] = deepcopy(stock_value)
            self.logger.debug(f'name space: {self.name_space}')
            
            self.logger.debug('clear shadow value')
            self.stock_shadow_values.clear()
            self.logger.debug(f'shadow value: {self.stock_shadow_values}')

            self.logger.debug('populate non-negative temp value')
            for k, v in self.stock_non_negative_temp_value.items():
                self.stock_non_negative_temp_value[k] = deepcopy(self.name_space[k])
            self.logger.debug('non-negative temp value: {}'.format(self.stock_non_negative_temp_value))
            
            self.name_space['TIME'] = self.sim_specs['current_time']
            self.name_space['DT'] = self.sim_specs['dt']

            self.logger.debug('--- end of preparation ---')

        self.state = 'simulated'

    # def trace_error(self, var_with_error, sub=None):
    #     self.debug_level_trace_error += 1

    #     self.logger.debug(self.debug_level_trace_error*'    '+'Tracing error on {} ...'.format(var_with_error))
    #     self.logger.debug(self.debug_level_trace_error*'    '+'asdm value    :', self.name_space[var_with_error])
    #     self.logger.debug(self.debug_level_trace_error*'    '+'Expected value:', self.df_debug_against.iloc[self.current_iteration][self.var_name_to_csv_entry(var_with_error)])
        
    #     if sub is not None:
    #         parsed_equation = (self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)[var_with_error][sub]
    #     else:
    #         parsed_equation = (self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)[var_with_error]
        
    #     leafs = [x for x in parsed_equation.nodes() if parsed_equation.out_degree(x)==0]
    #     self.logger.debug(self.debug_level_trace_error*'    '+'Dependencies of {}:'.format(var_with_error))
        
    #     for leaf in leafs:
    #         # self.logger.debug(self.debug_level_trace_error*'    '+parsed_equation.nodes[leaf])
    #         if parsed_equation.nodes[leaf]['operator'][0] in ['EQUALS', 'SPAREN']:
    #             operands = parsed_equation.nodes[leaf]['operands']
    #             if operands[0][0] == 'NUMBER':
    #                 pass
    #             elif operands[0][0] == 'NAME': # this refers to a variable like 'a'
    #                 var_dependent = operands[0][1]
    #                 self.logger.debug(self.debug_level_trace_error*'    '+'-- Dependent:', var_dependent)
    #                 self.logger.debug(self.debug_level_trace_error*'    '+'   asdm value    :', self.name_space[var_dependent])
    #                 self.logger.debug(self.debug_level_trace_error*'    '+'   Expected value:', self.df_debug_against.iloc[self.current_iteration][self.var_name_to_csv_entry(var_dependent)])
        
    #             elif operands[0][0] == 'FUNC': # this refers to a subscripted variable like 'a[ele1]'
    #                 # need to find that 'SPAREN' node
    #                 var_dependent_node_id = operands[0][2]
    #                 var_dependent = parsed_equation.nodes[var_dependent_node_id]['operands'][0][1]
    #                 self.logger.debug(self.debug_level_trace_error*'    '+'-- Dependent:', var_dependent)
    #                 self.logger.debug(self.debug_level_trace_error*'    '+'   asdm value    :', self.name_space[var_dependent])
    #                 self.logger.debug(self.debug_level_trace_error*'    '+'   Expected value:', self.df_debug_against.iloc[self.current_iteration][self.var_name_to_csv_entry(var_dependent)])
        
    #     if var_with_error in self.flow_stocks:
    #         connected_stocks = self.flow_stocks[var_with_error]
    #         for direction, connected_stock in connected_stocks.items():
    #             self.logger.debug(self.debug_level_trace_error*'    '+'-- Connected stock: {:<4} {}'.format(direction, connected_stock))
    #             self.logger.debug(self.debug_level_trace_error*'    '+'   asdm value    :', self.name_space[connected_stock])
    #             self.logger.debug(self.debug_level_trace_error*'    '+'   Expected value:', self.df_debug_against.iloc[self.current_iteration][self.var_name_to_csv_entry(connected_stock)])
        
    #     self.logger.debug()
    #     self.debug_level_trace_error -= 1

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
            stock.initialized = False

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
            var_dimension=self.var_dimension,
            name_space=self.name_space,
            graph_functions=self.graph_functions,
            )

        self.custom_functions = dict()

        self.state = 'loaded'

    def summary(self):
        print('\nSummary:\n')
        # print('------------- Definitions -------------')
        # pprint(self.stock_equations | self.flow_equations | self.aux_equations)
        # print('')
        print('-------------  Sim specs  -------------')
        pprint(self.sim_specs)
        print('')
        print('-------------  Runtime    -------------')
        pprint(self.name_space)
        print('-------------  State      -------------')
        print(self.state)
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
                try:
                    result.append(slice[name][subscript])
                except KeyError as e:
                    print('Subscript {} not found for variable {}; available subscripts: {}'.format(subscript, name, list(slice[name].keys())))
                    raise e
            return result
            
    def export_simulation_result(self, flatten=False, format='dict', to_csv=False):
        self.full_result = dict()
        self.full_result_df = None
        
        # generate full_result
        for time, slice in self.time_slice.items():
            for var, value in slice.items():
                if var == 'DT':
                    continue
                if type(value) is dict:
                    for sub, subvalue in value.items():
                        try:
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

    def create_variable_dependency_graph(self, var, graph=None):
        if graph is None:
            graph = nx.DiGraph()

        all_equations = (self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)
        if var in self.env_variables: # like 'TIME'
            return graph
        parsed_equation = all_equations[var]

        def get_dependent_variables(parsed_equation):
            dependent_variables = list()
            if type(parsed_equation) is not dict:
                leafs = [x for x in parsed_equation.nodes() if parsed_equation.out_degree(x)==0]
                for leaf in leafs:
                    if parsed_equation.nodes[leaf]['operator'] in ['EQUALS', 'SPAREN']:
                        dependent_name = parsed_equation.nodes[leaf]['value']
                        if dependent_name in self.stock_equations.keys() | self.flow_equations.keys() | self.aux_equations.keys(): # Dimension names are not variables, should be filtered out
                            dependent_variables.append(dependent_name)
            else:
                for _, sub_eqn in parsed_equation.items():
                    leafs = [x for x in sub_eqn.nodes() if sub_eqn.out_degree(x)==0]
                    for leaf in leafs:
                        if sub_eqn.nodes[leaf]['operator'] in ['EQUALS', 'SPAREN']:
                            dependent_name = sub_eqn.nodes[leaf]['value']
                            if dependent_name not in dependent_variables: # remove duplicates
                                if dependent_name in self.stock_equations.keys() | self.flow_equations.keys() | self.aux_equations.keys(): # Dimension names are not variables, should be filtered out
                                    dependent_variables.append(dependent_name)
            return dependent_variables
        
        if type(parsed_equation) is list: # this variable might be a conveyor
            if var in self.conveyors:
                dep_graph_len = get_dependent_variables(parsed_equation[0])
                dep_graph_val = get_dependent_variables(parsed_equation[1])
                # combine the two lists without duplicates
                dependent_variables = list(set(dep_graph_len + dep_graph_val))
            else:
                raise Exception("Non-conveyor variable with parsed equation as list: {}".format(var))
        else:
            dependent_variables = get_dependent_variables(parsed_equation)

        if len(dependent_variables) == 0:
            graph.add_node(var)
            return graph
        else:
            for dependent_var in dependent_variables:
                graph.add_edge(dependent_var, var)
                self.create_variable_dependency_graph(dependent_var, graph)
            return graph

    def generate_cld(self, vars=None, show=False, loop=True):
        # Make sure the model equations are parsed
        if self.state == 'loaded':
            self.parse()

        if vars is None:
            vars = list(self.flow_equations.keys())
        elif type(vars) is str:
            vars = [vars]
        
        dg = nx.DiGraph()

        for var in vars:
            dg_var = self.create_variable_dependency_graph(var)
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

    def generate_full_dependent_graph(self, show=False):
        stocks = list(self.stock_equations.keys())
        flows = list(self.flow_equations.keys())

        # Initialization phase
        dg_init = nx.DiGraph()
        for stock in stocks:
            dg_stock = self.create_variable_dependency_graph(stock)
            dg_init = nx.compose(dg_init, dg_stock)
        
        # check each non-negative stock for dependencies of inflow and outflow and add to dg_init
        for stock, in_out_flows in self.stock_flows.items():
            if self.stock_non_negative[stock] is True:
                self.logger.debug('GEN 0.init for non negative stock %s', stock)
                if 'out' in in_out_flows:
                    out_flows = in_out_flows['out']

                    for out_flow in out_flows:
                        if out_flow in dg_init:
                            self.logger.debug('GEN --1.init for outflow %s', out_flow)
                            # if stock explicitly depends on outflow for initiliazation, we cannot let outflow be constrained by stock in the initialization phase
                            if nx.has_path(dg_init, out_flow, stock):
                                pass
                            else: # out_flow 
                                nx.set_node_attributes(dg_init, {out_flow: {'out_from_non_negative_stock': stock}}) # this attribute triggers the constrains in runtime
                        else:
                            pass

                    if 'in' in in_out_flows:
                        in_flows = in_out_flows['in']
                        # for each inflow, we need to check if it is dependent on (affected by) any outflow; if yes, we exclude it from outflow constraining.
                        in_flow_sanities = {}

                        for in_flow in in_flows:
                            if in_flow in dg_init:
                                self.logger.debug('GEN ----2.init for inflow %s', in_flow)
                                in_flow_sanities[in_flow] = True
                                for out_flow in out_flows:
                                    if nx.has_path(dg_iter, out_flow, in_flow):
                                        in_flow_sanities[in_flow] = False
                                        # dg_init
                                        if in_flow in dg_init:
                                            nx.set_node_attributes(dg_init, {in_flow: {'considered_for_non_negative_stock': False}}) # this attribute excludes the inflow from 'how much can flow out'
                                    if not in_flow_sanities[in_flow]:
                                        break
                                if in_flow_sanities[in_flow]:
                                    # dg_init
                                    if in_flow in dg_init:
                                        nx.set_node_attributes(dg_init, {in_flow: {'considered_for_non_negative_stock': True}}) # this attribute includes the inflow in 'how much can flow out'
                            else:
                                pass

                        # for inflows without sanity, we need to make them dependent on all outflows, so that they are only calculated after constraining the outflows
                        for in_flow, sanity in in_flow_sanities.items():
                            if not sanity:
                                for out_flow in out_flows:
                                    # dg_init
                                    if (out_flow, in_flow) not in dg_init.edges: # avoid overwriting
                                        dg_init.add_edge(out_flow, in_flow)
                                        self.logger.debug('GEN ------3.init inflow %s implicitly depends on %s', in_flow, out_flow)

                            else: # for inflow with sanity, we need to make all outflows dependent on it, so that they are calculated before constraining the outflows
                                for out_flow in out_flows:
                                    # dg_init
                                    if (in_flow, out_flow) not in dg_init.edges: # avoid overwriting
                                        dg_init.add_edge(in_flow, out_flow)
                                        self.logger.debug('GEN ------4.init outflow %s implicitly depends on %s', out_flow, in_flow)
                    
                    else: # no inflow, just determine the prioritisation of outflows
                        pass
                
                    # set output priorities
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
                    
                    priority_level = 1
                    for out_flow in out_flows:
                        if out_flow in dg_init:
                            nx.set_node_attributes(dg_init, {out_flow: {'priority': priority_level}})

                    self.stock_non_negative_out_flows[stock] = out_flows
                
                else: # no outflows
                    if 'in' in in_out_flows: # no outflows, just inflows
                        self.logger.debug('GEN --5.init no outflow')
                        in_flows = in_out_flows['in']
                        for in_flow in in_flows:
                            if in_flow in dg_init:
                                nx.set_node_attributes(dg_iter, {in_flow: {'considered_for_non_negative_stock': True}}) # this attribute includes the inflow in 'how much can flow out'
                                self.logger.debug("GEN --6.init consider inflow %s", in_flow)

        # Conveyor: add dependency of leakflow on the conveyor
        for conveyor_name, conveyor in self.conveyors.items():
            leakflows=conveyor['leakflow']
            for leakflow in leakflows:
                # the 'value' (not leak_fraction) of leakflow depends on the conveyor
                dg_init.add_edge(conveyor_name, leakflow) 

                # the conveyor depends on the leak_fraction 
                dg_leakflow = self.create_variable_dependency_graph(leakflow)
                dg_leak_fraction = deepcopy(dg_leakflow)
                # replace leakflow with conveyor in the graph
                dg_leak_fraction.remove_node(leakflow)
                dg_leak_fraction.add_node(conveyor_name)
                for pred in dg_leakflow.predecessors(leakflow):
                    dg_leak_fraction.add_edge(pred, conveyor_name)
                
                dg_init = nx.compose(dg_init, dg_leak_fraction)

        # Iteration phase
        dg_iter = nx.DiGraph()
        for flow in flows:
            dg_flow = self.create_variable_dependency_graph(flow)
            dg_iter = nx.compose(dg_iter, dg_flow)

        # add obsolete flows and auxiliaries to the dg_iter
        for var in (self.flow_equations | self.aux_equations):
            if var not in dg_iter.nodes:
                dg_obsolete = self.create_variable_dependency_graph(var)
                dg_iter = nx.compose(dg_iter, dg_obsolete)

        # check each non-negative stock for dependencies of inflow and outflow and add to dg_iter
        for stock, in_out_flows in self.stock_flows.items():
            if self.stock_non_negative[stock] is True:
                self.logger.debug('GEN 0.iter for non negative stock %s', stock)
                if 'out' in in_out_flows:
                    out_flows = in_out_flows['out']

                    for out_flow in out_flows:
                        self.logger.debug('GEN --1.iter for outflow %s', out_flow)
                        nx.set_node_attributes(dg_iter, {out_flow: {'out_from_non_negative_stock': stock}}) # this attribute triggers the constrains in runtime

                    if 'in' in in_out_flows:
                        in_flows = in_out_flows['in']
                        # for each inflow, we need to check if it is dependent on (affected by) any outflow; if yes, we exclude it from outflow constraining.
                        in_flow_sanities = {}

                        for in_flow in in_flows:
                            self.logger.debug('GEN ----2.iter for inflow %s', in_flow)
                            in_flow_sanities[in_flow] = True
                            for out_flow in out_flows:
                                if nx.has_path(dg_iter, out_flow, in_flow):
                                    in_flow_sanities[in_flow] = False
                                    nx.set_node_attributes(dg_iter, {in_flow: {'considered_for_non_negative_stock': False}}) # this attribute excludes the inflow from 'how much can flow out'
                                if not in_flow_sanities[in_flow]:
                                    break
                            if in_flow_sanities[in_flow]:
                                nx.set_node_attributes(dg_iter, {in_flow: {'considered_for_non_negative_stock': True}}) # this attribute includes the inflow in 'how much can flow out'
                        
                        # for inflows without sanity, we need to make them dependent on all outflows, so that they are only calculated after constraining the outflows
                        for in_flow, sanity in in_flow_sanities.items():
                            if not sanity:
                                for out_flow in out_flows:
                                    if (out_flow, in_flow) not in dg_iter.edges: # avoid overwriting
                                        dg_iter.add_edge(out_flow, in_flow)
                                        self.logger.debug('GEN ------3.iter inflow %s implicitly depends on %s', in_flow, out_flow)

                            else: # for inflow with sanity, we need to make all outflows dependent on it, so that they are calculated before constraining the outflows
                                for out_flow in out_flows:
                                    if (in_flow, out_flow) not in dg_iter.edges: # avoid overwriting
                                        dg_iter.add_edge(in_flow, out_flow)
                                        self.logger.debug('GEN ------4.iter outflow %s implicitly depends on %s', out_flow, in_flow)

                    
                    else: # no inflow, just determine the prioritisation of outflows
                        pass
                
                    # set output priorities
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
                    
                    priority_level = 1
                    for out_flow in out_flows:
                        priority_level += 1

                    self.stock_non_negative_out_flows[stock] = out_flows
                
                else: # no outflows
                    if 'in' in in_out_flows: # no outflows, just inflows
                        self.logger.debug('GEN --5.iter no outflow')
                        in_flows = in_out_flows['in']
                        for in_flow in in_flows:
                            nx.set_node_attributes(dg_iter, {in_flow: {'considered_for_non_negative_stock': True}}) # this attribute includes the inflow in 'how much can flow out'
                            self.logger.debug("GEN --6.iter consider inflow %s", in_flow)

        self.logger.debug('GEN Dependent graph for init:')
        self.logger.debug('GEN --nodes %s', dg_init.nodes(data=True))
        self.logger.debug('GEN --edges %s', dg_init.edges(data=True))
        self.logger.debug('GEN Dependent graph for iter:')
        self.logger.debug('GEN --nodes %s', dg_iter.nodes(data=True))
        self.logger.debug('GEN --edges %s', dg_iter.edges(data=True))

        if not show:
            return (dg_init, dg_iter)
        else:
            if show == 'init':
                dg = dg_init
            elif show == 'iter':
                dg = dg_iter
            else:
                raise Exception('Invalid show parameter {}. Use "init" or "iter"'.format(show))

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
            return (dg_init, dg_iter)
    
    def generate_ordered_vars(self):
        self.dg_init, self.dg_iter = self.generate_full_dependent_graph()
        self.ordered_vars_init = list(nx.topological_sort(self.dg_init))
        self.ordered_vars_iter = list(nx.topological_sort(self.dg_iter))