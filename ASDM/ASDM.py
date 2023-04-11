import networkx as nx
import numpy as np
import re
from itertools import product
from pprint import pprint
from scipy import stats
from copy import deepcopy


class Parser(object):
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
        }

        self.functions = { # use lookahead (?=\() to ensure only match INIT( not INITIAL
            'MIN': r'MIN(?=\()',
            'MAX': r'MAX(?=\()',
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
        }

        self.names = {
            'ABSOLUTENAME': r'"[\s\S]*?"',
            'NAME': r'[a-zA-Z0-9_\?]*',
        }

        self.node_id = 0

        self.patterns_sub_var = {
            # token: to put as placeholder in 'items'
            # operator and operands: to put into parsed_equation 

            ### Subscripted Variables ###

            'NAME__LSPAREN__DOT+__RSPAREN':{
                'token':['FUNC', 'SPAREN'],
                'operator':['SPAREN'],
                'operand':['NUMBER', 'NAME']
            },
        }
        self.patterns_num = {
            ### Numbers ###

            'NUMBER':{
                'token':['FUNC', 'IS'],
                'operator':['IS'],
                'operand':['NUMBER']
            },
        }
        self.patterns_var = {
            ### User-defined Variables ###

            'NAME': {
                'token':['FUNC', 'EQUALS'],
                'operator':['EQUALS'],
                'operand':['NAME']
                },
        }
        self.patterns_custom_func = {}
        self.patterns_brackets = {
            ### Brackets ###

            'LPAREN__FUNC__RPAREN':{
                'token':['FUNC', 'PAREN'],
                'operator':['PAREN'],
                'operand':['FUNC']
            },
        }
        ### Arithmetics ###
        self.patterns_arithmetic_1 = {
            'FUNC__TIMES__FUNC': {
                'token':['FUNC', 'TIMES'],
                'operator':['TIMES'],
                'operand':['FUNC']
                },
            'FUNC__DIVIDE__FUNC': {
                'token':['FUNC', 'DIVIDE'],
                'operator':['DIVIDE'],
                'operand':['FUNC']
                },
            'FUNC__FLOORDIVIDE__FUNC': {
                'token':['FUNC', 'FLOORDIVIDE'],
                'operator':['FLOORDIVIDE'],
                'operand':['FUNC']
                },
        }
        self.patterns_arithmetic_2 = {
            'FUNC__MOD__FUNC':{
                'token':['FUNC', 'MOD'],
                'operator':['MOD'],
                'operand':['FUNC']
            },
        }
        self.patterns_arithmetic_3 = {

            'FUNC__PLUS__FUNC': {
                'token':['FUNC', 'PLUS'],
                'operator':['PLUS'],
                'operand':['FUNC']
                },
            'FUNC__MINUS__FUNC': {
                'token':['FUNC', 'MINUS'],
                'operator':['MINUS'],
                'operand':['FUNC']
                },
        }
        self.patterns_built_in_func = {
            ### Built in functions ###
            'MIN__LPAREN__FUNC__COMMA__FUNC__RPAREN':{
                'token':['FUNC', 'MIN'],
                'operator':['MIN'],
                'operand':['FUNC']
            },
            'MAX__LPAREN__FUNC__COMMA__FUNC__RPAREN':{
                'token':['FUNC', 'MAX'],
                'operator':['MAX'],
                'operand':['FUNC']
            },
            'RBINOM__LPAREN__FUNC__COMMA__FUNC__RPAREN':{
                'token':['FUNC', 'RBINOM'],
                'operator':['RBINOM'],
                'operand':['FUNC']
            },
            'INIT__LPAREN__FUNC__RPAREN':{
                'token':['FUNC', 'INIT'],
                'operator':['INIT'],
                'operand':['FUNC']
            },
            'DELAY__LPAREN__FUNC__COMMA__FUNC__RPAREN':{
                'token':['FUNC', 'DELAY'],
                'operator':['DELAY'],
                'operand':['FUNC']
            },
            'DELAY__LPAREN__FUNC__COMMA__FUNC__COMMA__FUNC__RPAREN':{
                'token':['FUNC', 'DELAY'],
                'operator':['DELAY'],
                'operand':['FUNC']
            },
            'DELAY1__LPAREN__FUNC__COMMA__FUNC__RPAREN':{
                'token':['FUNC', 'DELAY1'],
                'operator':['DELAY1'],
                'operand':['FUNC']
            },
            'DELAY1__LPAREN__FUNC__COMMA__FUNC__COMMA__FUNC__RPAREN':{
                'token':['FUNC', 'DELAY1'],
                'operator':['DELAY1'],
                'operand':['FUNC']
            },
            'DELAY3__LPAREN__FUNC__COMMA__FUNC__RPAREN':{
                'token':['FUNC', 'DELAY3'],
                'operator':['DELAY3'],
                'operand':['FUNC']
            },
            'DELAY3__LPAREN__FUNC__COMMA__FUNC__COMMA__FUNC__RPAREN':{
                'token':['FUNC', 'DELAY3'],
                'operator':['DELAY3'],
                'operand':['FUNC']
            },
            'SMTH1__LPAREN__FUNC__COMMA__FUNC__RPAREN':{
                'token':['FUNC', 'SMTH1'],
                'operator':['SMTH1'],
                'operand':['FUNC']
            },
            'SMTH1__LPAREN__FUNC__COMMA__FUNC__COMMA__FUNC__RPAREN':{
                'token':['FUNC', 'SMTH1'],
                'operator':['SMTH1'],
                'operand':['FUNC']
            },
            'SMTH3__LPAREN__FUNC__COMMA__FUNC__RPAREN':{
                'token':['FUNC', 'SMTH3'],
                'operator':['SMTH3'],
                'operand':['FUNC']
            },
            'SMTH3__LPAREN__FUNC__COMMA__FUNC__COMMA__FUNC__RPAREN':{
                'token':['FUNC', 'SMTH3'],
                'operator':['SMTH3'],
                'operand':['FUNC']
            },
            'HISTORY__LPAREN__FUNC__COMMA__FUNC__RPAREN':{
                'token':['FUNC', 'HISTORY'],
                'operator':['HISTORY'],
                'operand':['FUNC']
            },
            'STEP__LPAREN__FUNC__COMMA__FUNC__RPAREN':{
                'token':['FUNC', 'STEP'],
                'operator':['STEP'],
                'operand':['FUNC']
            },
            'LOOKUP__LPAREN__FUNC__COMMA__FUNC__RPAREN':{
                'token':['FUNC', 'LOOKUP'],
                'operator':['LOOKUP'],
                'operand':['FUNC']
            },
        }
        
        self.patterns_logic = {
            ### Logic ###

            'FUNC__GT__FUNC':{
                'token':['FUNC', 'GT'],
                'operator':['GT'],
                'operand':['FUNC']
            },
            'FUNC__LT__FUNC':{
                'token':['FUNC', 'LT'],
                'operator':['LT'],
                'operand':['FUNC']
            },
            'FUNC__NGT__FUNC':{
                'token':['FUNC', 'NGT'],
                'operator':['NGT'],
                'operand':['FUNC']
            },
            'FUNC__NLT__FUNC':{
                'token':['FUNC', 'NLT'],
                'operator':['NLT'],
                'operand':['FUNC']
            },
            'FUNC__EQS__FUNC':{
                'token':['FUNC', 'EQS'],
                'operator':['EQS'],
                'operand':['FUNC']
            },
            'FUNC__AND__FUNC':{
                'token':['FUNC', 'AND'],
                'operator':['AND'],
                'operand':['FUNC']
            },
            'FUNC__OR__FUNC':{
                'token':['FUNC', 'OR'],
                'operator':['OR'],
                'operand':['FUNC']
            },
            'NOT__FUNC':{
                'token':['FUNC', 'NOT'],
                'operator':['NOT'],
                'operand':['FUNC']
            },
        }
        self.patterns_conditional = {
            ### Conditional ###

            'CONIF__FUNC__CONTHEN__FUNC__CONELSE__FUNC':{
                'token':['FUNC', 'CON'],
                'operator':['CON'],
                'operand':['FUNC']
            }
        }

        self.HEAD = "PARSER"

    def tokenisation(self, s):
        items = []
        # l = len(items)
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
                    item = m[0]
                    if item[0] == "\"" and item[-1] == "\"": # strip quotation marks from matched string
                        item = item[1:-1]
                    if type_name == 'ABSOLUTENAME':
                        type_name = 'NAME'
                    if type_name == 'NUMBER' and item[0] == '-': # fix the -1: negative 1 vs minus 1 problem
                        if not(len(items) == 0 or items[-1] == ['LPAREN', '(']):
                            items.append(['MINUS', '-'])
                            items.append(['NUMBER', item[1:]])
                            s = s[m.span()[1]:].strip()
                        else:
                            items.append([type_name, item])
                            s = s[m.span()[1]:].strip()
                    else:
                        items.append([type_name, item])
                        s = s[m.span()[1]:].strip()
                    break
            # print(items)
            # if len(items) == l:
            #     raise Exception()
            # else:
            #     l = len(items)
        return items

    def parse(self, string, verbose=False, max_iter=100):
        # if type(string) is dict:
        #     parsed_equations = dict()
        #     for k, ks in string.items():
        #         parsed_equations[k] = self.parse(ks)
        #     return parsed_equations
        # else:
        if verbose:
            print(self.HEAD, 'Parse string:', string)
        items = self.tokenisation(string)
        if verbose:
            print(self.HEAD, 'Items:', items)
        graph = nx.DiGraph()
        if len(items) == 1:
            if items[0][0] == 'NAME':
                operator = ['EQUALS']
            elif items[0][0] == 'NUMBER':
                operator = ['IS']
                items[0][1] = float(items[0][1])
            graph.add_node(
                self.node_id, 
                operator=operator,
                operands=items
                )
            graph.add_edge('root', self.node_id)
            return graph
        r = max_iter
        while len(items) > 1 and r > 0:
            if verbose:
                print('\n'+self.HEAD, 'Iter:', r)
            r -= 1 # use this line to put a stop by number of iterations
            if r == 0:
                raise Exception('Parser timeout on String\n  {}, \nItems\n  {}'.format(string, items))
            items_changed = False
            if verbose:
                print(self.HEAD, '---Parsing---','\n')
                print(self.HEAD, 'Items', items)
                print(self.HEAD, 'Nodes', graph.nodes.data(True))
                print(self.HEAD, 'Edges', graph.edges.data(True))

            for pattern_set in [ # the order of the following patterns is IMPORTANT
                self.patterns_sub_var,
                self.patterns_num,
                self.patterns_var,
                self.patterns_custom_func,
                self.patterns_built_in_func,
                self.patterns_brackets,
                self.patterns_arithmetic_1,
                self.patterns_arithmetic_2,
                self.patterns_arithmetic_3,
                self.patterns_logic,
                self.patterns_conditional,
                ]: # loop over all patterns
                for i in range(len(items)): # loop over all positions for this pattern
                    # if verbose:
                        # print(self.HEAD, 'Position:', i)
                    #     print(self.HEAD, 'Checking item:', i, items[i])
                    for pattern, func in pattern_set.items():
                        pattern = pattern.split('__')
                        # if verbose:
                            # print(self.HEAD, "Searching for {} with operator {}".format(pattern, func['operator']))
                        pattern_len = len(pattern)

                        if len(items) - i >= pattern_len:
                            matched = True
                            for j in range(pattern_len): # matching pattern at this position
                                if pattern[j] == items[i+j][0]: # exact match
                                    pass
                                else: 
                                    if pattern[j] == 'DOT+': # fuzzy match
                                        dotplus_matched = False
                                        # print(self.HEAD, 'Matching DOT+')
                                        try:
                                            next_to_match = pattern[j+1] # it is pattern[j+1] that matters
                                            # print(self.HEAD, 'Next to match:', next_to_match)
                                            for k in range(i+j+1, len(items)):
                                                if next_to_match == items[k][0]:
                                                    # print(self.HEAD, 'Found next to match:', next_to_match, 'at', k, items[k])
                                                    pattern_len = k - i + 1
                                                    dotplus_matched = True
                                                    break                 
                                        except IndexError: # 'DOT+' is the last in the pattern
                                            pass
                                        if dotplus_matched:
                                            break
                                    else:
                                        matched = False
                                        break
                            if matched:
                                matched_items = items[i:i+pattern_len]
                                if verbose:
                                    print(self.HEAD, "Found {} at {}".format(pattern, matched_items))
                                operands = []
                                for item in matched_items:
                                    if item[0] in func['operand']:
                                        if item[0] in ['NAME', 'NUMBER']:
                                            if item[0] == 'NUMBER':
                                                # if item is a part of a[1,ele1] then it should remain str
                                                # otherwise it should be converted to float
                                                # print(self.HEAD, 'found a number', item, 'func:', func['operator'])
                                                if func['operator'][0] == 'IS':
                                                    item[1] = float(item[1])
                                        else:
                                            # print(self.HEAD, 'adding edge from {} to {}'.format(self.node_id, item[2]))
                                            graph.add_edge(self.node_id, item[2])
                                        operands.append(item)
                                graph.add_node(
                                    self.node_id, 
                                    operator=func['operator'],
                                    operands=operands
                                    )
                                items = items[0:i] + [func['token'][:] + [self.node_id]] + items[i+pattern_len:]
                                items_changed = True
                                self.node_id += 1
                                break # items has been updated and got a new length, need to start the for loop over again
                    
                    if items_changed:
                        break
                if items_changed:
                    break
        
        # add root node as entry
        graph.add_node('root')
        graph.add_edge('root', items[0][2])

        return graph

    def plot_ast(self, graph):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        labels = {}
        labels_operators = nx.get_node_attributes(graph, 'operator')
        labels_operands = nx.get_node_attributes(graph, 'operands')
        for id, label_operator in labels_operators.items():
            labels[id] = str(id) + '\n' + 'operator:' + str(label_operator) + '\n' + 'operands:' + str(labels_operands[id])
        labels['root'] = 'root'   
        pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
        nx.draw(graph, with_labels=True, labels=labels, node_color='C1', font_size=9, pos=pos)

        plt.show()


class Solver(object):
    def __init__(self, sim_specs=None, dimension_elements=None, name_space=None, graph_functions=None):
        
        self.sim_specs = sim_specs
        self.dimension_elements = dimension_elements
        self.name_space = name_space
        self.graph_functions = graph_functions

        ### Functions ###

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
                elif type(a) is dict and type(b) in [int, float]:
                    o = dict()
                    for k in a:
                        o[k] = a[k] * b
                    return o
                elif type(a) in [int, float] and type(b) is dict:
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
                elif type(a) is dict and type(b) in [int, float]:
                    o = dict()
                    for k in a:
                        o[k] = a[k] / b
                    return o
                elif type(a) in [int, float] and type(b) is dict:
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
                elif type(a) is dict and type(b) in [int, float]:
                    o = dict()
                    for k in a:
                        o[k] = a[k] // b
                    return o
                elif type(a) in [int, float] and type(b) is dict:
                    o = dict()
                    for k in b:
                        o[k] = a // b[k]
                    return o
                else:
                    raise e
                
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
                elif type(a) is dict and type(b) in [int, float]:
                    o = dict()
                    for k in a:
                        o[k] = a[k] % b
                    return o
                elif type(a) in [int, float] and type(b) is dict:
                    o = dict()
                    for k in b:
                        o[k] = a % b[k]
                    return o
                else:
                    raise e

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
            'CON':      con,
            'STEP':     step,
            'MOD':      mod,
            'RBINOM':   rbinom,
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

        self.lookup_functions = [
            'LOOKUP'
        ]

        self.custom_functions = {}
        self.time_expr_register = {}
        
        self.id_level = 0

        self.HEAD = "SOLVER"

    def calculate_node(self, parsed_equation, node_id='root', subscript=None, verbose=False, var_name=''):
        self.id_level += 1

        if verbose:
            print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'processing node {} on subscript {}:'.format(node_id, subscript))
        #     if type(parsed_equation) is dict:
        #         for k, p in parsed_equation.items():
        #             print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', k, p.nodes(data=True))
        #     else:
        #         print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', parsed_equation.nodes(data=True))
        
        if type(parsed_equation) is dict:  
            raise Exception('Parsed equation should not be a dict. var:', var_name)
            # # This section is not active; only kept for potential future reference
            # # if verbose:
            # #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'type of parsed_equation: dict')
            # value = dict()
            # for sub, sub_equaton in parsed_equation.items():
            #     if verbose:
            #         print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'sub', sub)
            #     value[sub] = self.calculate_node(parsed_equation=sub_equaton, node_id='root', subscript=sub, verbose=verbose)
            # if verbose:
            #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v1 Subscripted equation:', value)
        else:
            # if verbose:
            #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'type of parsed_equation: graph')
            if node_id == 'root':
                node_id = list(parsed_equation.successors('root'))[0]
            node = parsed_equation.nodes[node_id]
            if verbose:
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'node:', node_id, node)
            operator = node['operator']
            operands = node['operands']
            if operator[0] == 'IS':
                # if verbose:
                #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'oprt1')
                #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'o1')
                value = np.float64(operands[0][1])
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v2 IS:', value)
            elif operator[0] == 'EQUALS':
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'operator v3', operator)
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'operands v3', operands)
                if operands[0][0] == 'NAME':
                    if subscript:
                        value = self.name_space[operands[0][1]]
                        if type(value) is dict:
                            value = value[subscript]
                            if verbose:
                                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v3.1.1', value)
                        else:
                            if verbose:
                                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v3.1.2', value)
                    else:
                        value = self.name_space[operands[0][1]]
                        if verbose:
                            print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v3.1.3', value, operands)
                    if verbose:
                        print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v3.1 Name:', value)
                elif operands[0][0] == 'FUNC':
                    if verbose:
                        print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'o3')
                    value = self.calculate_node(parsed_equation=parsed_equation, node_id=node_id, subscript=subscript, verbose=verbose)
                    if verbose:
                        print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v3.2 Func:', value)
                else:
                    if verbose:
                        print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'o4')
                    raise Exception()
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v3 Equals:', value)
            
            elif operator[0] == 'SPAREN': # TODO this part is too dynamic, therefore can be slow. Need to resolve this when compiling.
                var_name = operands[0][1]
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a1', var_name, subscript)
                if len(operands) == 1: # only var_name; no subscript is specified
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
                else: # there are explicitly specified subscripts in oprands
                    # if verbose:
                    #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a1.2')
                    if subscript: # subscript is explicitly specified; like a[Element_1] or a[Dimension_1]
                        if verbose:
                            print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a1.2.0', subscript, self.dimension_elements)
                        subscript_from_operands = list()
                        operands_containing_subscript = operands[1:]
                        for i in range(len(operands_containing_subscript)):
                            if operands_containing_subscript[i][1] in self.dimension_elements.keys(): # it's sth like Dimension_1
                                # if verbose:
                                #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a1.2.0.1')
                                subscript_from_operands.append(subscript[i]) # take the element from arg subscript in the same position to replace Dimension_1
                            else: # it's sth like Element_1
                                # if verbose:
                                #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a1.2.0.2')
                                subscript_from_operands.append(operands_containing_subscript[i][1]) # add to list directly
                        subscript_from_operands = tuple(subscript_from_operands)
                        # if verbose:
                        #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a1.2.1', subscript_from_operands)
                        value = self.name_space[var_name][subscript_from_operands] # try if subscript is Element_1

                    else: # subscript is not explicitly specified
                        # if verbose:
                        #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'a1.2.2')
                        # like a[Dimension1] -> Which element of Dimension1 to use, 
                        # depends on the other variable.

                        subscript = tuple(operand[1] for operand in operands[1:]) # use tuple to make it hashable
                        if verbose:
                            print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'subscript of interest', subscript)
                        value = self.name_space[var_name][subscript]
                    if verbose:
                        print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v4.2 Sparen with sub:', value)
            
            elif operator[0] == 'PAREN':
                value = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[0][2], verbose=verbose)
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v6 Paren:', value)

            elif operator[0] in self.built_in_functions.keys(): # plus, minus, con, etc.
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'operator v7 Built-in operator:', operator, operands)
                func_name = operator[0]
                function = self.built_in_functions[func_name]
                oprds = []
                for operand in operands:
                    if verbose:
                        print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'oprd', operand)
                    v = self.calculate_node(parsed_equation=parsed_equation, node_id=operand[2], subscript=subscript, verbose=verbose)
                    if verbose:
                        print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'value', v)
                    oprds.append(v)
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'oprds', oprds)
                value = function(*oprds)
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v7 Built-in operation:', value)
            
            elif operator[0] in self.custom_functions.keys(): # graph functions
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'custom func operator', operator)
                func_name = operator[0]
                function = self.custom_functions[func_name]
                oprds = []
                for operand in operands:
                    if verbose:
                        print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'oprd', operand)
                    v = self.calculate_node(parsed_equation=parsed_equation, node_id=operand[2], verbose=verbose)
                    if verbose:
                        print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'value', v)
                    oprds.append(v)
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'oprds', oprds)
                value = function(*oprds)
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v8 GraphFunc:', value)

            elif operator[0] in self.time_related_functions: # init, delay, etc
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'time-related func. operator:', operator, 'operands:', operands)
                func_name = operator[0]
                if func_name == 'INIT':
                    if tuple(operands[0]) in self.time_expr_register.keys():
                        value = self.time_expr_register[tuple(operands[0])]
                    else:
                        value = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[0][2], verbose=verbose)
                        self.time_expr_register[tuple(operands[0])] = value
                elif func_name == 'DELAY':
                    # expr value
                    expr_value = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[0][2], verbose=verbose)
                    if tuple(operands[0]) in self.time_expr_register.keys():
                        self.time_expr_register[tuple(operands[0])].append(expr_value)
                    else:
                        self.time_expr_register[tuple(operands[0])] = [expr_value]
                    
                    # init value
                    if len(operands) == 2: # there's no initial value specified -> use the delyed expr's initial value
                        init_value = self.time_expr_register[tuple(operands[0])][0]
                    elif len(operands) == 3: # there's an initial value specified
                        init_value = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[2][2], verbose=verbose)
                    else:
                        raise Exception("Invalid initial value for DELAY in operands {}".format(operands))

                    # delay time
                    delay_time = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[1][2], verbose=verbose)
                    if delay_time > (self.sim_specs['current_time'] - self.sim_specs['initial_time']): # (- initial_time) because simulation might not start from time 0
                        value = init_value
                    else:
                        delay_steps = delay_time / self.sim_specs['dt']
                        value = self.time_expr_register[tuple(operands[0])][-int(delay_steps+1)]
                elif func_name == 'DELAY1':
                    # args values
                    order = 1
                    expr_value = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[0][2], verbose=verbose)
                    delay_time = self.calculate_node(parsed_equation=parsed_equation, node_id= operands[1][2], verbose=verbose)

                    if len(operands) == 3:
                        init_value = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[2][2], verbose=verbose)
                    elif len(operands) == 2:
                        init_value = expr_value
                    else:
                        raise Exception('Invalid number of args for DELAY1.')
                    
                    # register
                    if tuple(operands[0]) not in self.time_expr_register.keys():
                        self.time_expr_register[tuple(operands[0])] = list()
                        for i in range(order):
                            self.time_expr_register[tuple(operands[0])].append(delay_time/order*init_value)
                    # outflows
                    outflows = list()
                    for i in range(order):
                        outflows.append(self.time_expr_register[tuple(operands[0])][i]/(delay_time/order) * self.sim_specs['dt'])
                        self.time_expr_register[tuple(operands[0])][i] -= outflows[i]
                    # inflows
                    self.time_expr_register[tuple(operands[0])][0] += expr_value * self.sim_specs['dt']
                    for i in range(1, order):
                        self.time_expr_register[tuple(operands[0])][i] += outflows[i-1]

                    return outflows[-1] / self.sim_specs['dt']

                elif func_name == 'DELAY3':
                    # arg values
                    order = 3
                    expr_value = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[0][2], verbose=verbose)
                    delay_time = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[1][2], verbose=verbose)
                    if len(operands) == 3:
                        init_value = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[2][2], verbose=verbose)
                    elif len(operands) == 2:
                        init_value = expr_value
                    else:
                        raise Exception('Invalid number of args for SMTH3.')
                    
                    # register
                    if tuple(operands[0]) not in self.time_expr_register.keys():
                        self.time_expr_register[tuple(operands[0])] = list()
                        for i in range(order):
                            self.time_expr_register[tuple(operands[0])].append(delay_time/order*init_value)
                    # outflows
                    outflows = list()
                    for i in range(order):
                        outflows.append(self.time_expr_register[tuple(operands[0])][i]/(delay_time/order) * self.sim_specs['dt'])
                        self.time_expr_register[tuple(operands[0])][i] -= outflows[i]
                    # inflows
                    self.time_expr_register[tuple(operands[0])][0] += expr_value * self.sim_specs['dt']
                    for i in range(1, order):
                        self.time_expr_register[tuple(operands[0])][i] += outflows[i-1]

                    return outflows[-1] / self.sim_specs['dt']

                elif func_name == 'HISTORY':
                    # expr value
                    expr_value = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[0][2], verbose=verbose)
                    if tuple(operands[0]) in self.time_expr_register.keys():
                        self.time_expr_register[tuple(operands[0])].append(expr_value)
                    else:
                        self.time_expr_register[tuple(operands[0])] = [expr_value]
                    
                    # historical time
                    historical_time = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[1][2], verbose=verbose)
                    if historical_time > self.sim_specs['current_time']:
                        value = 0
                    else:
                        historical_steps = (historical_time - self.sim_specs['initial_time']) / self.sim_specs['dt']
                        value = self.time_expr_register[tuple(operands[0])][int(historical_steps)]
                elif func_name == 'SMTH1':
                    # arg values
                    order = 1
                    expr_value = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[0][2], verbose=verbose)
                    smth_time = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[1][2], verbose=verbose)
                    if len(operands) == 3:
                        init_value = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[2][2], verbose=verbose)
                    elif len(operands) == 2:
                        init_value = expr_value
                    else:
                        raise Exception('Invalid number of args for SMTH1.')
                    
                    # register
                    if tuple(operands[0]) not in self.time_expr_register.keys():
                        self.time_expr_register[tuple(operands[0])] = list()
                        for i in range(order):
                            self.time_expr_register[tuple(operands[0])].append(smth_time/order*init_value)
                    # outflows
                    outflows = list()
                    for i in range(order):
                        outflows.append(self.time_expr_register[tuple(operands[0])][i]/(smth_time/order) * self.sim_specs['dt'])
                        self.time_expr_register[tuple(operands[0])][i] -= outflows[i]
                    # inflows
                    self.time_expr_register[tuple(operands[0])][0] += expr_value * self.sim_specs['dt']
                    for i in range(1, order):
                        self.time_expr_register[tuple(operands[0])][i] += outflows[i-1]

                    return outflows[-1] / self.sim_specs['dt']

                elif func_name == 'SMTH3':
                    # arg values
                    order = 3
                    expr_value = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[0][2], verbose=verbose)
                    smth_time = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[1][2], verbose=verbose)
                    if len(operands) == 3:
                        init_value = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[2][2], verbose=verbose)
                    elif len(operands) == 2:
                        init_value = expr_value
                    else:
                        raise Exception('Invalid number of args for SMTH3.')
                    
                    # register
                    if tuple(operands[0]) not in self.time_expr_register.keys():
                        self.time_expr_register[tuple(operands[0])] = list()
                        for i in range(order):
                            self.time_expr_register[tuple(operands[0])].append(smth_time/order*init_value)
                    # outflows
                    outflows = list()
                    for i in range(order):
                        outflows.append(self.time_expr_register[tuple(operands[0])][i]/(smth_time/order) * self.sim_specs['dt'])
                        self.time_expr_register[tuple(operands[0])][i] -= outflows[i]
                    # inflows
                    self.time_expr_register[tuple(operands[0])][0] += expr_value * self.sim_specs['dt']
                    for i in range(1, order):
                        self.time_expr_register[tuple(operands[0])][i] += outflows[i-1]

                    return outflows[-1] / self.sim_specs['dt']
                else:
                    raise Exception('Unknown time-related operator {}'.format(operator[0]))
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v9 Time-related Func:', value)
            elif operator[0] in self.lookup_functions: # LOOKUP
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'Lookup func. operator:', operator, 'operands:', operands)
                func_name = operator[0]
                if func_name == 'LOOKUP':
                    look_up_func_node = operands[0][2]
                    look_up_func_name = parsed_equation.nodes[look_up_func_node]['operands'][0][1]
                    look_up_func = self.graph_functions[look_up_func_name]
                    input_value = self.calculate_node(parsed_equation=parsed_equation, node_id=operands[1][2], verbose=verbose)
                    value = look_up_func(input_value)
                else:
                    raise Exception('Unknown Lookup function {}'.format(operator[0]))
            else:
                raise Exception('Unknown operator {}'.format(operator[0]))
        
        self.id_level -= 1
        
        if verbose:
            print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'value for {} {}'.format(node_id, subscript), value)

        if subscript is None:
            if type(value) is dict:
                raise Exception('Value cannot be dict if subscript is None. Var: {}'.format(var_name))
            else:
                return value
        else:
            if type(value) is dict:
                return value[subscript]
            else:
                return value


class GraphFunc(object):
    def __init__(self, yscale, ypts, xscale=None, xpts=None):
        self.yscale = yscale
        self.ypts = ypts
        self.eqn = None
        
        if xpts:
            self.xpts = xpts
        else:
            self.xscale = xscale
            self.xpts = np.linspace(self.xscale[0], self.xscale[1], num=len(self.ypts))
        
        from scipy.interpolate import interp1d

        self.interp_func = interp1d(self.xpts, self.ypts, kind='linear')

    def __call__(self, input):
        # input out of xscale treatment:
        input = max(input, self.xpts[0])
        input = min(input, self.xpts[-1])
        output = float(self.interp_func(input)) # the output (like array([1.])) needs to be converted to float to avoid dimension explosion
        return output


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


class Structure(object):
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
            'time_units' :'Weeks'
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
            'TIME': 0
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
            print(self.HEAD, 'Reading XMILE model from {}'.format(from_xmile))
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
                    gf = var.find('gf')
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

                    equation = GraphFunc(yscale=yscale, ypts=ypts, xscale=xscale, xpts=xpts)
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
                        self.var_dimensions[var.get('name')] = None
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
        if type(equation) in [int, float, np.int_, np.float_]:
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
        if type(equation) in [int, float, np.int_, np.float_]:
            equation = str(equation)
        self.flow_positivity[name] = non_negative
        if leak:
            self.leak_conveyors[name] = None # to be filled when parsing the conveyor
        self.flow_equations[name] = equation
    
    def add_aux(self, name, equation):
        if type(equation) in [int, float, np.int_, np.float_]:
            equation = str(equation)
        self.aux_equations[name] = equation

    def replace_element_equation(self, name, new_equation):
        if type(new_equation) is str:
            pass
        elif type(new_equation) in [int, float, np.int_, np.float_]:
            new_equation = str(new_equation)
        else:
            raise Exception('Unsupported new equation {} type {}'.format(new_equation, type(new_equation)))
        
        if name in self.stock_equations:
            self.stock_equations[name] = new_equation
        elif name in self.flow_equations:
            self.flow_equations[name] = new_equation
        elif name in self.aux_equations:
            self.aux_equations[name] = new_equation
        else:
            raise Exception('Unable to find {} in the current model'.format(name))

    def parse_1(self, var, equation):
        if type(equation) is GraphFunc:
            gfunc_name = 'GFUNC{}'.format(len(self.graph_functions_renamed))
            self.graph_functions_renamed[gfunc_name] = equation # just for length ... for now
            self.graph_functions[var] = equation
            self.parser.functions.update({gfunc_name:gfunc_name+"(?=\()"}) # make name var also a function name and add it to the parser
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

        else:
            parsed_equation = self.parser.parse(equation)
            return parsed_equation
    
    def parse_0(self, equations, parsed_equations, verbose=False):
        for var, equation in equations.items():
            if verbose:
                print(self.HEAD, "Parsing:", var)
                print(self.HEAD, "    Eqn:", equation)
            
            if type(equation) is dict:
                parsed_equations[var] = dict()
                for k, ks in equation.items():
                    parsed_equations[var][k] = self.parse_1(var=var, equation=ks)
            
            else:
                parsed_equations[var] = self.parse_1(var=var, equation=equation)
            

    def parse(self, verbose=False):
        # string equation -> calculation tree

        self.parse_0(self.stock_equations, self.stock_equations_parsed, verbose=verbose)
        self.parse_0(self.flow_equations, self.flow_equations_parsed, verbose=verbose)
        self.parse_0(self.aux_equations, self.aux_equations_parsed, verbose=verbose)
        
    def is_dependent(self, var1, var2):
        # determine if var2 depends on var1, i.e., var1 --> var2 or var1 appears in var2's equation
        dependent = False
        parsed_equation_var2 = (self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)[var2]
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

    def calculate_dependents(self, parsed_equation):
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
                    # print('i5.1', operands[0][0])
                    var_dependent = operands[0][1]
                    # print('i5.2', var_dependent)
                    self.calculate_variable_dynamic(var=var_dependent)
                elif operands[0][0] == 'FUNC': # this refers to a subscripted variable like 'a[ele1]'
                    # print('i5.3')
                    # need to find that 'SPAREN' node
                    var_dependent_node_id = operands[0][2]
                    var_dependent = parsed_equation.nodes[var_dependent_node_id]['operands'][0][1]
                    # print('var_dependent2', var_dependent)
                    self.calculate_variable_dynamic(var=var_dependent)
    
    def calculate_variable_dynamic(self, var, subscript=None, verbose=False, leak_frac=False, conveyor_init=False, conveyor_len=False):
        # print("\nEngine Calculating: {:<15} on subscript {}".format(var, subscript), '\n', 'name_space:', self.name_space, '\n', 'flow_effects', self.stock_shadow_values)

        if var == 'TIME':
            return
        else:
            if subscript is not None:
                parsed_equation = (self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)[var][subscript]
            else:
                parsed_equation = (self.stock_equations_parsed | self.flow_equations_parsed | self.aux_equations_parsed)[var]
            
            # proceed to the 'real' calculation
            # A: var is a Conveyor
            if var in self.conveyors:
                # print('Calculating Conveyor {}'.format(var))
                if not (conveyor_init or conveyor_len):
                    if not self.conveyors[var]['conveyor'].is_initialized:
                        # print('Initialising {}'.format(var))
                        # when initialising, equation of the conveyor needs to be evaluated, setting flag conveyor_len to True 
                        self.calculate_variable_dynamic(var=var, subscript=subscript, verbose=verbose, conveyor_len=True)
                        conveyor_length = self.conveyors[var]['len']
                        length_steps = int(conveyor_length/self.sim_specs['dt'])
                        
                        # when initialising, equation of the conveyor needs to be evaluated, setting flag conveyor_init to True 
                        self.calculate_variable_dynamic(var=var, subscript=subscript, verbose=verbose, conveyor_init=True)
                        conveyor_init_value = self.conveyors[var]['val']
                        
                        leak_flows = self.conveyors[var]['leakflow']
                        if len(leak_flows) == 0:
                            leak_fraction = 0
                        else:
                            for leak_flow in leak_flows.keys():
                                self.calculate_variable_dynamic(var=leak_flow, subscript=subscript, verbose=verbose, leak_frac=True)
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
                    self.calculate_dependents(parsed_equation=parsed_equation)
                    self.conveyors[var]['len'] = self.solver.calculate_node(parsed_equation=parsed_equation, verbose=verbose, var_name=var)
                
                elif conveyor_init:
                    # print('Calculating INIT VAL for {}'.format(var))
                    # it is the intitial value of the conveyoer
                    parsed_equation = self.stock_equations_parsed[var][1]
                    self.calculate_dependents(parsed_equation=parsed_equation)
                    self.conveyors[var]['val'] = self.solver.calculate_node(parsed_equation=parsed_equation, verbose=verbose, var_name=var)
            
            # B: var is a normal stock
            elif var not in self.conveyors and var in self.stocks:
                if not self.stocks[var].initialised:
                    # print('Stock {} not initialised'.format(var))
                    if type(parsed_equation) is dict:
                        for k, sub_parsed_equation in parsed_equation.items():
                            # self.calculate_variable_dynamic(var=var, subscript=k)
                            self.calculate_dependents(parsed_equation=sub_parsed_equation)
                            value = self.solver.calculate_node(parsed_equation=sub_parsed_equation, subscript=k, verbose=verbose, var_name=var)
                            if var not in self.name_space:
                                self.name_space[var] = dict()
                            self.name_space[var][k] = value
                    else:
                        self.calculate_dependents(parsed_equation=parsed_equation)
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
                                    self.calculate_variable_dynamic(var=in_flow, subscript=subscript, verbose=verbose)
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
                                    self.calculate_variable_dynamic(var=out_flow, subscript=subscript, verbose=verbose)
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
                            self.calculate_variable_dynamic(var=self.leak_conveyors[var], subscript=subscript)
                    else:
                        # it is the value of the leak_fraction (a percentage) that is requested.    
                        # leak_fraction is calculated using leakflow's equation. 
                        parsed_equation = self.flow_equations_parsed[var]
                        self.calculate_dependents(parsed_equation=parsed_equation)
                        self.conveyors[self.leak_conveyors[var]]['leakflow'][var] = self.solver.calculate_node(parsed_equation=parsed_equation, verbose=verbose, var_name=var)

                elif var in self.outflow_conveyors:
                    # requiring an outflow's value triggers the calculation of its connected conveyor
                    if var not in self.name_space: # the outflow is not calculated, which means the conveyor has not been initialised
                        self.calculate_variable_dynamic(var=self.outflow_conveyors[var], subscript=subscript)
                        
                elif var in self.flow_equations:
                    if var not in self.name_space:
                        if type(parsed_equation) is dict:
                            for k, sub_parsed_equaton in parsed_equation.items():
                                # self.calculate_variable_dynamic(var=var, subscript=k)
                                self.calculate_dependents(parsed_equation=sub_parsed_equaton)
                                value = self.solver.calculate_node(parsed_equation=sub_parsed_equaton, subscript=k, verbose=verbose, var_name=var)
                                
                                # control flow positivity by itself
                                if self.flow_positivity[var] is True:
                                    if value < 0:
                                        value = 0
                                if var not in self.name_space:
                                    self.name_space[var] = dict()
                                self.name_space[var][k] = value
                        else:
                            self.calculate_dependents(parsed_equation=parsed_equation)
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
                        for k, sub_parsed_equation in parsed_equation.items():
                            # self.calculate_variable_dynamic(var=var, subscript=k)
                            self.calculate_dependents(parsed_equation=sub_parsed_equation)
                            value = self.solver.calculate_node(parsed_equation=sub_parsed_equation, subscript=k, verbose=verbose, var_name=var)
                            if var not in self.name_space:
                                self.name_space[var] = dict()
                            self.name_space[var][k] = value
                    else:
                        self.calculate_dependents(parsed_equation=parsed_equation)
                        value = self.solver.calculate_node(parsed_equation=parsed_equation, subscript=subscript, verbose=verbose, var_name=var)
                        self.name_space[var] = value
                            
                else:
                    pass
            
            else:
                raise Exception("Undefined var: {}".format(var))

    def calculate_variables_dynamic(self, verbose=False):
        for var in (self.stock_equations_parsed | self.aux_equations_parsed | self.flow_equations_parsed).keys():
            self.calculate_variable_dynamic(var=var, verbose=verbose)
    
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
            
    def simulate(self, time=None, dt=None, dynamic=True, verbose=False, debug_against=None):
        if debug_against is not None:
            import pandas as pd
            if debug_against is True:
                self.df_debug_against = pd.read_csv('stella.csv')
            else:
                self.df_debug_against = pd.read_csv(debug_against)

        self.parse(verbose=verbose)

        if time is None:
            time = self.sim_specs['simulation_time']
        if dt is None:
            dt = self.sim_specs['dt']
        steps = int(time/dt)

        def step(debug=False):
            if verbose:
                # print('--time {} --'.format(self.sim_specs['current_time']))
                print('--step {} start--'.format(s))
                # print('\n--step {} start--\n'.format(s), self.name_space)
            self.calculate_variables_dynamic(verbose=verbose)
            
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
                        print('ASDM   result:', asdm_result)
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
        
        self.current_step = 0
        for s in range(steps+1):
            if debug_against:
                step(debug=self.df_debug_against.iloc[self.current_step])
            else:
                step()
            self.current_step += 1

    def trace_error(self, var_with_error, sub=None):
        self.debug_level += 1

        print(self.debug_level*'\t', 'Tracing error on {} ...'.format(var_with_error))
        print(self.debug_level*'\t', 'ASDM value    :', self.name_space[var_with_error])
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
                    print(self.debug_level*'\t', '   ASDM value    :', self.name_space[var_dependent])
                    print(self.debug_level*'\t', '   Expected value:', self.df_debug_against.iloc[self.current_step][self.var_name_to_csv_entry(var_dependent)])
        
                elif operands[0][0] == 'FUNC': # this refers to a subscripted variable like 'a[ele1]'
                    # need to find that 'SPAREN' node
                    var_dependent_node_id = operands[0][2]
                    var_dependent = parsed_equation.nodes[var_dependent_node_id]['operands'][0][1]
                    print(self.debug_level*'\t', '-- Dependent:', var_dependent)
                    print(self.debug_level*'\t', '   ASDM value    :', self.name_space[var_dependent])
                    print(self.debug_level*'\t', '   Expected value:', self.df_debug_against.iloc[self.current_step][self.var_name_to_csv_entry(var_dependent)])
        
        if var_with_error in self.flow_stocks:
            connected_stocks = self.flow_stocks[var_with_error]
            for direction, connected_stock in connected_stocks.items():
                print(self.debug_level*'\t', '-- Connected stock: {:<4} {}'.format(direction, connected_stock))
                print(self.debug_level*'\t', '   ASDM value    :', self.name_space[connected_stock])
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

        self.full_result = dict()
        self.full_result_flattened = dict()

        self.solver = Solver(
            sim_specs=self.sim_specs,
            dimension_elements=self.dimension_elements,
            name_space=self.name_space,
            graph_functions=self.graph_functions,
            )

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
            
    def export_simulation_result(self, flatten=False, format='dict', to_csv=False):
        self.full_result = dict()
        for time, slice in self.time_slice.items():
            for var, value in slice.items():
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
        if to_csv or flatten or format == 'df':
            self.full_result_flattened = dict()
            for var, result in self.full_result.items():
                if type(result) is dict:
                    for sub, subresult in result.items():
                        self.full_result_flattened[var+'[{}]'.format(', '.join(sub))] = subresult
                else:
                    self.full_result_flattened[var] = result
        if to_csv or format == 'df':
            import pandas as pd
            from pprint import pprint
            df_full_result = pd.DataFrame.from_dict(self.full_result_flattened)
            df_full_result.drop([self.sim_specs['time_units']],axis=1, inplace=True)
            if type(to_csv) is not str:
                df_full_result.to_csv('asdm.csv', index=False)
            else:
                df_full_result.to_csv(to_csv)

        if format == 'dict':
            if flatten:
                return self.full_result_flattened
            else:
                return self.full_result
        elif format == 'df':
            return df_full_result
        else:
            raise Exception("Invalid value for arg 'format': {}".format(format))
    
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

