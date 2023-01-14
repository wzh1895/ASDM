import re
import networkx as nx
from pprint import pprint
from Var import Var

"""
Hierarchy in parsing

- Number                     1
- Var                        stock_1, variable[element1, element2]
- Custom functions           INIT(), DELAY(), etc
- Arithmetic1                * /
- Arithmetic2                + -
- Brackets                   ()
- Logic                      AND OR NOT > < >= <= <> == 
- Conditional statements     IF THEN ELSE

"""

class Parser(object):
    def __init__(self):
        self.special_symbols = {
            'COMMA': r',',
            'LPAREN': r'\(',
            'RPAREN': r'\)',
            'LSPAREN': r'\[',
            'RSPAREN': r'\]',
        }

        self.logic_operators ={
            'GT': r'\>',
            'LT': r'\<',
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
            'DIVIDE': r'\/',
        }

        self.functions = {
            'INIT': r'INIT',
            'DELAY': r'DELAY',
        }

        self.elements = {
            'NAME': r'[a-zA-Z_][a-zA-Z0-9_]*',
            'NUMBER': r'\d+',
        }

        self.node_id = 0

        self.patterns = {
            # token: to put as placeholder in 'items'
            # operator and operands: to put into parsed_equation 

            'NAME_LSPAREN_DOT+_RSPAREN':{
                'token':['FUNC', 'SPAREN'],
                'operator':['SPAREN'],
                'operand':['NUMBER', 'NAME']
            },

            'NUMBER':{
                'token':['FUNC', 'IS'],
                'operator':['IS'],
                'operand':['NUMBER']
            },
            'NAME': {
                'token':['FUNC', 'EQUALS'],
                'operator':['EQUALS'],
                'operand':['NAME']
                },

            'INIT_LPAREN_FUNC_RPAREN':{
                'token':['FUNC', 'INIT'],
                'operator':['INIT'],
                'operand':['FUNC']
            },
            'DELAY_LPAREN_FUNC_COMMA_FUNC_RPAREN':{
                'token':['FUNC', 'DELAY'],
                'operator':['DELAY'],
                'operand':['FUNC']
            },


            'FUNC_TIMES_FUNC': {
                'token':['FUNC', 'TIMES'],
                'operator':['TIMES'],
                'operand':['FUNC']
                },
            'FUNC_DIVIDE_FUNC': {
                'token':['FUNC', 'DIVIDE'],
                'operator':['DIVIDE'],
                'operand':['FUNC']
                },
            'FUNC_PLUS_FUNC': {
                'token':['FUNC', 'PLUS'],
                'operator':['PLUS'],
                'operand':['FUNC']
                },
            'FUNC_MINUS_FUNC': {
                'token':['FUNC', 'MINUS'],
                'operator':['MINUS'],
                'operand':['FUNC']
                },


            'LPAREN_FUNC_RPAREN':{
                'token':['FUNC', 'PAREN'],
                'operator':['PAREN'],
                'operand':['FUNC']
            },


            'FUNC_GT_FUNC':{
                'token':['FUNC', 'GT'],
                'operator':['GT'],
                'operand':['FUNC']
            },
            'FUNC_LT_FUNC':{
                'token':['FUNC', 'LT'],
                'operator':['LT'],
                'operand':['FUNC']
            },


            'CONIF_FUNC_CONTHEN_FUNC_CONELSE_FUNC':{
                'token':['FUNC', 'CON'],
                'operator':['CON'],
                'operand':['FUNC']
            }
        }

        self.HEAD = "PARSER"

    def tokenisation(self, s):
        items = []
        while len(s) > 0:
            for type_name, type_regex in (self.special_symbols | self.logic_operators | self.arithmetic_operators | self.functions | self.elements).items():
                m = re.match(pattern=type_regex, string=s)
                if m:
                    # if type_name == 'NUMBER':
                    #     item = float(m[0])
                    # else:
                    #     item = m[0]
                    item = m[0]
                    items.append([type_name, item])
                    s = s[m.span()[1]:].strip()
                    break
        return items

    def parse(self, string):
        if type(string) is dict:
            parsed_equations = dict()
            for k, ks in string.items():
                parsed_equations[k] = self.parse(ks)
            return parsed_equations
        else:
            # print(self.HEAD, 'Parse string:', string)
            items = self.tokenisation(string)
            # print(self.HEAD, 'Items:', items)
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
            r = 1
            while len(items) > 1 and r > 0:
                # r -= 1 # use this line to put a stop by number of iterations
                items_changed = False
                # print(self.HEAD, '\n', '---parsing---','\n')
                # print(self.HEAD, 'items1', items)
                # print(self.HEAD, graph.nodes.data(True))
                # print(self.HEAD, graph.edges.data(True))
                for pattern, func in self.patterns.items(): # loop over all patterns
                    pattern = pattern.split('_')
                    # print(self.HEAD, "Searching for {} with operator {}".format(pattern, func['operator']))
                    pattern_len = len(pattern)
                    for i in range(len(items)): # loop over all positions for this pattern
                        # print(self.HEAD, 'Checking item:', i, items[i])
                        if len(items) - i >= pattern_len:
                            matched = True
                            for j in range(pattern_len): # matching pattern at this position
                                if pattern[j] == items[i+j][0]: # exact match
                                    pass
                                else: # fuzzy match
                                    if pattern[j] == 'DOT+':
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
                                # print(self.HEAD, "Found {} at {}".format(pattern, matched_items))
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
                # print(self.HEAD, 'items2', items)
                # print(self.HEAD, 'graph:')
                # print(self.HEAD, graph.nodes.data(True))
                # print(self.HEAD, graph.edges.data(True))
            
            # add root node as entry
            graph.add_node('root')
            graph.add_edge('root', items[0][2])
            
            # print(self.HEAD, 'Parse finished for:', string)
            # print(self.HEAD, '  ', graph.nodes.data(True))
            # print(self.HEAD, '  ', graph.edges.data(True))
            return graph


if __name__ == '__main__':

    string_0a = 'a'
    string_1a = 'a+b-c'
    string_1b = 'a+b-2'
    string_2a = 'a*b'
    string_2b = 'a/2'
    string_3a = 'INIT(a)'
    string_3b = 'DELAY(a, 1)'
    string_4a = 'a*((b+c)-d)'
    string_5a = 'a > b'
    string_5b = 'a < 2'
    string_6a = 'IF a THEN b ELSE c'
    string_6b = 'IF a THEN IF ba THEN bb ELSE bc ELSE c'
    string_7a = 'a[b]'
    string_7b = 'a[b,c]'
    string_7c = 'a[1,b]'
    string_full = 'IF (a + h[ele1]) > (c * 10) THEN INIT(d / e) ELSE f - g'

    parser = Parser()

    ####
    graph = parser.parse(string_7c)
    ####

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    labels = {}
    labels_operators = nx.get_node_attributes(graph, 'operator')
    labels_operands = nx.get_node_attributes(graph, 'operands')
    for id, label_operator in labels_operators.items():
        labels[id] = str(id) + '\n' + 'operator:' + str(label_operator) + '\n' + 'operands:' + str(labels_operands[id])
    labels['root'] = 'root'
    nx.draw_shell(graph, with_labels=True, labels=labels, node_color='C1')
    plt.show()
