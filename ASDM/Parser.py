import re
import networkx as nx

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
            'NGT': r'\<\=',
            'NLT': r'\>\=',
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
            'DIVIDE': r'\/',
        }

        self.functions = {
            'MIN': r'MIN',
            'MAX': r'MAX',
            'INIT': r'INIT',
            'DELAY': r'DELAY',
            'STEP': r'STEP',
            'HISTORY': r'HISTORY',
        }

        self.names = {
            'NAME': r'[a-zA-Z_\?][a-zA-Z0-9_\?]*'
        }
        self.numbers = {
            'NUMBER': r'(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
            # 'NUMBER': r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
            # 'NUMBER': r'((?<!\d)-\d+|\d+)'
            # 'NUMBER': r'^-?\d+(\.\d+)?$'
            # 'NUMBER': r'((?:^\-?\d+)|(?:(?<=[-+/*])(?:\-?\d+)))'
            # 'NUMBER': r'^-?[0-9]\d*(\.\d+)?$'
            # 'NUMBER': r'(-?\d+\.\d+)|(-?\d+)'
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
        self.patterns_arithmetic_1 = {
            ### Arithmetic 1 ###

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
        }
        self.patterns_arithmetic_2 = {
            ### Arithmetic 2 ###

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
            'INIT__LPAREN__FUNC__RPAREN':{
                'token':['FUNC', 'INIT'],
                'operator':['INIT'],
                'operand':['FUNC']
            },
            'DELAY__LPAREN__DOT+__RPAREN':{
                'token':['FUNC', 'DELAY'],
                'operator':['DELAY'],
                'operand':['FUNC']
            },
            'HISTORY__LPAREN__DOT+__RPAREN':{
                'token':['FUNC', 'DELAY'],
                'operator':['HISTORY'],
                'operand':['FUNC']
            },
            'STEP__LPAREN__FUNC__COMMA__FUNC__RPAREN':{
                'token':['FUNC', 'STEP'],
                'operator':['STEP'],
                'operand':['FUNC']
            }
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
        while len(s) > 0:
            for type_name, type_regex in (self.numbers | self.special_symbols | self.logic_operators | self.arithmetic_operators | self.functions | self.names).items():
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
            r = 100
            while len(items) > 1 and r > 0:
                r -= 1 # use this line to put a stop by number of iterations
                if r == 0:
                    raise Exception('Parser timeout on String\n  {}, \nItems\n  {}'.format(string, items))
                items_changed = False
                # print(self.HEAD, '---Parsing---','\n')
                # print(self.HEAD, 'items1', items)
                # print(self.HEAD, graph.nodes.data(True))
                # print(self.HEAD, graph.edges.data(True))
                for pattern, func in ( # the order of the following patterns is IMPORTANT
                    self.patterns_sub_var | \
                    self.patterns_num | \
                    self.patterns_var | \
                    self.patterns_custom_func | \
                    self.patterns_built_in_func | \
                    self.patterns_brackets | \
                    self.patterns_arithmetic_1 | \
                    self.patterns_arithmetic_2 | \
                    self.patterns_logic | \
                    self.patterns_conditional
                    ).items(): # loop over all patterns
                    pattern = pattern.split('__')
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
    string_4b = '4*(1-(2*3))'
    string_4c = '10-4-3-2-1'
    string_5a = 'a > b'
    string_5b = 'a < 2'
    string_6a = 'IF a THEN b ELSE c'
    string_6b = 'IF a THEN IF ba THEN bb ELSE bc ELSE c'
    string_7a = 'a[b]'
    string_7b = 'a[b,c]'
    string_7c = 'a[1,b]'
    string_full = 'IF (a + h[ele1]) > (c * 10) THEN INIT(d / e) ELSE f - g'

    string_a = '(Pre_COVID_capacity_for_diagnostics*(1-(COVID_period/100)*(Reduced_diagnostic_capacity_during_COVID/100)*COVID_switch))+((((Percent_increase_in_diagnostic_capacity_post_COVID/100)*COVID_switch)*Pre_COVID_capacity_for_diagnostics)*(Timing_of_new_diagnostic_capacity/100))'
    string_b = '(a*(1-(b/100)*(c/100)*d))+((((e/100)*d)*a)*(e/100))' # equivalent to string_a
    string_c = 'a*b*c*d'
    string_d = '(a*(1-(b/100)*(c/100)*d))'
    string_e = 'DELAY(Waiting_more_than_12mths, 52)-Between_12_to_24mth_wait_to_urgent-Waiting_more_than_12mths-Routine_treatment_from_12_to_24mth_wait'
    string_f = 'DELAY(a, 52)-b-a-c'

    parser = Parser()

    ####
    graph = parser.parse(string_d)
    ####

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    labels = {}
    labels_operators = nx.get_node_attributes(graph, 'operator')
    labels_operands = nx.get_node_attributes(graph, 'operands')
    for id, label_operator in labels_operators.items():
        labels[id] = str(id) + '\n' + 'operator:' + str(label_operator) + '\n' + 'operands:' + str(labels_operands[id])
    labels['root'] = 'root'
    
    # nx.draw_shell(graph, with_labels=True, labels=labels, node_color='C1', font_size=9)
    # nx.draw(graph, with_labels=True, labels=labels, node_color='C1', font_size=9)
    
    pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
    nx.draw(graph, with_labels=True, labels=labels, node_color='C1', font_size=9, pos=pos)

    plt.show()
