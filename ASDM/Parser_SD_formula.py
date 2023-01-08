import re
from pprint import pprint

# tokens = {NAME, NUMBER, PLUS}

# # Tokens
# NAME = r'[a-zA-Z_][a-zA-Z0-9_]*'
# NUMBER = r'\d+'

# tokens = [NAME, NUMBER]

# # Special symbols
# PLUS = r'\+'
# MINUS = r'\-'
# TIMES = r'\*'
# DIVIDE = r'\/'
# ASSIGN = r'\='
# LPAREN = r'\('
# RPAREN = r'\)'
# LSPAREN = r'\['
# RSPAREN = r'\]'
# CON = r'IF_THEN_ELSE'
# CON_IF = r'IF'
# CON_THEN = r'THEN'
# CON_ELSE = r'ELSE'

# special_symbols = [PLUS, MINUS, TIMES, DIVIDE, ASSIGN, LPAREN, RPAREN, LSPAREN, RSPAREN, CON_IF, CON_THEN, CON_ELSE]

# Special symbols

special_symbols = {
    'COMMA': r',',
    'LPAREN': r'\(',
    'RPAREN': r'\)',
    'LSPAREN': r'\[',
    'RSPAREN': r'\]',
    'GT': r'\>',
    'LT': r'\<',
    'AND': r'AND',
    'OR': r'OR',
    'NOT': r'NOT',
    'CON': r'IF_THEN_ELSE',
    'CONIF': r'IF',
    'CONTHEN': r'THEN',
    'CONELSE': r'ELSE'
    }

arithmetics = {
    'PLUS': r'\+',
    'MINUS': r'\-',
    'TIMES': r'\*',
    'DIVIDE': r'\/'
}

functions = {
    'INIT': r'INIT',
    'DELAY': r'DELAY'
}

# Tokens
tokens = {
    'NAME': r'[a-zA-Z_][a-zA-Z0-9_]*',
    'NUMBER': r'\d+'
}

import networkx as nx

formula1 = "b = 1"
formula2 = "z = IF a + b[ele1] > c * 10 THEN INIT(d / e) ELSE f - g"

"""
Hierarchy in parsing

- Assignment                 =
- Conditional statements     IF THEN ELSE
- Custom functions           INIT(), DELAY(), etc
- Brackets                   ()
- Logic                      AND OR NOT > < >= <= <> == 
- Arithmetic1                * /
- Arithmetic2                + -
- Name                       stock_1, variable[element1, element2]

"""


# hierarchy = {
#     'AS': [ASSIGN], 
#     'CF': ['INIT\b'], 
#     'SS': [LSPAREN, RSPAREN], 
#     'CS': [r'IF\b', r'THEN\b', r'ELSE\b'], 
#     'BR': [LPAREN, RPAREN], 
#     'LG': [r'AND\b', r'OR\b', r'NOT\b', r'>', r'<', r'>=', r'<=', r'<>', r'=='],
#     'AR': [PLUS, MINUS, TIMES, DIVIDE]
#     }

# for hi, symbol in hierarchy.items():
#     if len(symbol) != 0:
#         for s in symbol:
#             a = re.search(s, formula2)
#             print(a)

# def parse_assign(string, graph):
#     if re.search(r'=', string):
#         p = re.split(r'=', string)
#         p = [x.strip() for x in p]
#         print(p)
#         graph.add_node(p[0], expression=p[1])
#     else:
#         pass

node_num = 0

"""
Node structure

{
    name: z,
    calc_map: {
        ele0: {
            func: if_then_else,
            args: {
                p1: { # IF
                    func: >,
                    args:{
                        p1: {
                            func: plus,
                            args: {
                                p1: {
                                    func: factor,   --> factor can retrieve var value by its name from name_space, or calculate and add it to the name_space
                                    args: {
                                        p1: 'a'
                                    },
                                },
                                p2: {
                                    func: factor,   
                                    args: {
                                        p1: 'b[ele1]'
                                    }
                                }
                            }
                        },
                        p2: {
                            func: time,
                            args: {
                                p1: {
                                    func: factor,
                                    args: {
                                        p1: 'c'
                                    },
                                p2: 10
                                }
                            }
                        }
                    }
                        
                },
                p2: { # THEN
                    func: init,             --> time-related function, takes an expression ("d/e") as its argument
                    args: {
                        p1: "d/e"
                        }
                    }
                },
                p3: { # ELSE
                    func: minus,
                    args: {
                        p1: {
                            func: factor,   
                            args: 'f'
                            }
                        },
                        p2: {
                            func: factor,   
                            args: 'g'
                        }
                    }
                }
            }
        }
    }
}

"""


"""
Step 1: use equations to generate
    (1) calculation graph of each variable
    (2) calsal loop diagram of variables

Step 2: use root(flow/end of calculation) to initiate calculation
    (1) dict-driven, i.e., v = dict['func'](dict['args'])
    (2) thus recursive

"""

node_id = 0

# def parse(items, graph):
#     global node_id
#     r_limit = 5
#     r = 0
#     while len(items) > 1 and r <= r_limit:
#         r+=1
#         print('parsing:', items)

#         ### IF_THEN_ELSE ###

#         # There is a pattern "xxx IF xxx THEN xxx ELSE xxx", with "xxx" includes no IF - that's the basic element
#         # In the scanning process, if the scanner encounters IF, it should register the location, then start to 
#         # look for THEN and ELSE. But this process if interrupted if another IF is found, in this case the new IF
#         # will replace the previous one, and the THEN should be associated with this new IF.
#         # Once a complete set of IF_THEN_ELSE is found, the range of ELSE should be determined. This could be done
#         # by continuing the scan until the end of the list of tokens, or until the next IF or THEN or ELSE

#         pointer_if = None
#         pointer_then = None
#         pointer_else = None
#         pointer_endif = len(items) - 1
#         con_counter = 0

#         for i in range(len(items)):
#             if items[i][1] == 'CON_IF':
#                 if pointer_if is None:
#                     pointer_if = i
#                 con_counter += 1
#             elif items[i][1] == 'CON_THEN':
#                 if con_counter == 1:
#                     pointer_then = i
#             elif items[i][1] == 'CON_ELSE':
#                 if con_counter == 1:
#                     pointer_else = i
#                     break
#                 else:
#                     con_counter -= 1

#         if pointer_if is not None:

#             # matched = [items[pointer_if+1:pointer_then], items[pointer_then+1:pointer_else], items[pointer_else+1:pointer_endif+1]]
#             # print(matched[0])
#             # print(matched[1])
#             # print(matched[2])

#             # print('new items')
#             node_id += 1
#             graph.add_node(str(node_id), token=[('CON', 'FUNC')])
#             items = items[:pointer_if] + [(node_id, 'FUNC')] + items[pointer_endif+1:]
#             # print(items)

#             # branch_id = 0
#             # for m in matched:
#             #     full_branch_id = str(node_id)+'.'+str(branch_id)
#             #     graph.add_node(full_branch_id, token=m)
#             #     parse(m, graph, full_branch_id)
#             #     graph.add_edge(str(node_id), full_branch_id)
#             #     branch_id += 1
            
#             continue

#         ### FUNCTIONS ###

#         pointer_custom_func = None
#         pointer_custom_func_args_left = None
#         pointer_custom_func_args_right = None
#         pointer_args = []
#         paren_counter = 0

#         for i in range(len(items)):
#             if items[i][1] in functions.keys():
#                 pointer_custom_func = i
#                 continue
#             if pointer_custom_func is not None:
#                 if items[i][1] == 'LPAREN':
#                     if not pointer_custom_func_args_left:
#                         pointer_custom_func_args_left = i
#                         pointer_args.append(i+1)
#                         paren_counter += 1
#                     else:
#                         paren_counter += 1
#             if pointer_custom_func is not None and pointer_custom_func_args_right is None:
#                 if items[i][1] == 'COMMA':
#                     pointer_args.append(i)
#                     pointer_args.append(i+1)
#             if pointer_custom_func is not None:
#                 if items[i][1] == 'RPAREN':
#                     if paren_counter == 1:
#                         pointer_custom_func_args_right = i
#                         pointer_args.append(i)
#                         break
#                     else:
#                         paren_counter -= 1
        
#         if pointer_custom_func is not None:
#             # matched = []
#             # for i in range(0, len(pointer_args)-1, 2):
#             #     matched.append(items[pointer_args[i]:pointer_args[i+1]])
            
#             node_id += 1
#             graph.add_node(str(node_id), token=[('INIT', 'FUNC')])
#             items = items[:pointer_custom_func] + [(node_id, 'FUNC')] + items[pointer_custom_func_args_right+1:]

#             # branch_id = 0
#             # for m in matched:
#             #     full_branch_id = str(node_id)+'.'+str(branch_id)
#             #     graph.add_node(full_branch_id, token=m)
#             #     parse(m, graph, full_branch_id)
#             #     graph.add_edge(str(node_id), full_branch_id)
#             #     branch_id += 1
            
#             continue
        
#         ### PARENTES ###

#         pointer_lp = None
#         pointer_rp = None
#         paren_counter = 0

#         for i in range(len(items)):
#             if items[i][1] == 'LPAREN':
#                 if pointer_lp is None:
#                     pointer_lp = i
#                 paren_counter += 1
#                 continue
#             if items[i][1] == 'RPAREN':
#                 if paren_counter == 1:
#                     pointer_rp = i
#                     break
#                 else:
#                     paren_counter -= 1
        
#         if pointer_lp is not None:
#             # print('matched parentes in', items)
#             # matched = [items[pointer_lp+1:pointer_rp]]

#             node_id += 1
#             graph.add_node(str(node_id), token=[('PARENTES', 'FUNC')])
#             items = items[:pointer_lp] + [('PARENTES', 'FUNC')] + items[pointer_rp+1:]

#             # branch_id = 0
#             # for m in matched:
#             #     full_branch_id = str(node_id)+'.'+str(branch_id)
#             #     graph.add_edge(str(node_id), full_branch_id)
#             #     parse(m, graph, full_branch_id)
#             #     graph.add_node(full_branch_id, token=m)
#             #     branch_id += 1
            
#             continue

#         ### ARITHMETICS + - ###

#         pointer_plus_minus = None
#         pointer_type = None

#         for i in range(len(items)):
#             if items[i][1] in ['PLUS', 'MINUS']:
#                 if pointer_plus_minus is None:
#                     pointer_plus_minus = i
#                     pointer_type = items[i][1]
#                     break
        
#         if pointer_plus_minus is not None:
#             operands = [items[:pointer_plus_minus], items[pointer_plus_minus+1:]]
            
#             node_id += 1
#             graph.add_node(str(node_id), token=[(pointer_type, 'FUNC')])
#             node_id += 1
#             graph.add_node(str(node_id), token=operands[0])
#             graph.add_edge(str(node_id-1), str(node_id))
#             node_id += 1
#             graph.add_node(str(node_id), token=operands[1])
#             graph.add_edge(str(node_id-2), str(node_id))

#             items = [(node_id-2, 'FUNC')] + items[pointer_plus_minus+1:]
#             # items = items[:pointer_plus_minus] + [(pointer_type, 'FUNC')] + items[pointer_plus_minus+1:]
#             # branch_id = 0
#             # for m in matched:
#             #     full_branch_id = str(node_id)+'.'+str(branch_id)
#             #     graph.add_node(full_branch_id, token=m)
#             #     parse(m, graph, full_branch_id)
#             #     graph.add_edge(str(node_id), full_branch_id)
#             #     branch_id += 1
            
#             continue
        
#         ### ARITHMETICS * / ###

#         pointer_times_divide = None
#         pointer_type = None

#         for i in range(len(items)):
#             if items[i][1] in ['TIMES', 'DIVIDE']:
#                 pointer_times_divide = i
#                 pointer_type = items[i][1]
        
#         if pointer_times_divide is not None:
#             # matched = [items[:pointer_times_divide], items[pointer_times_divide+1:]]

#             node_id += 1
#             graph.add_node(str(node_id), token=[(pointer_type, 'FUNC')])
#             items = items[:pointer_times_divide] + [(pointer_type, 'FUNC')] + items[pointer_times_divide+1:]
            
#             # branch_id = 0
#             # for m in matched:
#             #     full_branch_id = str(node_id)+'.'+str(branch_id)
#             #     graph.add_node(full_branch_id, token=m)
#             #     parse(m, graph, full_branch_id)
#             #     graph.add_edge(str(node_id), full_branch_id)
#             #     branch_id += 1
            
#             continue

node_id = 0

patterns = {
    'NUMBER':{
        'token':['FUNC', 'EQUALS'],
        'operator':['EQUALS'],
        'operand':['NUMBER']
    },
    'NAME': {
        'token':['FUNC', 'EQUALS'],
        'operator':['EQUALS'],
        'operand':['NAME']
        },
    'FUNC_LSPAREN_FUNC_RSPAREN':{
        'token':['FUNC', 'SPAREN'],
        'operator':['SPAREN'],
        'operand':['FUNC']
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

def parse(items, graph):
    r_max = 100
    r = 0
    global node_id
    while len(items) > 1 and r<r_max:
        items_changed = False
        r += 1
        print('\n', '---parsing---','\n')
        pprint(items)
        print(graph.nodes.data(True))
        print(graph.edges.data(True))
        for pattern, func in patterns.items(): # loop over all patterns
            pattern = pattern.split('_')
            print("\nSearching for {} with operator {}".format(pattern, func['operator']))
            pattern_len = len(pattern)
            # till_end = False
            # while not till_end:
            for i in range(len(items)): # loop over all positions for this pattern
                # print(i, items[i])
                if len(items) - i >= pattern_len:
                    matched = True
                    for j in range(pattern_len): # matching pattern at this position
                        if pattern[j] != items[i+j][0]:
                            matched = False
                            break
                    if matched:
                        print("Found {} at {}".format(pattern, items[i:i+pattern_len]))
                        operands = []
                        for item in items[i:i+pattern_len]:
                            if item[0] in func['operand']:
                                operands.append(item)
                                if item[0] not in ['NAME', 'NUMBER']:
                                    print('adding edge from {} to {}'.format(node_id, item[2]))
                                    graph.add_edge(node_id, item[2])
                        graph.add_node(
                            node_id, 
                            operator=func['operator'],
                            operands=operands
                            )
                        items = items[0:i] + [func['token'][:] + [node_id]] + items[i+pattern_len:]
                        items_changed = True
                        node_id += 1
                        break # items has got a new length, need to start the for loop over again
                
                    # if i == len(items) - 1:
                    #     till_end = True
                    #     print('Search for{} has reached the end of the items.'.format(pattern))
            # print("Finished searching for {} with operator {}".format(pattern, func['operator']))
            if items_changed:
                break
        print('items:')
        pprint(items)
        print('graph:')
        print(graph.nodes.data(True))
        print(graph.edges.data(True))
    # add root node as entry
    graph.add_node('root')
    graph.add_edge('root', items[0][2])
                

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
string_full = 'IF (a + b[ele1]) > (c * 10) THEN INIT(d / e) ELSE f - g'

items = []
s = string_full
while len(s) > 0:
    print(s)
    for type_name, type_regex in (special_symbols | arithmetics | functions | tokens).items():
        m = re.match(pattern=type_regex, string=s)
        if m:
            items.append([type_name, m[0]])
            s = s[m.span()[1]:].strip()
            break

graph = nx.DiGraph()

parse(items, graph)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
labels = {}
labels_operators = nx.get_node_attributes(graph, 'operator')
labels_operands = nx.get_node_attributes(graph, 'operands')
for id, label_operator in labels_operators.items():
    labels[id] = str(id) + '\n' + 'operator:' + str(label_operator) + '\n' + 'operands:' + str(labels_operands[id])
labels['root'] = 'root'
# nx.draw(graph, with_labels=True, node_color='C1')
nx.draw_shell(graph, with_labels=True, labels=labels, node_color='C1')
plt.show()