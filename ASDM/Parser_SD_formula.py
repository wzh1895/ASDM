import re

# tokens = {NAME, NUMBER, PLUS}

# Tokens
NAME = r'[a-zA-Z_][a-zA-Z0-9_]*'
NUMBER = r'\d+'

# Special symbols
PLUS = r'\+'
MINUS = r'-'
TIMES = r'\*'
DIVIDE = r'/'
ASSIGN = r'='
LPAREN = r'\('
RPAREN = r'\)'
LSPAREN = r'\['
RSPAREN = r'\]'

PLUS = r'+'

# # Special symbols
# PLUS = r'\+'

import networkx as nx

formula1 = "b = 1"
formula2 = "z = IF a + b[ele1] > c * 10 THEN INIT(d / e) ELSE f - g"

"""
Hierarchy in parsing

- Assignment                 =
- Custom functions           INIT, DELAY(), etc
- Conditional statements     IF THEN ELSE
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

string = 'a+b'

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

def parse_plus(string, graph):
    if re.search(r'+', string):
        p = re.split(r'+', string)
        p = [x.strip() for x in p]
        print(p)
        graph.add


g = nx.DiGraph()
# parse_assign(string, g)
print(g.nodes.data(True))