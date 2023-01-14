class Solver(object):
    def __init__(self, dimension_elements=None, name_space=None, var_history=None):
        def greater_than(a, b):
            if a > b:
                return True
            elif a <= b:
                return False
            else:
                raise Exception

        def less_than(a, b):
            if a < b:
                return True
            elif a >= b:
                return False
            else:
                raise Exception

        def plus(a, b):
            # print(self.HEAD, 'plus', a,type(a), b, type(b))
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
            # print(self.HEAD, 'minus', a,type(a), b, type(b))
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
                    # print(self.HEAD, 'a/b', a, b)
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

        def con(a, b, c):
            if a:
                return b
            else:
                return c
        
        self.dimension_elements = dimension_elements
        self.name_space = name_space
        self.var_history = var_history

        self.functions = {
                'GT': greater_than,
                'LT': less_than,
                'PLUS': plus,
                'MINUS': minus,
                'TIMES': times,
                'DIVIDE': divide,
                'CON': con,
            }
        
        # self.id_level = 0

        self.HEAD = "SOLVER"

    def calculate_node(self, parsed_equation, node_id='root', subscript=None):
        # self.id_level += 1

        # print(self.HEAD, 'processing node {} on subscript {}:'.format(node_id, subscript))
        # if type(parsed_equation) is dict:
        #     for k, p in parsed_equation.items():
        #         print(self.HEAD, k, p.nodes(data=True))
        # else:
        #     print(self.HEAD, parsed_equation.nodes(data=True))
        
        if type(parsed_equation) is dict:
            # print(self.HEAD, 'type of parsed_equation: dict')
            value = dict()
            for sub, sub_equaton in parsed_equation.items():
                # print(self.HEAD, 'sub', sub)
                value[sub] = self.calculate_node(parsed_equation=sub_equaton, node_id='root', subscript=sub)
        else:
            # print(self.HEAD, 'type of parsed_equation: graph')
            if node_id == 'root':
                node_id = list(parsed_equation.successors('root'))[0]
            node = parsed_equation.nodes[node_id]
            # print(self.HEAD, 'node:', node_id, node)
            operator = node['operator']
            operands = node['operands']
            if operator[0] == 'IS':
                # print(self.HEAD, 'oprt1')
                # print(self.HEAD, 'o1')
                value = operands[0][1]
            elif operator[0] == 'EQUALS':
                if operands[0][0] == 'NAME':
                    if subscript:
                        try:
                            value = self.name_space[operands[0][1]][subscript]
                        except TypeError:
                            value = self.name_space[operands[0][1]]
                    else:
                        value = self.name_space[operands[0][1]]
                elif operands[0][0] == 'FUNC':
                    # print(self.HEAD, 'o3')
                    value = self.calculate_node(parsed_equation, node_id, subscript)
                else:
                    # print(self.HEAD, 'o4')
                    raise Exception()
            
            elif operator[0] == 'SPAREN': # TODO this part is too dynamic, therefore can be slow. Need to resolve this when compiling.
                var_name = operands[0][1]
                # print('a1', var_name, subscript)
                if len(operands) == 1: # only var_name; no subscript is specified
                    # print('a1.1')
                    # this could be 
                    # (1) this variable (var_name) is not subscripted therefore the only value of it should be used;
                    # (2) this variable (var_name) is subscripted in the same way as the variable using it (a contextual info is needed and provided in the arg subscript)
                    if subscript:
                        # print('a1.1.1')
                        value = self.name_space[var_name][subscript] 
                    else:
                        # print('a1.1.2')
                        value = self.name_space[var_name]
                else: # there are explicitly specified subscripts in oprands
                    # print('a1.2')
                    if subscript: # subscript is explicitly specified; like a[Element_1] or a[Dimension_1]
                        # print('a1.2.0', subscript, self.dimension_elements)
                        subscript_from_operands = list()
                        operands_containing_subscript = operands[1:]
                        for i in range(len(operands_containing_subscript)):
                            if operands_containing_subscript[i][1] in self.dimension_elements.keys(): # it's sth like Dimension_1
                                # print('a1.2.0.1')
                                subscript_from_operands.append(subscript[i]) # take the element from arg subscript in the same position to replace Dimension_1
                            else: # it's sth like Element_1
                                # print('a1.2.0.2')
                                subscript_from_operands.append(operands_containing_subscript[i][1]) # add to list directly
                        subscript_from_operands = tuple(subscript_from_operands)
                        # print('a1.2.1', subscript_from_operands)
                        value = self.name_space[var_name][subscript_from_operands] # try if subscript is Element_1

                    else: # subscript is not explicitly specified
                        # print('a1.2.2')
                        # like a[Dimension1] -> Which element of Dimension1 to use, 
                        # depends on the other variable.

                        subscript = tuple(operand[1] for operand in operands[1:]) # use tuple to make it hashable
                        # print(self.HEAD, 'subscripts', subscripts, 'subscript of interest', subscript)
                        value = self.name_space[var_name][subscript]

            
            elif operator[0] == 'PAREN':
                value = self.calculate_node(parsed_equation, operands[0][2])
            
            elif operator[0] in self.functions.keys():
                # print(self.HEAD, 'operator', operator)
                func_name = operator[0]
                function = self.functions[func_name]
                oprds = []
                for operand in operands:
                    # print(self.HEAD, 'oprd', operand)
                    v = self.calculate_node(parsed_equation, operand[2])
                    # print(self.HEAD, 'value', v)
                    oprds.append(v)
                # print(self.HEAD, 'oprds', oprds)
                value = function(*oprds)
            else:
                raise Exception('Unknown operator {}'.format(operator[0]))
        
        # print(self.HEAD, 'value for {} {}'.format(node_id, subscript), value)
        # self.id_level -= 1
        try:
            return value[subscript]
        except KeyError as e: # due to dict[None]
            return value
        except TypeError as e: # due to float[sub]
            return value


if __name__ == '__main__':
    from Parser import Parser

    name_space= {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5,
        'f': 6,
        'g': 7,
        'h': {('ele1',):8},
        'aa': True,
        'ba': False,
        'bb': 0,
        'bc': 1,
        'ac': 2,
        'cc': 3,
        'i' : {('ele1',):9}
    }

    tests = [
        (1, 'a', 1),
        (2, 'a+b-c', 0),
        (3, 'a+b-2', 1),
        (4, 'a*b', 2),
        (5, 'a/2', 0.5),
        (6, 'INIT(a)', None),
        (7, 'DELAY(a, 1)', None),
        (8, 'a*((b+c)-d)', 1),
        (9, 'a > b', False),
        (10, 'a < 2', True),
        (11, 'IF aa THEN bb ELSE cc', 0),
        (12, 'IF aa THEN IF ba THEN bb ELSE bc ELSE ac', 1),
        (13, 'h[ele1]', 8),
        (14, 'IF (a + h[ele1]) > (c * 10) THEN INIT(d / e) ELSE f - g', None),
        (15, 'h+i-i', {('ele1',): 8})
    ]

    # for test in tests[12:13]:
    for test in tests:
        if test[2] is not None:
            n = test[0]
            formula = test[1]
            result = test[2]
            parser = Parser()
            graph = parser.parse(formula)
            
            # print('graph_nodes', graph.nodes(data=True))
            # print('graph_edges', graph.edges())
            
            # fig, ax = plt.subplots()
            # labels = {}
            # labels_operators = nx.get_node_attributes(graph, 'operator')
            # labels_operands = nx.get_node_attributes(graph, 'operands')
            # for id, label_operator in labels_operators.items():
            #     labels[id] = str(id) + '\n' + 'operator:' + str(label_operator) + '\n' + 'operands:' + str(labels_operands[id])
            # labels['root'] = 'root'
            # nx.draw_shell(graph, with_labels=True, labels=labels, node_color='C1')
            # plt.show()
            
            print('{:2} formula: {:30}'.format(n, formula))
            solver = Solver(name_space=name_space)
            outcome = solver.calculate_node(graph)
            # if outcome != result:
            print('   outcome: {:<20} type: {:20} {:2} {:5}'.format(str(outcome), str(type(outcome)), str(n), str(outcome==result)))