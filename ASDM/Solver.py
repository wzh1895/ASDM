from scipy import stats

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
            # print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'plus', a,type(a), b, type(b))
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
            # print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'minus', a,type(a), b, type(b))
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

        # if verbose:
        #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'processing node {} on subscript {}:'.format(node_id, subscript))
        #     if type(parsed_equation) is dict:
        #         for k, p in parsed_equation.items():
        #             print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', k, p.nodes(data=True))
        #     else:
        #         print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', parsed_equation.nodes(data=True))
        
        if type(parsed_equation) is dict:  
            # This section is not active; only kept for potential future reference
            # if verbose:
            #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'type of parsed_equation: dict')
            value = dict()
            for sub, sub_equaton in parsed_equation.items():
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'sub', sub)
                value[sub] = self.calculate_node(parsed_equation=sub_equaton, node_id='root', subscript=sub, verbose=verbose)
            if verbose:
                print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v1 Subscripted equation:', value)
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
                value = operands[0][1]
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v2 IS:', value)
            elif operator[0] == 'EQUALS':
                if verbose:
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'operator v3', operator)
                    print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'operands v3', operands)
                if operands[0][0] == 'NAME':
                    if subscript:
                        try:
                            value = self.name_space[operands[0][1]][subscript]
                            # if verbose:
                            #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v3.1.1', value)
                        except TypeError:
                            value = self.name_space[operands[0][1]]
                            # if verbose:
                            #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v3.1.2', value)
                    else:
                        value = self.name_space[operands[0][1]]
                        # if verbose:
                        #     print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v3.1.3', value, operands)
                    if verbose:
                        print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'v3.1 Name:', value)
                elif operands[0][0] == 'FUNC':
                    if verbose:
                        print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'o3')
                    value = self.calculate_node(parsed_equation, node_id, subscript, verbose=verbose)
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
                value = self.calculate_node(parsed_equation, operands[0][2], verbose=verbose)
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
                    v = self.calculate_node(parsed_equation, operand[2], verbose=verbose)
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
                    v = self.calculate_node(parsed_equation, operand[2], verbose=verbose)
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
                        value = self.calculate_node(parsed_equation, operands[0][2], verbose=verbose)
                        self.time_expr_register[tuple(operands[0])] = value
                elif func_name == 'DELAY':
                    # expr value
                    expr_value = self.calculate_node(parsed_equation, operands[0][2], verbose=verbose)
                    if tuple(operands[0]) in self.time_expr_register.keys():
                        self.time_expr_register[tuple(operands[0])].append(expr_value)
                    else:
                        self.time_expr_register[tuple(operands[0])] = [expr_value]
                    
                    # init value
                    if len(operands) == 2: # there's no initial value specified -> use the delyed expr's initial value
                        init_value = self.time_expr_register[tuple(operands[0])][0]
                    elif len(operands) == 3: # there's an initial value specified
                        init_value = self.calculate_node(parsed_equation, operands[2][2], verbose=verbose)
                    else:
                        raise Exception("Invalid initial value for DELAY in operands {}".format(operands))

                    # delay time
                    delay_time = self.calculate_node(parsed_equation, operands[1][2], verbose=verbose)
                    if delay_time > (self.sim_specs['current_time'] - self.sim_specs['initial_time']): # (- initial_time) because simulation might not start from time 0
                        value = init_value
                    else:
                        delay_steps = delay_time / self.sim_specs['dt']
                        value = self.time_expr_register[tuple(operands[0])][-int(delay_steps+1)]
                elif func_name == 'DELAY1':
                    # args values
                    order = 1
                    expr_value = self.calculate_node(parsed_equation, operands[0][2], verbose=verbose)
                    delay_time = self.calculate_node(parsed_equation, operands[1][2], verbose=verbose)

                    if len(operands) == 3:
                        init_value = self.calculate_node(parsed_equation, operands[2][2], verbose=verbose)
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
                    expr_value = self.calculate_node(parsed_equation, operands[0][2], verbose=verbose)
                    delay_time = self.calculate_node(parsed_equation, operands[1][2], verbose=verbose)
                    if len(operands) == 3:
                        init_value = self.calculate_node(parsed_equation, operands[2][2], verbose=verbose)
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
                    expr_value = self.calculate_node(parsed_equation, operands[0][2], verbose=verbose)
                    if tuple(operands[0]) in self.time_expr_register.keys():
                        self.time_expr_register[tuple(operands[0])].append(expr_value)
                    else:
                        self.time_expr_register[tuple(operands[0])] = [expr_value]
                    
                    # historical time
                    historical_time = self.calculate_node(parsed_equation, operands[1][2], verbose=verbose)
                    if historical_time > self.sim_specs['current_time']:
                        value = 0
                    else:
                        historical_steps = (historical_time - self.sim_specs['initial_time']) / self.sim_specs['dt']
                        value = self.time_expr_register[tuple(operands[0])][int(historical_steps)]
                elif func_name == 'SMTH1':
                    # arg values
                    order = 1
                    expr_value = self.calculate_node(parsed_equation, operands[0][2], verbose=verbose)
                    smth_time = self.calculate_node(parsed_equation, operands[1][2], verbose=verbose)
                    if len(operands) == 3:
                        init_value = self.calculate_node(parsed_equation, operands[2][2], verbose=verbose)
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
                    expr_value = self.calculate_node(parsed_equation, operands[0][2], verbose=verbose)
                    smth_time = self.calculate_node(parsed_equation, operands[1][2], verbose=verbose)
                    if len(operands) == 3:
                        init_value = self.calculate_node(parsed_equation, operands[2][2], verbose=verbose)
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
                    input_value = self.calculate_node(parsed_equation, operands[1][2], verbose=verbose)
                    value = look_up_func(input_value)
                else:
                    raise Exception('Unknown Lookup function {}'.format(operator[0]))
            else:
                raise Exception('Unknown operator {}'.format(operator[0]))
        
        self.id_level -= 1
        
        if verbose:
            print('\t'*self.id_level+self.HEAD+' [ '+var_name+' ] ', 'value for {} {}'.format(node_id, subscript), value)

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
        (15, 'h+i-i', {('ele1',): 8}),
        (16, '10-4-3-2-1', 0)
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
            
            # print('{:2} formula: {:30}'.format(n, formula))
            solver = Solver(name_space=name_space)
            outcome = solver.calculate_node(graph)
            # if outcome != result:
            # print('   outcome: {:<20} type: {:20} {:2} {:5}'.format(str(outcome), str(type(outcome)), str(n), str(outcome==result)))