from ASDM import Parser, Solver


if __name__ == '__main__':

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
        # (15, 'h+i-i', {('ele1',): 8}), # this test no longer used as Solver now requires explicit subscript to run
        (16, '10-4-3-2-1', 0)
    ]

    # for test in tests[12:13]:
    for test in tests:
        print('Testing:', test)
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