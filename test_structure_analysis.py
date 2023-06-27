from ASDM.ASDM import Structure
import networkx as nx
import matplotlib.pyplot as plt

model = Structure(from_xmile='BuiltinTestModels/Goal_gap.stmx')
dep_graph = model.generate_dependent_graph()
# nx.draw(dep_graph, with_labels=True, pos=pos)
# plt.show()

nx.nx_agraph.to_agraph(dep_graph).draw('test_dep_graph.png', prog='dot', format='png')