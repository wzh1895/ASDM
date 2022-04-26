from lxml import etree
from pathlib import Path
import networkx as nx

# citing: https://github.com/JamesPHoughton/pysd/blob/master/pysd/py_backend/xmile/xmile2py.py

class StmxParser(object):
    def __init__(self, path_to_stmx):
        try:
            self.path_to_stmx = Path(path_to_stmx)
        except:
            print('Path to stmx error.')
        
        with open(self.path_to_stmx, 'r') as f:  # read the stmx as Binary so that the from_string() function can follow the xml file's declared encoding.
            self.stmx_content = f.readlines()
            self.stmx_content.pop(0)  # avoid the Unicode string encoding issue
            self.stmx_content[0] = '<xmile>'  # avoid the namespace issue
            self.stmx_content = '\n'.join(self.stmx_content)

        xml_parser = etree.XMLParser(encoding='utf-8', recover=True)
        self.stmx_tree = etree.fromstring(self.stmx_content, parser=xml_parser)

    @staticmethod
    def refine_name(name):
        return name.replace(' ', '_')

    def get_all_nodes(self):
        node_Elements = self.stmx_tree.xpath('/xmile/model/variables/aux')
        nodes = list()
        for node_Element in node_Elements:
            attributes = node_Element.attrib
            name = self.refine_name(attributes['name'])
            nodes.append(name)
        return nodes

    def get_all_coordinates(self, x_streth=1, y_streth=1):
        node_coordinates = dict()
        node_view_Elements = self.stmx_tree.xpath('/xmile/model/views/view/aux')
        for node_view_Element in node_view_Elements:
            attributes = node_view_Element.attrib
            node_coordinates[self.refine_name(attributes['name'])] = (float(attributes['x'])*x_streth, float(attributes['y'])*y_streth)
        return node_coordinates
    
    def get_all_colors(self):
        node_colors = dict()
        node_view_Elements = self.stmx_tree.xpath('/xmile/model/views/view/aux')
        for node_view_Element in node_view_Elements:
            attributes = node_view_Element.attrib
            node_colors[self.refine_name(attributes['name'])] = attributes['color']
        return node_colors

    def get_all_edges(self):
        edge_Elements = self.stmx_tree.xpath('/xmile/model/views/view/connector')
        edges = list()
        for edge_Element in edge_Elements:
            f = self.refine_name(edge_Element.xpath('./from')[0].text)
            t = self.refine_name(edge_Element.xpath('./to')[0].text)
            edges.append((f, t))
        return(edges)

    def get_network_view(self):
        G = nx.DiGraph()
        edges = self.get_all_edges()
        G.add_edges_from(edges)
        return G
