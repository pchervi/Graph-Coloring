# -*- coding: utf-8 -*-
""" "Coloring_networks_addons.py" file contains a ThinGraph() version of 
Graph() from networks package and a routine to check coloring feasibility
@author: Philippe Chervi"""

import networkx as nx


class ThinGraph(nx.Graph):
    ''' define networkx graph subclass with neither node nor edge attributes
    to save memory'''
    all_edge_dict = {'weight': 1}

    def single_dict(self):
        return self.all_edge_dict

    edge_attr_dict_factory = single_dict
    node_attr_dict_factory = single_dict


def is_coloring_feasible(G, colors):
    ''' assess whether all edge start and end points have different colors'''
    return all(colors[u] != colors[v] for (u, v) in G.edges())
