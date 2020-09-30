# -*- coding: utf-8 -*-
""" "Coloring_Tabucol.py" file contains a Tabucol algorithm to color
a graph (vertex coloring) with k colors
@author: Philippe Chervi"""

from random import randint
from itertools import combinations, chain
import networkx as nx
from networkx.algorithms.approximation import max_clique
from Coloring_networkx_addons import ThinGraph, is_coloring_feasible
from timeit import default_timer


def Tabucol_opt(G, k, C_L=6, C_lambda=0.6, C_maxiter=100000, verbose=False):
    '''Tabucol_opt provides the graph coloring with the smallest number of
    colors'''
    assert len(G) > 0
    assert k > 0 and k <= len(G)
    best_colors, best_niter = {}, 0
    # compute length of max clique as number of colors cannot be less
    length_max_clique = len(max_clique(G))
    if k < length_max_clique:
        print('Tabucol_opt cannot find a solution with less colors than length of maximum clique ', length_max_clique)
    else:
        ncolors = k
        while ncolors >= length_max_clique:
            colors, niter = Tabucol(G, ncolors, C_L, C_lambda, C_maxiter)
            if is_coloring_feasible(G, colors):
                best_colors = dict(colors)
                best_niter = niter
                ncolors = max(colors.values())
                if verbose:
                    print('Tabucol_opt found a solution with %d colors' % (max(colors.values())+1))
            else:
                if not best_colors:
                    print('Tabucol_opt did not find a solution with %d colors' % k)
                break

    return best_colors, best_niter


def Tabucol(G, k, C_L=6, C_lambda=0.6, C_maxiter=100000, verbose=False):
    '''Tabucol provides a vertex k-coloring if such coloring is found
    using a tabu search scheme. Tabucol features are inspired from
    "A survey of local search methods for graph coloring" by Philippe Galiniera
    and Alain Hertzb" '''

    def is_tabu_allowed(tabu_d, nc, n_iter):
        ''' assess whether candidate is in tabu dictionary'''
        return (nc not in tabu_d) or (nc in tabu_d and tabu_d[nc] < n_iter)

    # generate random coloring and compute color classes
    colors = {i: randint(0, k-1) for i in G.nodes()}
    color_classes = {col: set(i for i in G.nodes()
                     if colors[i] == col) for col in range(k)}
    # compute actual violations and number of violated edges as F
    viol_edges = {col: [edge for edge in combinations(col_set, 2)
                        if edge in G.edges()]
                  for col, col_set in color_classes.items()}
    #  viol_nodes contains nodes involved in violated edges, a node
    #  being counted as many times it appears in viol_edges
    viol_nodes = {col: list(chain.from_iterable(viol_edges[col]))
                  for col in color_classes.keys()}
    # F is the total number of violations (violated edges)
    F = sum(len(v) for v in viol_edges.values())

    # initiate local search in 1-move neighborhood
    # create tabu dictionary with (node, col) as key and niter as value
    tabu = {}
    niter = 0
    restrictive = False
    while F > 0 and niter < C_maxiter:

        # generate candidates with violation variation
        delta = {}
        for col, node_list in viol_nodes.items():
            for node in node_list:
                old_count = viol_nodes[col].count(node)
                for col_cand in range(k):
                    nc = (node, col_cand)
                    if col_cand != col and is_tabu_allowed(tabu, nc, niter):
                        new_count = len(set(G[node]).intersection(color_classes[col_cand]))
                        delta[nc] = new_count-old_count
        if not delta:
            # skip current iteration
            if not restrictive:
                if verbose:
                    print('tabu scheme is probably too restrictive')
                restrictive = True
        else:
            # select a candidate among the ones with lowest delta
            delta_c = min(delta.values())
            final_cand = [(n, c) for (n, c), value in delta.items()
                          if value == delta_c]
            # choose a candidate with lowest value at random
            (node_c, col_c) = final_cand[randint(0, len(final_cand)-1)]
            # update tabu dictionary
            deleting_tabu_list = [key_t for key_t, iter_t in tabu.items()
                                  if iter_t <= niter]
            for key_t in deleting_tabu_list:
                del tabu[key_t]
            old_col = colors[node_c]
            tabu[(node_c, old_col)] = niter + int(C_L + C_lambda*F)
            # update number of violations
            F += delta_c
            # update modified violation edge and node classes
            viol_edges[old_col] = [edge for edge in viol_edges[old_col]
                                   if node_c not in edge]
            viol_nodes[old_col] = list(chain.from_iterable(viol_edges[old_col]))
            new_viol_edges = ((node_c, u) for u in set(G[node_c]).intersection(color_classes[col_c]))
            viol_edges[col_c].extend(new_viol_edges)
            viol_nodes[col_c] = list(chain.from_iterable(viol_edges[col_c]))
            # update colors and color classes
            colors[node_c] = col_c
            color_classes[old_col] -= {node_c}
            color_classes[col_c] = color_classes[col_c].union({node_c})

        niter += 1

    if verbose and niter == C_maxiter:
        print('exiting loop as max iterations exceeded')

    # get rid of empty color classes to speed up next search
    if not all(col_class for col_class in color_classes.values()):
        colors = {}
        count = 0
        for col_class in color_classes.values():
            if col_class:
                for n in col_class:
                    colors[n] = count
                count += 1

    return colors, niter


def read_test_data(file_location):
    ''' file_location file features a graph represented by its node and edge
    counts in first line, and then edges as two integers per line'''
    print("reading test data file ", file_location)
    # open file
    input_data_file = open(file_location, 'r')
    input_data = input_data_file.read()
    #  read number of nodes and edges
    lines = input_data.split('\n')
    part0, part1 = lines[0].split()
    node_count = int(part0)
    edge_count = int(part1)
    # create graph, add nodes, and edges from edge list
    G = ThinGraph()
    G.add_nodes_from(range(node_count))
    edge_list = []
    for i in range(1, edge_count + 1):
        u, v = lines[i].split()
        edge_list.append((int(u), int(v)))
    G.add_edges_from(edge_list)
    assert len(edge_list) == edge_count
    return G


if __name__ == "__main__":
    sols = []
    for ntest in range(1, 9):
        # chromatic number of mycielski_graph(n) is n
        G = nx.mycielski_graph(ntest)
        colors = {}
        start_time = default_timer()
        colors, niter = Tabucol_opt(G,
                                    len(G),
                                    C_L=6,
                                    C_lambda=0.6,
                                    C_maxiter=100000,
                                    verbose=True)
        elapsed = default_timer() - start_time
        sol = ['mycielski_graph', ntest, max(colors.values())+1, niter, elapsed]
        print(sol)
        sols.append(sol)
        del G
    print("\n")
    for sol in sols:
        print(sol)
