# -*- coding: utf-8 -*-
""" "Coloring_main.py" explores an exact means of solving graph (vertex) coloring using PySCIPOpt,
a Python wrapper for SCIP, a prior 'test_coloring.py' example and a column
generation technique.
Key words; SCIP, PySCIPOpt, Graph Coloring, column generation, Tabucol
@author: Philippe CHERVI"""

try:
    import networkx as nx
    from networkx.algorithms.approximation import max_clique
    from itertools import combinations
    from Coloring_MWIS_heuristics import greedy_MWIS, exact_MWIS
    from Coloring_Tabucol import Tabucol_opt
    from Coloring_networks_addons import ThinGraph, is_coloring_feasible
    from pyscipopt import Model, Branchrule, Conshdlr, Pricer, SCIP_RESULT
    from pyscipopt import SCIP_LPSOLSTAT, SCIP_PARAMSETTING, SCIP_PROPTIMING
    import inspect
    # debugging magic
    entering = lambda: print("\nIN ", inspect.stack()[1][3])
    leaving = lambda: print("OUT ", inspect.stack()[1][3])
except:
    import pytest
    pytest.skip()

EPS = 1e-6
MyFile = open("test_data.dat", "w")


class unique_numbers:
    '''create a unique number for selected nodes in branchexeclp and
    branchexecps as varsets are uniquely identified by this number
    (during B&B, a specific conshdlr node is thus uniquely identified)'''

    def __iter__(self):
        self.a = 1
        return self

    def __next__(self):
        x = self.a
        self.a += 1
        return x


##### THIS WOULD BE THE EQUIVALENT OF PROBDATA


class Coloring(Model):
    def __init__(self, graph=None):
        entering()
        super().__init__(defaultPlugins=True)
        print("scip is initiated, continue")
        if graph:
            self.originalgraph = graph
            # graph is the preprocessed original graph
            self.graph = graph.copy()
        # self.varsets is a dictionary with unique var index as key,
        # and ordered tuple varset as value
        self.varsets = {}
        self.conss = {}
        # create class for unique numbers (used in branching)
        self.unique_number_iter = iter(unique_numbers())
        leaving()

    def is_integral_sol_feasible(self, transformed=False):
        '''check basis variable for integrality and basis sets for
        independence'''
        #  check if basis has integral values
        if self.is_lp_integral(transformed):
            # generate sets corresponding to basis variables
            indep_set_iter = (self.varsets[v.getIndex()]
                              for v in self.getVars(transformed)
                              if self.isFeasEQ(self.getVal(v), 1.0))
            #  check if such sets are independent sets
            for indep_set in indep_set_iter:
                if any(edge in self.originalgraph.edges()
                       for edge in combinations(indep_set, 2)):
                    return False
        else:
            return False
        return True

    def is_lp_integral(self, transformed=False):
        '''check that all LP variables are integral'''
        return all(self.isFeasIntegral(self.getVal(v))
                   for v in self.getVars(transformed))        

    def preprocess_graph(self):
        '''remove nodes with fewer degrees than the degree of a maximal
        clique and from dominated sets'''
        entering()

        G = self.originalgraph.copy()
        # compute maximum clique from networkx.algorithms.approximation
        maxclique = max_clique(G)
        len_maxclique = len(maxclique)

        pruned_nodes = []
        while True:
            nnodes_at_start = len(G)
            # iteration to remove nodes with degree < |max_clique|
            partial_node_list = [n for (n, deg) in G.degree()
                                 if n not in maxclique and deg < len_maxclique]
            if partial_node_list:
                G.remove_nodes_from(partial_node_list)
                pruned_nodes.extend(partial_node_list)
            # iteration to remove nodes from dominated neighborhoods
            partial_node_list = []
            for u in (u for u in G.nodes() if u not in maxclique):
                for v in G.nodes():
                    if u == v or G.has_edge(u, v):
                        continue
                    if set(G[u]).issubset(set(G[v])):
                        partial_node_list.append(u)
                        break
            if partial_node_list:
                G.remove_nodes_from(partial_node_list)
                pruned_nodes.extend(partial_node_list)
            # repeat whenever there is improvement
            if nnodes_at_start == len(G):
                break
        # update working graph and keep pruned nodes for post processing
        self.graph.remove_nodes_from(pruned_nodes)
        self.pruned_nodes = pruned_nodes[:]
        if self.pruned_nodes:
            print("pruned nodes are:", pruned_nodes)
            self.graph = G.copy()
            del G
        else:
            print("no nodes were pruned at pre-processing stage")
        leaving()

    # color nodes which were removed at preprocessing stage
    def postprocess_graph(self):
        '''re-insert pruned nodes into final color scheme'''
        if self.pruned_nodes:
            # compute color of already colored nodes
            color = {}
            for col, indep_set in self.best_integral_sol.items():
                for node in indep_set:
                    color[node] = col
            # compute color of pruned nodes based their on coloring choice
            color_neighbors = {}
            maxcolor = len(self.best_integral_sol)
            for pruned_node in self.pruned_nodes:
                color_set = set(range(maxcolor))
                for neigh in self.originalgraph[pruned_node]:
                    if neigh in color.keys():
                        color_set -= {color[neigh]}
                color_neighbors[pruned_node] = color_set
            # color with minimum color based on available neighbors' colors
            while color_neighbors:
                color_set = set(range(maxcolor))
                node_with_min_colors, col_set = min(color_neighbors.items(),
                                                    key=lambda x: len(x[1]))
                color_chosen = min(col_set)
                color[node_with_min_colors] = color_chosen
                # update color_neighbors
                for u in self.originalgraph[node_with_min_colors]:
                    if u in self.pruned_nodes and u in color_neighbors:
                        color_neighbors[u] -= {color_chosen}
                del color_neighbors[node_with_min_colors]
            # update best integral solution
            for pruned_node in self.pruned_nodes:
                col = color[pruned_node]
                self.best_integral_sol[col] += (pruned_node, )
            # sort tuples of best integral solution
            for i, indep_set in self.best_integral_sol.items():
                self.best_integral_sol[i] = tuple(sorted(indep_set))
            #  final checks
            assert len(list(self.originalgraph)) == sum(len(s) for s in self.best_integral_sol.values())
            assert self.is_integral_sol_feasible(transformed=True) is True

    def set_up_constraints(self):
        '''set up initial node constraints based on a greedy solution of
        independent sets refined by a Tabucol heuristic'''
        from operator import itemgetter
        entering()

        G = self.graph
        node_list = list(G.degree)
        node_list.sort(key=itemgetter(1), reverse=True)
        pi_list = list(map(itemgetter(0), node_list))
        self.best_integral_sol = {}
        # greedy heuristic to find initial solution
        while pi_list:
            indep_set = (pi_list[0],)
            currentNumber = len(self.best_integral_sol)
            for e in pi_list[1:]:
                # additional edge must not be connected to current indep_set
                feasible_iter = ((c, e) not in G.edges for c in indep_set)
                if all(feasible_iter):
                    indep_set += (e,)
            self.best_integral_sol[currentNumber] = tuple(sorted(indep_set))
            for e in indep_set:
                pi_list.remove(e)
        # check all nodes are accounted for
        assert len(list(G)) == sum(len(s) for s in self.best_integral_sol.values())
        print("Greedy solution found with %d colors" % len(self.best_integral_sol))

        # use number of colors found as initial Tabucol number
        ncolors = len(self.best_integral_sol)
        colors, _ = Tabucol_opt(G, ncolors, verbose=True)
        new_ncolors = max(colors.values())+1
        if new_ncolors < ncolors:
            self.best_integral_sol = {}
            for col in range(new_ncolors):
                set_with_this_col = (i for i, c in colors.items() if c == col)
                self.best_integral_sol[col] = tuple(sorted(set_with_this_col))
            print("Final Tabucol solution has %d colors" % new_ncolors)
        else:
            print("Tabucol did not improve greedy solution")

        # # keep best initial integral solution found so far
        self.initial_sol = self.best_integral_sol.copy()
        # define variables and constraints
        for i, initial_set in self.initial_sol.items():
            var = self.addVar(vtype='B',
                              name="Initial_Set_" + str(i),
                              obj=1.0)
            self.varsets[var.getIndex()] = initial_set
            for n in initial_set:
                self.conss[n] = self.addCons(var >= 1,
                    name = "Node_Constraint_" + str(n),
                    initial=True, separate=False, enforce=True, check=True,
                    propagate=True, local=False, modifiable=True, dynamic=True,
                    removable=False, stickingatnode=False)
        # help scip with integrality
        self.setObjIntegral()
        leaving()

########### END THE PROBDATA
############ THE BRANCHING RULE


class ColoringBranch(Branchrule):

    def find_nodes_to_branch_lp(self, scip, lpcands, ordered_cands):
        '''find suitable nodes for branchexeclp'''
        graph = getCurrentGraph(scip)
        for s1_num, s2_num in combinations(ordered_cands, 2):
            s1 = lpcands[s1_num]
            set_tuple1 = scip.varsets[s1.getIndex()]
            s2 = lpcands[s2_num]
            set1 = set(scip.varsets[s1.getIndex()])
            set2 = set(scip.varsets[s2.getIndex()])
            # node1 in set1 and set2, node2 in set1 \ set2
            for node1 in set1.intersection(set2):
                for node2 in set1.difference(set2):
                    if (node1, node2) not in graph.edges():
                        return node1, node2
            # node2 in set1 and set2, node1 in set2 \ set1
            for node2 in set2.intersection(set1):
                for node1 in set2.difference(set1):
                    if (node2, node1) not in graph.edges():
                        return node2, node1
        raise ValueError("Should not be here!")

    def branchexeclp(self, allowaddcons):
        '''branch on suitable nodes'''
        # entering()
        scip = self.model
        # lpcands, lpcandssol, lpcandsfrac, nlpcands, npriolpcands, nfracimplvars = scip.getLPBranchCands()
        lpcands, _, fracvals, _, _, _ = scip.getLPBranchCands()
        assert len(lpcands) > 0
        assert len(scip.varsets) > 0

        # get least fractional candidate
        fractionalities = list(map(lambda x: min(x, 1-x), fracvals))
        # find nodes to branch based on fractional value sets
        ordered_cands = sorted(range(len(lpcands)), key = lambda x: fractionalities[x])
        node1, node2 = self.find_nodes_to_branch_lp(scip, lpcands, ordered_cands)
        if node1 > node2:
            node1, node2 = node2, node1
        # print(">>>>>>>>>>>>>>.. found nodes to branch on! ", node1, node2)

        # assert that nodes are not connected
        # assert not current_graph.has_edge(node1, node2)

        # create children
        estimate = scip.getLocalEstimate()
        childsame = scip.createChild(0.0, estimate)
        childdiff = scip.createChild(0.0, estimate)
        # print("children created!")

        # create constraints
        # get current constraint: Python's black magic
        parent_cons = getCurrentCons(scip)
        counter_global = next(scip.unique_number_iter)
        str_nodes = str(node1) + "_" + str(node2) + "_" + str(counter_global)
        conssame = scip.conshdlr.createCons("same_" + str_nodes, node1, node2, parent_cons, "same", childsame)
        consdiff = scip.conshdlr.createCons("differ_" + str_nodes, node1, node2, parent_cons, "differ", childdiff)
        # print("constraints created!")
        # add constraints to node
        scip.addConsNode(childsame, conssame)
        scip.addConsNode(childdiff, consdiff)
        # print("constraints added!")

        # leaving()
        return {'result': SCIP_RESULT.BRANCHED}

    def find_nodes_to_branch_ps(self, scip, graph):
        '''find suitable nodes for branchexecp'''
        # create current_set dictionary with non-zero LP variables (even fractional)
        current_set = {}
        for myvar in scip.getVars(transformed=True):
            myset = scip.varsets[myvar.getIndex()]
            currentNumber = len(current_set)
            if myvar.isInLP() and not scip.isZero(scip.getVal(myvar)):
                current_set[currentNumber] = myset
        # select two nodes in different sets with no edge between them
        for set_tuple1, set_tuple2 in combinations(current_set.values(), 2):
            for n1 in set_tuple1:
                for n2 in set_tuple2:
                    if not graph.has_edge(n1, n2):
                        return n1, n2
        raise ValueError("Should not be here!")

    def branchexecps(self, allowaddcons):
        '''branch on suitable nodes for infeasible problem'''
        entering()

        scip = self.model
        # get current graph: Python's black magic
        current_graph = getCurrentGraph(scip)
        node1, node2 = self.find_nodes_to_branch_ps(scip, current_graph)
        if node1 > node2:
            node1, node2 = node2, node1
        # print(">>>>>>>>>>>>>>.. found nodes to branch on! ", node1, node2)

        estimate = scip.getLocalEstimate()
        childsame = scip.createChild(0.0, estimate)
        childdiff = scip.createChild(0.0, estimate)
        # print("children created!")

        # create constraints
        parent_cons = getCurrentCons(scip)
        counter_global = next(scip.unique_number_iter)
        str_nodes = str(node1) + "_" + str(node2) + "_" + str(counter_global)
        conssame = scip.conshdlr.createCons("same_" + str_nodes, node1, node2, parent_cons, "same", childsame)
        consdiff = scip.conshdlr.createCons("differ_" + str_nodes, node1, node2, parent_cons, "differ", childdiff)
        # print("constraints created!")
        # add constraints to node - what about releaseCons() ?
        scip.addConsNode(childsame, conssame)
        scip.addConsNode(childdiff, consdiff)
        # print("constraints added!")

        leaving()
        return {'result': SCIP_RESULT.BRANCHED}

########### END BRANCHING
############ THE CONSHDLR FOR STOREGRAPH CONSTRAINTS


def is_coloring_feasible(G, colors):
    ''' assess whether edge end points have not the same color'''
    return all(colors[u] != colors[v] for (u, v) in G.edges())


def getCurrentGraph(scip):
    '''get current graph in conshdlr stack'''
    conshdlr = scip.conshdlr
    assert len(conshdlr.stack) > 0
    cons = conshdlr.stack[-1]
    assert cons is not None and cons.name is not None
    consData = conshdlr.consDatas[cons.name]
    assert consData is not None
    graph = consData['current_graph']
    assert graph is not None
    return graph


def getCurrentCons(scip):
    '''get current conshdlr constraint from conshdlr stack'''
    conshdlr = scip.conshdlr
    assert len(conshdlr.stack) > 0
    cons = conshdlr.stack[-1]
    assert cons is not None and cons.name is not None
    consData = conshdlr.consDatas[cons.name]
    assert consData is not None
    return cons


class StoreGraphConshdlr(Conshdlr):
    '''StoreGraphConshdlr contraint handler'''
    
    def __init__(self):
        '''local variables for contraint handler'''
        entering()
        # self.consDatas is a dict of dict for constraint data, referencing 
        # consData[name] with name being the unique cons.name
        self.consDatas = {}
        self.stack = []
        leaving()

    def createCons(self, name, node1, node2, parent_cons, current_type, stickingnode):
        '''create constraint for B&B nodes, distinguishing root from others'''
        # create a bare root constraint
        if current_type == "root":
            cons = self.model.createCons(self, name,
                   initial=False, separate=False, enforce=False, check=False,
                   propagate=False, local=True, modifiable=False, dynamic=True,
                   removable=False, stickingatnode=False)
            self.consDatas[name] = {}
            consData = self.consDatas[name]
            consData['created'] = True
            consData['current_graph'] = self.model.graph
            consData['type'] = current_type
        # create a full-fledged non-root constraint (all dict components)
        else:
            cons = self.model.createCons(self, name,
                   initial=False, separate=False, enforce=False, check=False,
                   propagate=True, local=True, modifiable=False, dynamic=True,
                   removable=False, stickingatnode=True)
            self.consDatas[name] = {}
            consData = self.consDatas[name]
            consData['created'] = False
            consData['type'] = current_type
            consData['parent_cons'] = parent_cons
            consData['node1'] = node1
            consData['node2'] = node2
            consData['npropagatedvars'] = 0
            consData['stickingatnode'] = stickingnode
        return cons

    def consinitsol(self, constraints):
        '''B&B is going to start now, so we create the constraint containing
        the graph of the root node'''
        entering()
        # reset propagation parameter for non-root nodes
        self.model.setIntParam("propagating/maxrounds", 100)
        # create root constraint
        cons = self.createCons("root", -1, -1, None, "root", None)
        self.stack.append(cons)
        assert len(self.stack) == 1
        leaving()

    def consexitsol(self, constraints, restart):
        '''check stack has only the root constraint'''
        entering()
        assert len(self.stack) == 1
        self.stack[0] = None
        leaving()

    def consdelete(self, constraint):
        '''delete constraint data'''
        entering()
        assert constraint is not None
        consData = self.consDatas[constraint.name]
        assert consData is not None
        del consData
        leaving()

    def consfree(self):
        '''delete local variables of constraint handler '''
        entering()
        assert self.consDatas is not None
        del self.consDatas
        assert self.stack is not None
        del self.stack
        leaving()

    # consactive: [more info in FAQ of SCIP]
    # In general, we are adding constraints to nodes. Every time the node is entered, the consactive callback
    # of the constraints added to the node are called. So here we do
    # - if entered for the first time, create the data of the constraint (graph, etc)
    # - if re-entered, check if new vars were generated in between calls. We have to repropagate the node in the affirmative case
    # - place the constraint on top of the stack (to know which constraint is the current active constraint, i.e. to know the
    #   current graph)
    def consactive(self, constraint):
        # entering()
        assert constraint is not None and constraint.name is not None
        consData = self.consDatas[constraint.name]
        assert consData is not None

        self.stack.append(constraint)
        assert consData['type'] == "root" or consData['parent_cons'] is not None

        current_type = consData['type']
        if not consData['created']:
            node1 = consData['node1']
            node2 = consData['node2']
            assert str(node1) in constraint.name and str(node2) in constraint.name
            # get constraint's parent_graph
            parent_cons = consData['parent_cons']
            parentconsData = self.consDatas[parent_cons.name]
            parent_graph = parentconsData['current_graph']
            
            if parent_graph.has_edge(node1, node2):
                print(node1, node2)
                print(constraint.name)
                print(consData)
            
            assert not parent_graph.has_edge(node1, node2)
            # add edges to graph, depending on B&B node type
            if current_type == "same":
                consData['current_graph'] = parent_graph.copy()
                assert not consData['current_graph'].has_edge(node1, node2)
                # add (node1, w) edges for w in the neighborhood of node2
                consData['current_graph'].add_edges_from(((node1, w) for w in parent_graph[node2] if not parent_graph.has_edge(node1, w)))
                # add (node2, w) edges for w in the neighborhood of node1
                consData['current_graph'].add_edges_from([(node2, w) for w in parent_graph[node1] if not parent_graph.has_edge(node2, w)])
                # assert not consData['current_graph'].has_edge(node1, node2)
            elif current_type == "differ":
                # add (node1, node2) edge
                consData['current_graph'] = parent_graph.copy()
                consData['current_graph'].add_edge(node1, node2)
            else:
                raise ValueError("type %s unkonwn" % current_type)
            consData['created'] = True
        else:
            if current_type != "root" and consData['npropagatedvars'] < len(self.model.varsets) :
                self.model.repropagateNode(consData['stickingatnode'])
        # leaving()

    def consdeactive(self, constraint):
        '''remove the constraint from the stack'''
        # entering()
        assert constraint is not None
        consData = self.consDatas[constraint.name]
        assert consData is not None
        assert len(self.stack) > 0
        current_cons = self.stack.pop()
        assert id(constraint) == id(current_cons)
        # leaving()

    # propagation: we have to set to 0 all variable associated to invalid stable sets
    # we *only* need to check whether the stable sets contain node1 and node2 when they shouldn't
    # or don't contain both when they should (i.e. we do not have to check whether the stable set
    # is valid in the quotient graph, because all unfixed variables represent stable sets which are
    # valid for the parent graph and the only difference between the parent and ours is SAME(node1,node2)
    # or DIFF(node1, node2))
    def consprop(self, constraints, nusefulconss, nmarkedconss, proptiming):
        # entering()
        # get the only constraint we care about
        assert len(self.stack) > 0
        cons = self.stack[-1]
        assert cons is not None and cons.name is not None
        consData = self.consDatas[cons.name]

        node1 = consData['node1']
        node2 = consData['node2']
        scip = self.model

        # define iterator to select vars depending on intersection cardinal
        def feasible_var_iter(intersect_length):
            for v in scip.getVars(transformed=True):
                s = scip.varsets[v.getIndex()]
                if v.isInLP() and not scip.isFeasZero(v.getUbLocal()) \
                   and len(set(s).intersection({node1, node2})) == intersect_length:
                    yield v
        # change upper bound for selected vars
        if consData['type'] == "differ":
            # set Ub to 0 for sets containing both node1 and node2
            for v in feasible_var_iter(2):
                scip.chgVarUb(v, 0.0)
        elif consData['type'] == "same":
            # set Ub to 0 for sets containing exactly either node1 or node2
            for v in feasible_var_iter(1):
                scip.chgVarUb(v, 0.0)
        else:
            raise ValueError("Should not be here!")
        consData['npropagatedvars'] = len(self.model.varsets)
        return {'result': SCIP_RESULT.DIDNOTFIND}

    # fundamental callbacks do nothing
    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        entering()
        leaving()
        return {'result': SCIP_RESULT.FEASIBLE}

    def consenfops(self, constraints, nusefulconss, solinfeasible, objinfeasible):
        entering()
        leaving()
        return {'result': SCIP_RESULT.FEASIBLE}

    def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason, completely):
        entering()
        leaving()
        return {'result': SCIP_RESULT.FEASIBLE}

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        entering()
        leaving()

########### END THE CONSHDLR
############ THE PRICER


class ColoringPricer(Pricer):

    def __init__(self):
        '''set up pricer data'''
        entering()
        self.nround = 0
        self.current_graph = None
        self.lowerbound = 0
        leaving()

    def pricerinit(self):
        entering()
        scip = self.model
        self.maxroundsroot = max(5, len(scip.initial_sol))*max(50, len(scip.graph))
        self.maxroundsnode = self.maxroundsroot
        # get transformed conss
        for v, cons in scip.conss.items():
            scip.conss[v] = scip.getTransformedCons(cons)
        # get transformed vars and update index for each varset
        for var in scip.getVars(transformed=False):
            myset = scip.varsets[var.getIndex()]
            var = self.model.getTransformedVar(var)
            # add transformed var to scip.varsets as var.getIndex() is unique
            scip.varsets[var.getIndex()] = myset
        leaving()

    def pricerexit(self):
        entering()
        #  how to properly release var / del does not cut it...
        leaving()

    def pricerfarkas(self):
        '''farkas pricing method of variable pricer for infeasible LPs'''
        entering()
        scip = self.model
        # get current node's graph
        G = getCurrentGraph(self.model)
        # mark all nodes in the current stable sets as colored:
        # colored_nodes = union_{s \in stable_sets} s
        colored_nodes = set()
        for myvar in scip.getVars(transformed=True):
            myset = scip.varsets[myvar.getIndex()]
            if not scip.isFeasZero(myvar.getUbLocal()) and \
               (scip.getNNodes() == 0 or myvar.isInLP() or
               scip.getNumber() == 1):
                colored_nodes.update(myset)
        # build maximal stable sets until all nodes are colored
        uncolored_nodes = list(set(G.nodes()).difference(colored_nodes))
        assert len(uncolored_nodes) > 0
        while uncolored_nodes:
            mwis_set = (uncolored_nodes[0],)
            for e in uncolored_nodes[1:]:
                feasible_iter = ((c, e) not in G.edges for c in mwis_set)
                if all(feasible_iter):
                    mwis_set += (e,)
            mwis_set = tuple(sorted(mwis_set))
            if mwis_set not in (s for s in scip.varsets.values()):
                print("pricerfarkas, max stable set found: ", mwis_set)
                # create newVar
                currentNumVar = len(scip.varsets)
                newVar = scip.addVar(vtype='B',
                                     name="StableVar"+str(currentNumVar),
                                     obj=1.0, pricedVar=True)
                scip.varsets[newVar.getIndex()] = mwis_set
                # add variable to constraints
                for v in mwis_set:
                    scip.addConsCoeff(scip.conss[v], newVar, 1.0)
            # clean uncolored_nodes
            for e in mwis_set:
                uncolored_nodes.remove(e)
        leaving()
        return {'result': SCIP_RESULT.SUCCESS}

    def pricerredcost(self):
        '''the reduced cost function for the variable pricer'''
        # entering()
        scip = self.model
        # stop pricing if limit for pricing rounds reached
        if self.current_graph == getCurrentGraph(scip):
            if self.nround > 0:
                self.nround -= 1
        else:
            if self.current_graph is None:
                self.nround = self.maxroundsroot
            else:
                self.nround = self.maxroundsnode
            self.lowerbound = -self.model.infinity()
            self.current_graph = getCurrentGraph(scip)
        if self.nround == 0:
            print("maxrounds reached, pricing interrupted")
            leaving()
            return {'result': SCIP_RESULT.DIDNOTRUN,
                    "lowerbound": self.lowerbound}

        # keep best integral solution found so far
        # do we really need to push it as a solution to SCIP?
        currentLPval = scip.getLPObjVal()
        if scip.isFeasIntegral(currentLPval) and \
           scip.is_lp_integral(transformed=True) and \
           currentLPval + 0.5 < len(scip.best_integral_sol):
            print("\nbetter integral solution found", currentLPval)
            scip.best_integral_sol = {}
            sol = scip.createSol()
            for myvar in scip.getVars(transformed=True):
                index = myvar.getIndex()
                myset = scip.varsets[index]
                if scip.isGT(scip.getVal(myvar), 0):
                    currentNumber = len(scip.best_integral_sol)
                    scip.best_integral_sol[currentNumber] = myset
                    scip.setSolVal(sol, myvar, 1)
            assert len(scip.best_integral_sol) == currentLPval
            print(scip.best_integral_sol)
            assert scip.trySol(sol) is True

        # assert self.current_graph is not None
        graph = self.current_graph
        # print('Master Primal bound', scip.getPrimalbound())
        # print('LP Primal and dual bounds', scip.getLPObjVal(), self.lowerbound)
        # get dual solution
        pi = {n: scip.getDualsolLinear(c) for n, c in scip.conss.items()}
        # enforce zero for non-zero dual values below tolerance level (EPS)
        pi_values_below_tolerance = (n for (n, p) in pi.items()
                              if abs(p) > 0 and scip.isFeasZero(p))
        for n in pi_values_below_tolerance:
            pi[n] = 0.0

        # try greedy heuristic
        threshold = 1.0 + EPS
        improvement = False
        n_shuffle = len(pi)
        for mwis_set, mwis_weight in greedy_MWIS(graph, pi, threshold, n_shuffle):
            if mwis_set not in (s for s in scip.varsets.values()):
                # print("G round %d, mwis_set found: " % (self.nround), mwis_set)
                # print("G round %d, mwis_set value found: " % (self.nround), mwis_weight)
                currentNumVar = len(scip.varsets)
                # create new variable for mwis_set
                newVar = scip.addVar(vtype='B',
                                     name="StableVar"+str(currentNumVar),
                                     obj=1.0, pricedVar=True)
                scip.varsets[newVar.getIndex()] = mwis_set
                # add variable to constraints
                for v in mwis_set:
                    scip.addConsCoeff(scip.conss[v], newVar, 1.0)
                improvement = True
        # try exact algorithm if greedy heuristic failed
        if not improvement:
            mwis_set, mwis_weight = exact_MWIS(graph, pi)
            if mwis_weight > threshold:
                if mwis_set not in (s for s in scip.varsets.values()):
                    # print("E round %d, exact mwis_set found: " % (self.nround), mwis_set)
                    # print("E round %d, mwis_set value found: " % (self.nround), mwis_weight)
                    currentNumVar = len(scip.varsets)
                    # create new variable for mwis_set
                    newVar = scip.addVar(vtype='B',
                                         name="StableVar"+str(currentNumVar),
                                         obj=1.0, pricedVar=True)
                    scip.varsets[newVar.getIndex()] = mwis_set
                    # add variable to constraints
                    for v in mwis_set:
                        scip.addConsCoeff(scip.conss[v], newVar, 1.0)
                    improvement = True
                # update lower bound
                if scip.getLPSolstat() == SCIP_LPSOLSTAT.OPTIMAL:
                    potential_lb = currentLPval + \
                                   (1.0 - mwis_weight)*scip.getPrimalbound()
                    if self.lowerbound < potential_lb:
                        self.lowerbound = potential_lb

        if not improvement:
            cons = getCurrentCons(scip)
            consData = scip.conshdlr.consDatas[cons.name]
            # track B&B node processing to get an optimality check
            if "parent_cons" in consData:
                parent_cons = consData['parent_cons']
                MyFile.write('parent' + " " + str(parent_cons.name) + " " + 'child' + " " + str(cons.name) + "\n")
            else:
                MyFile.write('parent None child' + " " + str(cons.name) + "\n")
            # either LP finds an optimal solution or mwis_set has been already found
            if scip.isFeasZero(mwis_weight - 1.0):
                print("BS=1 Primal and dual LP bounds", currentLPval, currentLPval)
                MyFile.write("BS=1 " + str(currentLPval) + " " + str(currentLPval) + "\n")
                return {'result': SCIP_RESULT.SUCCESS,
                        'lowerbound': currentLPval}
            elif scip.isFeasZero(mwis_weight):
                print("BS=0 Primal and dual LP bounds", currentLPval, currentLPval)
                MyFile.write("BS=1 " + str(currentLPval) + " " + str(currentLPval) + "\n")
                return {'result': SCIP_RESULT.DIDNOTRUN,
                        'lowerbound': self.lowerbound}
            else:
                print("mwis set exists LP Primal and dual bounds", currentLPval, self.lowerbound)
                MyFile.write("mwis set exists " + str(currentLPval) + " " + str(self.lowerbound) + "\n")
        return {'result': SCIP_RESULT.SUCCESS}

############# END THE PRICER
############ THE READER

def read_problem(file_location):
    '''read problem file'''
    # file_location file features a graph represented by its node and edge
    # counts in first line, and then edges as two integers per line
    print("reading file ", file_location)
    # open and read file
    input_data_file = open(file_location, 'r')
    input_data = input_data_file.read()
    lines = input_data.split('\n')
    # read first line
    parts = lines[0].split()
    node_count = int(parts[0])
    edge_count = int(parts[1])
    # create graph
    G = nx.ThinGraph()
    G.add_nodes_from(range(node_count))
    # read the other lines
    edge_list = []
    for i in range(1, edge_count + 1):
        u, v = lines[i].split()
        edge_list.append((int(u), int(v)))
    assert len(edge_list) == edge_count
    G.add_edges_from(edge_list)
    return G


def create_problem(G):
    '''create SCIP model, pre-process resulting graph
    and set up model constraints'''
    # create model
    print("CREATING THE ORIGINAL COLORING OBJECT NOW")
    color_scip = Coloring(G)
    print("object created id: ", id(color_scip))
    # color_scip.printVersion()  # just to check that scip is not null!
    color_scip.preprocess_graph()
    color_scip.set_up_constraints()
    return color_scip


def write_dict_to_file(dictionary, file_location):
    f = open(file_location, "w")
    # pi dictionary data
    for k, v in dictionary.items():
        f.write(str(k) + " " + str(v) + "\n")
    f.close()
    print('dictionary written to file %s' % (file_location))

############ END THE READER
############ THE MAIN PART


def test_main():
    '''create model and associated plug-ins '''
    entering()

    # # opting to read graph in predefined x_y files with x as number of nodes,
    # # and y as graph density
    # ext = "70_3"  # 50_1 100_5 20_1 50_3
    # # file_location file features a graph represented by its node and edge
    # # counts in first line, and then edges as two integers per line
    # file_location = "./data/gc_" + ext
    # G = read_problem(file_location)

    # opting to select predefined graph in networkx
    # chromatic number of mycielski_graph(n) is n
    ntest = 6
    G = nx.mycielski_graph(ntest)

    print("creating problem")
    scip = create_problem(G)
    print("problem setted: printing it")

    # include branching rule
    print("including branchingrule")
    branching_rule = ColoringBranch()
    scip.includeBranchrule(branching_rule, "coloring", "branching rule for coloring",
                           priority=50000, maxdepth=-1, maxbounddist=1.0)
    print("done including branchingrule: id ", id(branching_rule))

    # include conshdlr
    print("including conshdlr")
    conshdlr = StoreGraphConshdlr()
    scip.includeConshdlr(conshdlr, "storeGraph", "storing graph at nodes of the tree constraint handler",
                         enfopriority=0, chckpriority=2000000,
                         propfreq=1, eagerfreq=100,
                         delayprop=False, needscons=True,
                         proptiming=SCIP_PROPTIMING.BEFORELP)
    scip.conshdlr = conshdlr
    print("done including conshdlr: id ", id(conshdlr))
    print("conshdlr is weak referencing model id ", id(conshdlr.model))
    assert conshdlr.model == scip
    print(id(conshdlr.model), id(scip))
    print(conshdlr.model, scip)

    # include pricer
    print("including pricer")
    pricer = ColoringPricer()
    scip.includePricer(pricer, "coloring", "pricer for coloring",
                       priority=5000000, delay=True)
    print("done including pricer: id", id(pricer))

    # set parameters to enable correct dual values
    scip.setPresolve(SCIP_PARAMSETTING.OFF)
    scip.setSeparating(SCIP_PARAMSETTING.OFF)
    scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    scip.disablePropagation()

    print("---------------start optimizing!!!!")
    scip.optimize()
    MyFile.close()

    print("\nLength of initial solution", len(scip.initial_sol))
    print("Initial solution", scip.initial_sol)
    scip.postprocess_graph()
    print("Length of final solution", len(scip.best_integral_sol))
    print("Final solution", scip.best_integral_sol)
    print("Final solution is feasible?", scip.is_integral_sol_feasible(transformed=True))
    write_dict_to_file(scip.best_integral_sol, 'final_coloring.dat')

############ END THE MAIN PART


if __name__ == "__main__":

    test_main()
