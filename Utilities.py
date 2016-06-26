# Authors: Majid Alkaee Taleghan, Mark Crowley, Thomas Dietterich
# Invasive Species Project
# 2012 Oregon State University
# Send code issues to: alkaee@gmail.com
# Date: 12/31/12:9:24 AM
#

from networkx import DiGraph, networkx
import random
from itertools import product
import numpy as np

class SamplingUtility:
    """
    This class is being used in InvasiveAgent.py to translate the state representation to a number.
    """
    sMap = dict()

    @staticmethod
    def getStateId(s):
        sid = 0
        s = tuple(s)
        if SamplingUtility.sMap.has_key(s):
            sid = SamplingUtility.sMap.get(s)
        else:
            sid = len(SamplingUtility.sMap.keys())
            SamplingUtility.sMap[s] = sid
        return sid

    @staticmethod
    def getStateValue(sid):
        index=SamplingUtility.sMap.values().index(sid)
        return SamplingUtility.sMap.keys().__getitem__(index)


class InvasiveUtility:
    """
    This class has useful methods for calculation and also creating the river graph. The river graph is a tree as descriped
    in the problem section
    """
    #The following variables are implicitly constant variables.

    #dispersion kernel selection, we use the Levin model in this version
    Levin = 1
    Alternative_One = 2
    Alternative_Two = 3
    #different slot occupancy
    #tamarisk
    Tam = 1
    #native
    Nat = 2
    #empty
    Emp = 3
    #a character that represents each slot occupancy
    #empty
    Emp_Sym = 'E'
    #tamarisk
    Tam_Sym = 'T'
    #native
    Nat_Sym = 'N'

    #different actions
    #nothing
    Not = 1
    #eradication
    Erad = 2
    #restoration
    Res = 3
    #eradication+restoration
    EradRes = 4

    #a character that represents the action
    #nothing
    Not_Sym = 'N'
    #eradication
    Erad_Sym = 'E'
    #restoration
    Res_Sym = 'R'
    #eradication+restoration
    EradRes_Sym = 'S'

    @staticmethod
    def getActionName(action):
        """
        Translate action of numbers to action of chars
        :param action: an array of numbers
        :return: a char array of action
        """
        action_str = ['0'] * len(action)
        for i in range(len(action)):
            if action[i] == InvasiveUtility.Not:
                action_str[i] = InvasiveUtility.Not_Sym
            elif action[i] == InvasiveUtility.Erad:
                action_str[i] = InvasiveUtility.Erad_Sym
            elif action[i] == InvasiveUtility.Res:
                action_str[i] = InvasiveUtility.Res_Sym
            elif action[i] == InvasiveUtility.EradRes:
                action_str[i] = InvasiveUtility.EradRes_Sym
        return action_str

    @staticmethod
    def calculatePath(notDirectedG, adj, edges, downStreamRate, upStreamRate):
        """
        calculate the dispersion kernel values, without operating the C normalization
        This is based on Levin model
        :param adj: adjacency matrix
        :param edges: an array of edges
        :param downStreamRate: downstream rate
        :param upStreamRate: upstream rate
        :return: a matrix
        """
        n = len(adj)
        reaches = range(len(edges))#np.unique(edges[:,1])
        nbr_reaches = len(reaches)
        parameters = np.ones((nbr_reaches, nbr_reaches))
        allpaths = networkx.all_pairs_dijkstra_path(notDirectedG)

        for i in xrange(nbr_reaches):
            src_edge = edges[i]#reaches[i]#parent of edge
            for j in xrange(nbr_reaches):
                if i == j:
                    continue

                dest_edge = edges[j]
                first = allpaths[src_edge[0]][dest_edge[0]]
                second = allpaths[src_edge[0]][dest_edge[1]]
                third = allpaths[src_edge[1]][dest_edge[0]]
                forth = allpaths[src_edge[1]][dest_edge[1]]

                idx = np.argmax((len(first), len(second), len(third), len(forth)))

                if idx == 0:
                    path=first
                elif idx == 1:
                    path=second
                elif idx == 2:
                    path=third
                elif idx == 3:
                    path=forth

                for k in xrange(len(path) - 1):
                    if any([(x, y) for x, y in edges if x == path[k] and y == path[k + 1]]):
                        parameters[i, j] = downStreamRate * parameters[i, j]
                    else:
                        parameters[i, j] = upStreamRate * parameters[i, j]
        return parameters

    @staticmethod
    def get_budget_cost_actions_reach(action, S_reach, eradicationCost, restorationCost, varEradicationCost,
                                      varEmptyRestorationCost, varInvasiveRestorationCost):
        """
        calculate the cost of action per reach
        :param action:
        :param S_reach: the sub-state in each reach
        :param eradicationCost:
        :param restorationCost:
        :param varEradicationCost:
        :param varEmptyRestorationCost:
        :param varInvasiveRestorationCost:
        :return: cost (float)
        """
        cost = 0
        if(type(action) is str and action == InvasiveUtility.Erad_Sym) or (
        type(action) is not str and action == InvasiveUtility.Erad):
            if type(S_reach) is str:
                cost = eradicationCost + sum(S_reach == InvasiveUtility.Tam_Sym) * varEradicationCost
            else:
                cost = eradicationCost + sum(S_reach == InvasiveUtility.Tam) * varEradicationCost
        elif (type(action) is str and action == InvasiveUtility.Res_Sym) or (
        type(action) is not str and action == InvasiveUtility.Res):
            if type(S_reach) is str:
                cost = restorationCost + sum(S_reach == InvasiveUtility.Emp_Sym) * varEmptyRestorationCost
            else:
                cost = restorationCost + sum(S_reach == InvasiveUtility.Emp) * varEmptyRestorationCost
        elif (type(action) is str and action == InvasiveUtility.EradRes_Sym) or (
        type(action) is not str and action == InvasiveUtility.EradRes):
            if type(S_reach) is str:
                cost = restorationCost + sum(S_reach == InvasiveUtility.Tam_Sym) * varInvasiveRestorationCost
            else:
                cost = restorationCost + sum(S_reach == InvasiveUtility.Tam) * varInvasiveRestorationCost
        return cost

    @staticmethod
    def get_budget_cost_actions(action, state, actionParameterObj):
        """
        calculate the cost for the action on all reaches
        :param action:
        :param state:
        :param actionParameterObj:
        :return: total cost
        """
        H = len(state) / len(action)
        cost = 0
        nbr_reaches = len(state) / H
        #actionParameterObj=ActionParameterClass(actionParameterObj)
        for i in xrange(nbr_reaches):
            actionReach = action[i]
            S_reach = state[i * H:(i + 1) * H]
            cost += InvasiveUtility.get_budget_cost_actions_reach(actionReach, S_reach,
                actionParameterObj.eradicationCost, actionParameterObj.restorationCost,
                actionParameterObj.varEradicationCost,
                actionParameterObj.varEmptyRestorationCost, actionParameterObj.varInvasiveRestorationCost)
        return cost

    @staticmethod
    def is_action_allowable(action, state):
        """
        checks if the specific action is allowable on state state
        :param action:
        :param state:
        :return: boolean, true mean the action is allowable
        """
        H = len(state) / len(action)
        bool = True
        nbr_reaches = len(action)
        for i in xrange(nbr_reaches):
            action_type = action[i]
            S_reach = state[i * H:(i + 1) * H]
            if(type(action_type) is str and action_type == InvasiveUtility.Erad_Sym) or (
            type(action_type) is not str and action_type == InvasiveUtility.Erad):
                if type(S_reach) is str:
                    if sum(S_reach == InvasiveUtility.Tam_Sym) == 0:
                        bool = False
                else:
                    if sum(S_reach == InvasiveUtility.Tam) == 0:
                        bool = False
            elif (type(action_type) is str and action_type == InvasiveUtility.Res_Sym ) or (
            type(action_type) is not str and action_type == InvasiveUtility.Res):
                if type(S_reach) is str:
                    if sum(S_reach == InvasiveUtility.Emp_Sym) == 0:
                        bool = False
                else:
                    if sum(S_reach == InvasiveUtility.Emp) == 0:
                        bool = False
            elif (type(action_type) is str and action_type == InvasiveUtility.EradRes_Sym ) or (
            type(action_type) is not str and action_type == InvasiveUtility.EradRes ):
                if type(S_reach) is str:
                    if sum(S_reach == InvasiveUtility.Tam_Sym) == 0:
                        bool = False
                else:
                    if sum(S_reach == InvasiveUtility.Tam) == 0:
                        bool = False
        return bool

    @staticmethod
    def getActions(state, nbr_reaches, H):
        """
        Returns the possible actions that could be allowable on a given state, regardless of budget consideration
        :param state:
        :param nbr_reaches:
        :param H:
        :return:
        """
        action = [[]] * nbr_reaches
        for r in xrange(nbr_reaches):
            S_reach = state[r * H:(r + 1) * H]
            if sum(S_reach == InvasiveUtility.Nat) == H:
                action[r] = [InvasiveUtility.Not]
            elif sum(S_reach == InvasiveUtility.Tam) == 0:
                action[r] = [InvasiveUtility.Not, InvasiveUtility.Res]
            elif sum(S_reach == InvasiveUtility.Tam) == H:
                action[r] = [InvasiveUtility.Not, InvasiveUtility.Erad, InvasiveUtility.EradRes]
            elif sum(S_reach == InvasiveUtility.Emp) == H:
                action[r] = [InvasiveUtility.Not, InvasiveUtility.Res]
            elif sum(S_reach == InvasiveUtility.Emp) == 0:#N or T
                action[r] = [InvasiveUtility.Not, InvasiveUtility.Erad, InvasiveUtility.EradRes]
            else:
                action[r] = [InvasiveUtility.Not, InvasiveUtility.Erad, InvasiveUtility.Res, InvasiveUtility.EradRes]
        return list(product(*action))

    @staticmethod
    def get_invaded_reaches(state):
        """
        return the number of invaded reaches
        :param state:
        :return int:
        """
        if type(state) is str:
            invaded_reaches = sum(state == InvasiveUtility.Tam_Sym)
        else:
            invaded_reaches = sum(state == InvasiveUtility.Tam)
        return invaded_reaches

    @staticmethod
    def get_empty_slots(state):
        """
        return total number of empty slots
        :param state:
        :return: int
        """
        if type(state) is str:
            empty_slots = sum(state == InvasiveUtility.Emp_Sym)
        else:
            empty_slots = sum(state == InvasiveUtility.Emp)
        return empty_slots

    @staticmethod
    def get_unit_invaded_reaches_str(state, habitatSize):
        """
        return the number of invaded reaches
        :param state: array of char
        :param habitatSize:
        """
        invaded_reaches = 0
        for i in xrange(0, len(state), habitatSize):
            S_reach = state[i:i + habitatSize]
            invaded_reaches = invaded_reaches + (sum(S_reach == InvasiveUtility.Tam_Sym) > 0)

    @staticmethod
    def  get_unit_invaded_reaches_num(state, habitatSize):
        """
        return the number of invaded reaches
        :param state: array of char
        :param habitatSize:
        """
        invaded_reaches = 0
        for i in xrange(0, len(state), habitatSize):
            S_reach = state[i:i + habitatSize]
            invaded_reaches = invaded_reaches + (sum(S_reach == InvasiveUtility.Tam) > 0)
        return invaded_reaches

    @staticmethod
    def get_unit_invaded_reaches(state, habitatSize):
        if type(state) is str:
            invaded_reaches = InvasiveUtility.get_unit_invaded_reaches_str(state, habitatSize)
        else:
            invaded_reaches = InvasiveUtility.get_unit_invaded_reaches_num(state, habitatSize)
        return invaded_reaches

    @staticmethod
    def createRandomGraph(n, balanced=None, randGenerator=None, seed=None):

        """
        Parameters
        ----------
        :rtype : graph
        n : int
            The number of nodes.
        balanced : bool
            False: The tree is not balanced
            True: The tree is balanced
        seed : int, optional
            Seed for random number generator (default=None).
        """
        G = DiGraph()

        if not randGenerator is None:
            randGen = randGenerator
        else:
            randGen=random
            if not seed is None:
                randGen.seed(seed)
        parents = range(0, n - 1)
        nodes = []
        visited = [0] * (n - 1)
        #each node can have only one child and maximum of two parents
        root = parents[randGen.randint(0,len(parents) - 1)]
        nodes.append(root)
        del parents[parents.index(root)]
        G.add_edge(root, n - 1)
        while len(parents) > 0:
            if balanced:
                node = 0
            else:
                if len(nodes) == 1:
                    node = 0
                else:
                    node = randGen.randint(0,len(nodes) - 1)
            node = nodes[node]
            if len(parents) == 1:
                parent = 0
            else:
                parent = randGen.randint(0,len(parents) - 1)
            parent = parents[parent]
            G.add_edge(parent, node)
            del parents[parents.index(parent)]
            nodes.append(parent)
            visited[node] += 1
            if visited[node] == 2:
                del nodes[nodes.index(node)]

        return G