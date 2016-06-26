# Authors: Majid Alkaee Taleghan, Mark Crowley, Thomas Dietterich
# Invasive Species Project
# 2012 Oregon State University
# Send code issues to: alkaee@gmail.com
# Date: 1/1/13:7:51 PM
#

import networkx
from numpy.matlib import  repmat
from numpy.numarray import array
from SimulateNextState import SimulationParameterClass, ActionParameterClass, GerminationDispersionParameterClass, simulateNextState
from Utilities import InvasiveUtility
import random
from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Reward_observation_terminal
import numpy as np
from networkx import adjacency_matrix

class InvasiveEnvironment(Environment):
    """
    This class implements the invasive species environment. You could set almost all of the parameters by setting the variables
    in simulationParameterObj and actionParameterObj. To use a specific initialization you need to pass all the input parameters, otherwise
    the domain and action parameters will be set by default. I used some of Brian Tanner's mines code for this environment.
    """

    def __init__(self, simulationParameterObj, actionParameterObj, Bad_Action_Penalty, nbrReaches=7, habitatSize=4, fixedStartState=False,
                 discountFactor=0.9, seed=None):
        """
        :param simulationParameterObj (SimulationParameterClass), contains all the parameters for the domain
        :param actionParameterObj (ActionParameterClass), contains all the parameters for the actions
        :param Bad_Action_Penalty (float), a negative value which will be returned as the consequence of over-budget
        action or non-allowable action on a state
        :param nbrReaches (int), number of reaches in the river network
        :param habitatSize (int), number of habitat in each reach
        :param fixedStartState (bool), indicates using a random starting state or fixed starting state
        :param discountFactor (float), discount factor
        :param seed (int), seed for random number generator (default=None)
        """
        self.seed = seed
        self.fixedStartState = fixedStartState
        self.discountFactor = discountFactor
        self.Bad_Action_Penalty=Bad_Action_Penalty
        if not self.seed is None:
            self.randGenerator = random.Random(self.seed)
        else:
            self.randGenerator = random.Random()
        if simulationParameterObj != None:
            self.simulationParameterObj = simulationParameterObj
            self.actionParameterObj = actionParameterObj
            self.dispertionTable = []
            self.germinationObj = None
        else:
            #upstream rate
            upStreamRate = 0.1
            #downstream rate
            downStreamRate = 0.5
            #exogenous arrival indicator
            exogenousArrivalIndicator = SimulationParameterClass.ExogenousArrivalOn
            #competiton parameter
            competitionFactor = 1
            #there is the same number of
            reachArrivalRates = array([[random.randint(100, 1000) for i in xrange(2)] for i in xrange(nbrReaches)])
            reachArrivalProbs = array([[random.random() for i in xrange(2)] for i in xrange(nbrReaches)])
            #first value is for native and the second one for tamarisk
            prodRate = [200, 200]
            #first value is for native and the second one for tamarisk
            deathRate = [0.2, 0.2]
            graph = InvasiveUtility.createRandomGraph(nbrReaches + 1, balanced=True,randGenerator=self.randGenerator)

            self.simulationParameterObj = SimulationParameterClass(nbrReaches, habitatSize, prodRate, deathRate,
                exogenousArrivalIndicator, reachArrivalRates, reachArrivalProbs, upStreamRate, downStreamRate,
                competitionFactor, graph)
            self.actionParameterObj = ActionParameterClass(costPerTree=0.1, eradicationCost=0.5, restorationCost=0.9,
                eradicationRate=1, restorationRate=1,
                costPerReach=1, emptyCost=0, varEradicationCost=0.5, varInvasiveRestorationCost=0.1,
                varEmptyRestorationCost=0, budget=100)

    def env_init(self):
        """
            Based on the levin model, the dispersion probability is initialized.
        """
        self.dispersionModel = InvasiveUtility.Levin
        notDirectedG = networkx.Graph(self.simulationParameterObj.graph)
        adjMatrix = adjacency_matrix(notDirectedG)

        edges = self.simulationParameterObj.graph.edges()
        simulationParameterObj = self.simulationParameterObj
        if self.dispersionModel == InvasiveUtility.Levin:
            parameters = InvasiveUtility.calculatePath(notDirectedG,adjMatrix, edges, simulationParameterObj.downStreamRate,
                simulationParameterObj.upStreamRate)
            C = (1 - simulationParameterObj.upStreamRate * simulationParameterObj.downStreamRate) / (
                (1 - 2 * simulationParameterObj.upStreamRate) * (1 - simulationParameterObj.downStreamRate))
            self.dispertionTable = np.dot(1 / C, parameters)
            self.germinationObj = GerminationDispersionParameterClass(1, 1)
        #calculating the worst case fully invaded rivers cost
        worst_case = repmat(1, 1, self.simulationParameterObj.nbrReaches * self.simulationParameterObj.habitatSize)[0]
        cost_state_unit = InvasiveUtility.get_unit_invaded_reaches(worst_case,
            self.simulationParameterObj.habitatSize) * self.actionParameterObj.costPerReach
        stateCost = cost_state_unit + InvasiveUtility.get_invaded_reaches(
            worst_case) * self.actionParameterObj.costPerTree
        stateCost = stateCost + InvasiveUtility.get_empty_slots(worst_case) * self.actionParameterObj.emptyCost
        costAction = InvasiveUtility.get_budget_cost_actions(repmat(3, 1, self.simulationParameterObj.nbrReaches)[0],
            worst_case, self.actionParameterObj)
        networkx.adjacency_matrix(self.simulationParameterObj.graph)
        return "VERSION RL-Glue-3.0 PROBLEMTYPE non-episodic DISCOUNTFACTOR " + str(
            self.discountFactor) + " OBSERVATIONS INTS (" + str(
            self.simulationParameterObj.nbrReaches * self.simulationParameterObj.habitatSize) + " 1 3) ACTIONS INTS (" + str(
            self.simulationParameterObj.nbrReaches) + " 1 4) REWARDS (" + str(self.Bad_Action_Penalty)+" "+str(
            -1 * (costAction + stateCost)) + ") EXTRA "+str(self.simulationParameterObj.graph.edges()) + " BUDGET "+str(self.actionParameterObj.budget) +" by Majid Taleghan."

    def env_start(self):
        if self.fixedStartState:
            stateValid = self.setAgentState(self.state)
            if not stateValid:
                print "The fixed start state was NOT valid: " + str(self.state)
                self.setRandomState()
        else:
            self.setRandomState()

        returnObs = Observation()
        #        print self.state
        returnObs.intArray = map(int, list(self.state))
        return returnObs #return observation

    def env_step(self, action):
        action = action.intArray
        assert len(action) == self.simulationParameterObj.nbrReaches, "Expected " + str(
            self.simulationParameterObj.nbrReaches) + " integer action."
        if not InvasiveUtility.is_action_allowable(action, self.state):
            theObs = Observation()
            InvasiveUtility.is_action_allowable(action, self.state)
            #map(int, results)
            theObs.intArray = [-1]
            returnRO = Reward_observation_terminal()
            returnRO.r = self.Bad_Action_Penalty
            returnRO.o = theObs
            return returnRO
        cost_state_unit = InvasiveUtility.get_unit_invaded_reaches(self.state,
            self.simulationParameterObj.habitatSize) * self.actionParameterObj.costPerReach
        stateCost = cost_state_unit + InvasiveUtility.get_invaded_reaches(
            self.state) * self.actionParameterObj.costPerTree
        stateCost = stateCost + InvasiveUtility.get_empty_slots(self.state) * self.actionParameterObj.emptyCost
        costAction = InvasiveUtility.get_budget_cost_actions(action, self.state, self.actionParameterObj)
        if costAction > self.actionParameterObj.budget:
            theObs = Observation()
            InvasiveUtility.is_action_allowable(action, self.state)
            #map(int, results)
            theObs.intArray = [-1]
            returnRO = Reward_observation_terminal()
            returnRO.r = self.Bad_Action_Penalty
            returnRO.o = theObs
            return returnRO

        nextState = simulateNextState(self.state, action, self.simulationParameterObj,
            self.actionParameterObj, self.dispertionTable, self.germinationObj)
        self.state = nextState
        theObs = Observation()
        theObs.intArray = self.state
        returnRO = Reward_observation_terminal()
        returnRO.r = -1 * (costAction + stateCost)
        returnRO.o = theObs
        return returnRO

    def env_cleanup(self):
        pass

    def env_message(self, inMessage):
        #	Message Description
        # 'set-random-start-state'
        #Action: Set flag to do random starting states (the default)
        if inMessage.startswith("set-random-start-state"):
            self.fixedStartState = False
            return "Message understood.  Using random start state."

        #	Message Description
        # 'set-start-state array of length E*H'
        if inMessage.startswith("set-start-state"):
            splitString = inMessage.split(" ")
            self.state = array(eval(splitString[1]))
            self.fixedStartState = True
            return "Message understood.  Using fixed start state."

        return "InvasiveEnvironment(Python) does not respond to messages."

    def setAgentState(self, S):
        assert len(S)==self.simulationParameterObj.habitatSize*self.simulationParameterObj.nbrReaches
        self.state = S
        valid = True
        return valid

    def setRandomState(self):
        S = array([random.randint(1, 3) for i in
                   xrange(self.simulationParameterObj.nbrReaches * self.simulationParameterObj.habitatSize)])
        self.setAgentState(S)

    def checkValid(self, S):
        valid = True
        return valid

    def printState(self):
        print "Agent is at: " + str(self.state)

if __name__ == "__main__":
    EnvironmentLoader.loadEnvironment(
        InvasiveEnvironment(simulationParameterObj=None, actionParameterObj=None, Bad_Action_Penalty=-10000,fixedStartState=False, nbrReaches=7,
            habitatSize=4, seed=1))
