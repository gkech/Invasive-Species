# Authors: Majid Alkaee Taleghan, Mark Crowley, Thomas Dietterich
# Invasive Species Project
# 2012 Oregon State University
# Send code issues to: alkaee@gmail.com
# Date: 1/1/13:7:48 PM
#
# I used some of Brian Tanner's Sarsa agent code for the demo version of invasive agent.
#

from Utilities import SamplingUtility, InvasiveUtility
import copy
from random import Random
from rlglue.agent import AgentLoader
from rlglue.agent.Agent import Agent
from rlglue.types import Action, Observation
from rlglue.utils import TaskSpecVRLGLUE3

class InvasiveAgent(Agent):
    randGenerator = Random()
    #initializes from rlglue.types, the last action done <---
    lastAction = Action()
    #initializes from rlglue.types, the last observation had <---
    lastObservation = Observation()
    sarsa_stepsize = 0.1
    a_learning_rate=0.05
    gamma_dis_factor=0.5
    sarsa_epsilon = 0.1
    sarsa_gamma = 1.0
    policyFrozen = False
    exploringFrozen = False
    edges=[]

    def agent_init(self, taskSpecString):
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpecString)
        self.all_allowed_actions = dict()
        self.Q_value_function = dict()
        if TaskSpec.valid:
            self.nbrReaches = len(TaskSpec.getIntActions())
            self.Bad_Action_Penalty=min(TaskSpec.getRewardRange()[0])
            rewardRange = (min(TaskSpec.getRewardRange()[0]), max(TaskSpec.getRewardRange()[0]))
            self.habitatSize = len(TaskSpec.getIntObservations()) / self.nbrReaches
            self.sarsa_gamma = TaskSpec.getDiscountFactor()
            theExtra=TaskSpec.getExtra().split('BUDGET')
            self.edges=eval(theExtra[0])
            self.budget=eval(theExtra[1].split("by")[0])
#            self.nbrReaches = TaskSpec.getIntActions()[0][0][0]
#            self.Bad_Action_Penalty=min(TaskSpec.getRewardRange()[0])
#            rewardRange = (min(TaskSpec.getRewardRange()[0]), max(TaskSpec.getRewardRange()[0]))
#            self.habitatSize = TaskSpec.getIntObservations()[0][0][0] / self.nbrReaches
#            self.sarsa_gamma = TaskSpec.getDiscountFactor()
#            self.edges=eval(TaskSpec.getExtra().split('by')[0])
        else:
            print "Task Spec could not be parsed: " + taskSpecString

        self.lastAction = Action()
        self.lastObservation = Observation()

    def egreedy(self, state):
        #find the actions for the state
        stateId = SamplingUtility.getStateId(state)
        #print 'state '+ str(state)[1:-1]
        if len(self.Q_value_function) == 0 or not self.Q_value_function.has_key(stateId):
            self.all_allowed_actions[stateId] = InvasiveUtility.getActions(state, self.nbrReaches, self.habitatSize)
            self.Q_value_function[stateId] = len(self.all_allowed_actions[stateId]) * [0.0]
        if not self.exploringFrozen and self.randGenerator.random() < self.sarsa_epsilon:
            index = self.randGenerator.randint(0, len(self.all_allowed_actions[stateId]) - 1)
        else:
            index = self.Q_value_function[stateId].index(max(self.Q_value_function[stateId]))
        #print 'a '+str(self.all_allowed_actions[stateId][index])[1:-1]
        return self.all_allowed_actions[stateId][index]


    def agent_start(self, observation):
        theState = observation.intArray
        thisIntAction = self.egreedy(theState)
        if type(thisIntAction) is tuple:
            thisIntAction = list(thisIntAction)
        returnAction = Action()
        returnAction.intArray = thisIntAction

        self.lastAction = copy.deepcopy(returnAction)
        self.lastObservation = copy.deepcopy(observation)

        return returnAction

    def agent_step(self, reward, observation):
		
        lastState = self.lastObservation.intArray
        lastAction = self.lastAction.intArray
        lastStateId = SamplingUtility.getStateId(lastState)
        lastActionIdx = self.all_allowed_actions[lastStateId].index(tuple(lastAction))
        if reward == self.Bad_Action_Penalty:
            self.all_allowed_actions[lastStateId].pop(lastActionIdx)
            self.Q_value_function[lastStateId].pop(lastActionIdx)
            newAction = self.egreedy(self.lastObservation.intArray)
            print InvasiveUtility.get_budget_cost_actions(lastAction, lastState, self.actionParameterObj)
            returnAction = Action()
            returnAction.intArray = newAction
            self.lastAction = copy.deepcopy(returnAction)
            return returnAction

        newState = observation.intArray
        newAction = self.egreedy(newState)
        if type(newAction) is tuple:
            newAction = list(newAction)
        Q_sa = self.Q_value_function[lastStateId][lastActionIdx]
        #print "THE Q_sa IS : "
        #print Q_sa
        Q_sprime_aprime = self.Q_value_function[SamplingUtility.getStateId(newState)][
                          self.all_allowed_actions[SamplingUtility.getStateId(newState)].index(tuple(newAction))]
        new_Q_sa = Q_sa + self.sarsa_stepsize * (reward + self.sarsa_gamma * Q_sprime_aprime - Q_sa)
        #print "THE new_Q_sa IS : "
        #print new_Q_sa
        if not self.policyFrozen:
            self.Q_value_function[SamplingUtility.getStateId(lastState)][
            self.all_allowed_actions[SamplingUtility.getStateId(lastState)].index(tuple(lastAction))] = new_Q_sa
        returnAction = Action()
        returnAction.intArray = newAction
        self.lastAction = copy.deepcopy(returnAction)
        self.lastObservation = copy.deepcopy(observation)
        return returnAction

    def agent_end(self, reward):
        lastState = self.lastObservation.intArray
        lastAction = self.lastAction.intArray
        Q_sa = self.Q_value_function[SamplingUtility.getStateId(lastState)][
               self.all_allowed_actions[SamplingUtility.getStateId(lastState)].index(tuple(lastAction))]
        new_Q_sa = Q_sa + self.sarsa_stepsize * (reward - Q_sa)
        if not self.policyFrozen:
            self.Q_value_function[SamplingUtility.getStateId(lastState)][
            self.all_allowed_actions[SamplingUtility.getStateId(lastState)].index(tuple(lastAction))] = new_Q_sa

    def agent_cleanup(self):
        pass


    def agent_message(self, inMessage):
        #	Message Description
        # 'freeze learning'
        # Action: Set flag to stop updating policy
        #
        if inMessage.startswith("freeze learning"):
            self.policyFrozen = True
            return "message understood, policy frozen"

        #	Message Description
        # unfreeze learning
        # Action: Set flag to resume updating policy
        #
        if inMessage.startswith("unfreeze learning"):
            self.policyFrozen = False
            return "message understood, policy unfrozen"

        #Message Description
        # freeze exploring
        # Action: Set flag to stop exploring (greedy actions only)
        #
        if inMessage.startswith("freeze exploring"):
            self.exploringFrozen = True
            return "message understood, exploring frozen"

        #Message Description
        # unfreeze exploring
        # Action: Set flag to resume exploring (e-greedy actions)
        #
        if inMessage.startswith("unfreeze exploring"):
            self.exploringFrozen = False
            return "message understood, exploring frozen"

        return "Invasive agent does not understand your message."


if __name__ == "__main__":
    AgentLoader.loadAgent(InvasiveAgent())
