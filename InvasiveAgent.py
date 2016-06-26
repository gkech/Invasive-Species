# Authors: Majid Alkaee Taleghan, Mark Crowley, Thomas Dietterich
# Invasive Species Project
# 2012 Oregon State University
# Send code issues to: alkaee@gmail.com
# Date: 1/1/13:7:48 PM
#
# I used some of Brian Tanner's Sarsa agent code for the demo version of invasive agent.
#

from Utilities import SamplingUtility, InvasiveUtility
from SimulateNextState import ActionParameterClass
import copy
import random
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
    stepsize = 0.1
    epsilon = 0.1
    discount = 1.0
    policyFrozen = False
    exploringFrozen = False
    edges=[]
    
    def random_player(self,state):
		#find the actions for the state
        stateId = SamplingUtility.getStateId(state)
        #print 'state '+ str(state)[1:-1]
        #if len(self.Q_value_function) == 0 or not self.Q_value_function.has_key(stateId): #len() : Return the length (the number of items) of an object. 
        self.all_allowed_actions[stateId] = InvasiveUtility.getActions(state, self.nbrReaches, self.habitatSize)
            #self.Q_value_function[stateId] = len(self.all_allowed_actions[stateId]) * [0.0]
            
        index = self.randGenerator.randint(0, len(self.all_allowed_actions[stateId]) - 1)
        return self.all_allowed_actions[stateId][index]

    def agent_init(self, taskSpecString):
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpecString)
        self.all_allowed_actions = dict()
        self.Q_value_function = dict()
        if TaskSpec.valid:
            self.nbrReaches = len(TaskSpec.getIntActions())
            self.Bad_Action_Penalty=min(TaskSpec.getRewardRange()[0])
            rewardRange = (min(TaskSpec.getRewardRange()[0]), max(TaskSpec.getRewardRange()[0]))
            self.habitatSize = len(TaskSpec.getIntObservations()) / self.nbrReaches
            self.discount = TaskSpec.getDiscountFactor()
            theExtra=TaskSpec.getExtra().split('BUDGET')
            self.edges=eval(theExtra[0])
            self.budget=eval(theExtra[1].split("by")[0])
#            self.nbrReaches = TaskSpec.getIntActions()[0][0][0]
#            self.Bad_Action_Penalty=min(TaskSpec.getRewardRange()[0])
#            rewardRange = (min(TaskSpec.getRewardRange()[0]), max(TaskSpec.getRewardRange()[0]))
#            self.habitatSize = TaskSpec.getIntObservations()[0][0][0] / self.nbrReaches
#            self.discount = TaskSpec.getDiscountFactor()
#            self.edges=eval(TaskSpec.getExtra().split('by')[0])
        else:
            print "Task Spec could not be parsed: " + taskSpecString

        self.lastAction = Action()
        self.lastObservation = Observation()
        
        # COSTS
        cost_per_invaded_reach = 10
        cost_per_tree = 0.1
        cost_per_empty_slot = 0.09
        eradication_cost = 0.5
        restoration_cost = 0.9
        variable_eradication_cost = 0.4
        variable_restoration_cost_empty = 0.4
        variable_restoration_cost_invaded = 0.8
        
        #CREATE ACTION PARAMETER OBJECT
        self.actionParameterObj = ActionParameterClass(cost_per_tree, eradication_cost, restoration_cost, 0, 0, cost_per_invaded_reach,
                 cost_per_empty_slot, variable_eradication_cost, variable_restoration_cost_invaded, variable_restoration_cost_empty, self.budget)
	#we choose greedy the Q that will tell us which action to make
    def egreedy(self, state):
        #find the actions for the state
        stateId = SamplingUtility.getStateId(state)
        #print 'state '+ str(state)[1:-1]
        if len(self.Q_value_function) == 0 or not self.Q_value_function.has_key(stateId): #len() : Return the length (the number of items) of an object. 
            self.all_allowed_actions[stateId] = InvasiveUtility.getActions(state, self.nbrReaches, self.habitatSize)
            self.Q_value_function[stateId] = len(self.all_allowed_actions[stateId]) * [0.0]
        if not self.exploringFrozen and self.randGenerator.random() < self.epsilon:
            index = self.randGenerator.randint(0, len(self.all_allowed_actions[stateId]) - 1)
        else:
            index = self.Q_value_function[stateId].index(max(self.Q_value_function[stateId]))
        #print 'a '+str(self.all_allowed_actions[stateId][index])[1:-1]
        return self.all_allowed_actions[stateId][index]


    def agent_start(self, observation):
        theState = observation.intArray
        thisIntAction = self.egreedy(theState) #for random player, egreedy=random_player
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
            returnAction = Action()
            returnAction.intArray = newAction
            self.lastAction = copy.deepcopy(returnAction)
            return returnAction

        newState = observation.intArray
        newAction = self.egreedy(newState) #for random player, egreedy=random_player
        
        if type(newAction) is tuple:
            newAction = list(newAction)
            #print newAction
        #we kept the same names from sarsa because it was a bit convenient ---> test test sarsa again, just replace max(blah,blah), with Q_sprime_aprime and uncomment the code below
        Q_sprime_aprime = self.Q_value_function[SamplingUtility.getStateId(newState)][
                          self.all_allowed_actions[SamplingUtility.getStateId(newState)].index(tuple(newAction))]   
        #------>comment lines 133-139 when you want random player
        Q_sa = self.Q_value_function[lastStateId][lastActionIdx]
        new_Q_sa = Q_sa + self.stepsize * (reward + self.discount *Q_sprime_aprime - Q_sa)
        
        if not self.policyFrozen:
            self.Q_value_function[SamplingUtility.getStateId(lastState)][
            self.all_allowed_actions[SamplingUtility.getStateId(lastState)].index(tuple(lastAction))] = new_Q_sa
        #------>comment lines<-----
        returnAction = Action()
        returnAction.intArray = newAction
        self.lastAction = copy.deepcopy(returnAction)
        self.lastObservation = copy.deepcopy(observation)
        return returnAction

	#for the final update, both SARSA and QLearning has the same update function, so we keep the same piece of code
    def agent_end(self, reward):
        lastState = self.lastObservation.intArray
        lastAction = self.lastAction.intArray
        Q_sa = self.Q_value_function[SamplingUtility.getStateId(lastState)][
               self.all_allowed_actions[SamplingUtility.getStateId(lastState)].index(tuple(lastAction))]
        new_Q_sa = Q_sa + self.stepsize * (reward - Q_sa)
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
