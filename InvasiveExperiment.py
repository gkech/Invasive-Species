# Authors: Majid Alkaee Taleghan, Mark Crowley, Thomas Dietterich
# Invasive Species Project
# 2012 Oregon State University
# Send code issues to: alkaee@gmail.com
# Date: 1/1/13:7:51 PM
#
# I used some of Brian Tanner's experiment code for invasive experiment.
#

import numpy
from numpy.numarray import array

import random

import math
import rlglue.RLGlue as RLGlue

def demo():
    statistics = []
    episodeLength = 100
    #this_score = evaluateAgent()
    #printScore(0, this_score)
    #statistics.append(this_score)

    for i in range(1, 1000):
		
        RLGlue.RL_env_message("set-start-state " + S)
        RLGlue.RL_start()
        RLGlue.RL_episode(episodeLength)
        this_return = RLGlue.RL_return()
        print "%d\t\t%.2f" % (i, this_return)
        statistics.append(this_return)
        saveResultToCSV(statistics, "MyResults_sarsa1000_ver2.csv")

		#saveResultToCSV(statistics, "results.csv")


def printScore(afterEpisodes, score_tuple):
    print "%d\t\t%.2f\t\t%.2f" % (afterEpisodes, score_tuple[0], score_tuple[1])

#
# Tell the agent to stop learning, then execute n episodes with his current
# policy.  Estimate the mean and variance of the return over these episodes.
#
def evaluateAgent():
    sum = 0
    sum_of_squares = 0
    n = 10
    episodeLength = 100
    RLGlue.RL_agent_message("freeze learning")
    #print "FREEZE LEARNING"
    for i in range(0, n):
        RLGlue.RL_episode(100)
        this_return = RLGlue.RL_return()
        sum += this_return
        sum_of_squares += this_return ** 2

    mean = sum / n
    variance = (sum_of_squares - n * mean * mean) / (n - 1.0)
    standard_dev = math.sqrt(variance)

    RLGlue.RL_agent_message("unfreeze learning")
    #print "UNFREEZE LEARNING"
    return mean, standard_dev


def saveResultToCSV(statistics, fileName):
    numpy.savetxt(fileName, statistics, delimiter=",")


#
# Just do a single evaluateAgent and print it
#
#def single_evaluation():
#    this_score = evaluateAgent()
#    printScore(0, this_score)

RLGlue.RL_init()
print "Telling the environment to use fixed start state."
nbrReaches=7
habitatSize=4
S = array([random.randint(1, 3) for i in xrange(nbrReaches * habitatSize)])
#S=array([1,1,2, 1, 3, 3, 1])
#S=[1,2,3,3,2,1,3,2,2,3,2,1,2,2,3,2,2,1,3,1,1,2,2,3,3,2,1,1]
S = ",".join(map(str, S)) # just a way to display a list in python
print S
RLGlue.RL_env_message("set-start-state "+S)
RLGlue.RL_start()

print "Starting offline demo\n----------------------------\nWill alternate learning for 10 episodes, then freeze policy and evaluate for 10 episodes.\n"
print "After Episode\tMean Return\tStandard Deviation\n-------------------------------------------------------------------------"
demo()

print "Evaluating the agent again with the random start state:\n\t\tMean Return\tStandardDeviation\n-----------------------------------------------------"
RLGlue.RL_env_message("set-random-start-state")
#single_evaluation()


RLGlue.RL_cleanup()
print "\nProgram Complete."
