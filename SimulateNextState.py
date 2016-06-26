# Authors: Majid Alkaee Taleghan, Mark Crowley, Thomas Dietterich
# Invasive Species Project
# 2012 Oregon State University
# Send code issues to: alkaee@gmail.com
# Date: 12/20/12:3:26 PM
#
from Utilities import InvasiveUtility
from numpy.random import random
import numpy as np
from numpy.matlib import *

class GerminationDispersionParameterClass:
    def __init__(self, germinationSuccNat, germinationSuccTam):
        self.germinationSuccTam = germinationSuccTam
        self.germinationSuccNat = germinationSuccNat


class SimulationParameterClass:
    """
    This class contains the related parameters, which define the invasive species domain
    """
    #two different alternatives that exogenousOnOffIndicator could have
    #the exogenous arrival is on
    ExogenousArrivalOn = 2
    #the exogenous arrival is off
    ExogenousArrivalOff = 1

    def __init__(self, nbrReaches, habitatSize, prodRate, deathRate, exogenousOnOffIndicator, reachArrivalRates,
             reachArrivalProbs, upStreamRate, downStreamRate, competitionFactor, graph):
        """
            :param competitionFactor, competition parameter
            :param deathRate (array of length 2 of float), first column shows Native, and the second column shows Tamarisk
            :param habitatSize (int)
            :param exogenousOnOffIndicator (On=2, Off=1), indicates if there is exogenous arrival
            :param prodRate (array of length 2 of float) production rate
            :param reachArrivalProbs (matrix of size (nbrReaches,2)), first column shows Native, and the second column shows Tamarisk
            :param reachArrivalRates (matrix of size (nbrReaches,2)), first column shows Native, and the second column shows Tamarisk
            :param upStreamRate (float)
            :param downStreamRate (float)
            :param graph (networkx graph), a graph representing the river network
            Note that the position of the reaches in the state and action is based on the graph.edges() output
        """
        self.nbrReaches = nbrReaches
        self.habitatSize = habitatSize
        self.prodRate = prodRate
        self.deathRate = deathRate
        self.exogenousArrivalIndicator = exogenousOnOffIndicator
        self.reachArrivalRates = reachArrivalRates
        self.reachArrivalProbs = reachArrivalProbs
        self.downStreamRate = downStreamRate
        self.upStreamRate = upStreamRate
        self.competitionFactor = competitionFactor
        self.graph = graph


class ActionParameterClass:
    """
    This class contains the related parameters, which define the actions and state costs
    """
    def __init__(self, costPerTree, eradicationCost, restorationCost, eradicationRate, restorationRate, costPerReach,
                 emptyCost, varEradicationCost, varInvasiveRestorationCost, varEmptyRestorationCost, budget):
        """
        :param budget (float)
        :param costPerReach (float), cost per invaded reach
        :param costPerTree (float), cost per invaded tree
        :param emptyCost (float), cost for empty slot
        :param eradicationCost (float), fixed eradication cost
        :param eradicationRate (float), eradication success rate
        :param restorationCost (float), fixed restoration cost
        :param restorationRate (float), restoration success rate
        :param varEmptyRestorationCost (float), variable restoration cost for empty slot
        :param varEradicationCost (float), variable eradication cost for empty slot
        :param varInvasiveRestorationCost (float), variable restoration cost for empty slot
        """
        self.costPerTree = costPerTree
        self.eradicationCost = eradicationCost
        self.restorationCost = restorationCost
        self.eradicationRate = eradicationRate
        self.restorationRate = restorationRate
        self.costPerReach = costPerReach
        self.emptyCost = emptyCost
        self.varEradicationCost = varEradicationCost
        self.varInvasiveRestorationCost = varInvasiveRestorationCost
        self.varEmptyRestorationCost = varEmptyRestorationCost
        self.budget = budget


def binomial(nv, pv):
    if size(pv) == 1:
        if pv == 1:
            return nv
        else:
            [rows, cols] = shape(nv)
            result = np.zeros((rows, cols))
            nnz, cnz = np.nonzero(nv)
            for index in range(len(nnz)):
                row = nnz[index]
                col = cnz[index]
                result[row, col] = random.binomial(nv[row, col], pv)
            return result

    else:
        assert size(nv,0)==size(pv,0)
        assert size(nv,1)==size(pv,1)
        [rows, cols] = shape(nv)
        result = np.zeros((rows, cols))
        nnz, cnz = np.nonzero(nv)
        for index in range(len(nnz)):
            row = nnz[index]
            col = cnz[index]
            #H = nv.shape[0] / nv.shape[1]
            #result[row, col] = random.binomial(nv[row, col], pv[row / float(H), col])
            result[row, col] = random.binomial(nv[row, col], pv[row, col])
        return result


def simulateNextState(state, action, simulationParameterObj, actionParameterObj, dispertionTable,
                      germinationObj):
    """
    simulate based on the input parameters and state and action
    :param state (an array of length simulationParameterObj.nbrReaches* simulationParameterObj.habitatSize)
    :param action (array of length simulationParameterObj.nbrReaches)
    :param simulationParameterObj (SimulationParameterClass)
    :param dispertionTable (matrix of size (simulationParameterObj.nbrReaches,simulationParameterObj.nbrReaches))
    :param germinationObj (GerminationClass)
    :return next state
    """
    H = simulationParameterObj.habitatSize
    Prod_rate = simulationParameterObj.prodRate
    Death_Rate = simulationParameterObj.deathRate
    on_off_indicator = simulationParameterObj.exogenousArrivalIndicator
    reach_arrival_rates = simulationParameterObj.reachArrivalRates
    reach_arrival_probs = simulationParameterObj.reachArrivalProbs
    beta = simulationParameterObj.competitionFactor
    eradication_rate = actionParameterObj.eradicationRate
    restoration_rate = actionParameterObj.restorationRate
    nbr_Reaches = simulationParameterObj.nbrReaches

    Nat_Prod_rate = Prod_rate[0]
    Tam_Prod_rate = Prod_rate[1]
    Nat_Death_Rate = Death_Rate[0]
    Tam_Death_Rate = Death_Rate[1]
    nbr_samples = 1
    result = np.zeros((nbr_samples, len(state)))
    for sampling_idx in range(nbr_samples):
        S_ad = np.zeros((len(state), 1))
        rnd_v = random.rand(1, len(state))
        for i in range(len(state)):
            beforeDeath = state[i]
            action_type = action[int(np.floor(i / H))]
            afterDeath = 0
            rnd = rnd_v[0, i]
            if(action_type == InvasiveUtility.Not):
                if (beforeDeath == InvasiveUtility.Emp):
                    afterDeath = InvasiveUtility.Emp
                else:#(beforeDeath!=Emp)
                    if beforeDeath == InvasiveUtility.Tam and rnd <= Tam_Death_Rate:
                        afterDeath = InvasiveUtility.Emp
                    elif beforeDeath == InvasiveUtility.Nat and rnd <= Nat_Death_Rate:
                        afterDeath = InvasiveUtility.Emp
                    else:
                        afterDeath = beforeDeath
            elif(action_type == InvasiveUtility.Erad):
                if (beforeDeath == InvasiveUtility.Emp):
                    afterDeath = InvasiveUtility.Emp
                else:#(beforeDeath!=Emp)
                    if (beforeDeath == InvasiveUtility.Tam):
                        if rnd <= eradication_rate:
                            afterDeath = InvasiveUtility.Emp
                        else:
                            afterDeath = beforeDeath
                    elif (beforeDeath == InvasiveUtility.Nat):
                        if rnd <= Nat_Death_Rate:
                            afterDeath = InvasiveUtility.Emp
                        else:
                            afterDeath = beforeDeath

            elif(action_type == InvasiveUtility.EradRes):
                if (beforeDeath == InvasiveUtility.Nat):
                    if rnd <= Nat_Death_Rate:
                        afterDeath = InvasiveUtility.Emp
                    else:
                        afterDeath = InvasiveUtility.Nat

                elif(beforeDeath == InvasiveUtility.Tam):
                    if rnd <= eradication_rate * (1 - restoration_rate):
                        afterDeath = InvasiveUtility.Emp
                    elif rnd <= eradication_rate:
                        afterDeath = InvasiveUtility.Nat
                    else:
                        afterDeath = InvasiveUtility.Tam

                elif(beforeDeath == InvasiveUtility.Emp):
                    afterDeath = InvasiveUtility.Emp
                #                    if rnd <= (1 - restoration_rate):
                #                        afterDeath = Invasive_Utility.Emp
                #                    else:
                #                        afterDeath = Invasive_Utility.Nat
            elif(action_type == InvasiveUtility.Res):
                if (beforeDeath == InvasiveUtility.Emp):
                    if rnd <= (1 - restoration_rate):
                        afterDeath = InvasiveUtility.Emp
                    else:
                        afterDeath = InvasiveUtility.Nat
                else:#(beforeDeath!=Emp)
                    if beforeDeath == InvasiveUtility.Tam and rnd <= Tam_Death_Rate:
                        afterDeath = InvasiveUtility.Emp
                    elif beforeDeath == InvasiveUtility.Nat and rnd <= Nat_Death_Rate:
                        afterDeath = InvasiveUtility.Emp
                    else:
                        afterDeath = beforeDeath
            S_ad[i] = afterDeath
        G_T = Tam_Prod_rate * (S_ad == InvasiveUtility.Tam)
        G_N = Nat_Prod_rate * (S_ad == InvasiveUtility.Nat)
        G_T=sum(reshape(G_T,(nbr_Reaches,-1)),1)
        G_N=sum(reshape(G_N,(nbr_Reaches,-1)),1)
        Exg_T = 0
        Exg_N = 0
        if on_off_indicator == SimulationParameterClass.ExogenousArrivalOn:
            if size(reach_arrival_rates) > 0:
                Exg = binomial(reach_arrival_rates, reach_arrival_probs)
                Exg_T = Exg[:, 1]
                Exg_N = Exg[:, 0]
        gT_to = np.sum(binomial(repmat(G_T, nbr_Reaches,1).T, dispertionTable), axis=0)
        gN_to = sum(binomial(repmat(G_N, nbr_Reaches,1).T, dispertionTable), axis=0)
        arr = array([gT_to + Exg_T, gN_to + Exg_N])
        gt_vec = reshape(arr.conj().transpose(), (1, size(arr)))
        #Germination Process
        gt_vec[0:2:] = binomial(gt_vec[0:2:], germinationObj.germinationSuccTam)
        gt_vec[1:2:] = binomial(gt_vec[1:2:], germinationObj.germinationSuccNat)
        landed = repmat(gt_vec, H, 1)
        new_S = binomial(landed, 1 / float(H))

        final_S = np.zeros((len(state), 1), dtype='int')
        for i in range(nbr_Reaches):
            for h in range(H):
                idx = H * i + h
                si = S_ad[idx, 0]
                ghT_land = new_S[h, 2 * i]
                ghN_land = new_S[h, 2 * i + 1]
                if (si == InvasiveUtility.Emp):
                    if (ghT_land == 0 and ghN_land == 0):
                        final_S[idx] = InvasiveUtility.Emp
                    else:
                        rnd = random.random()
                        Tam_p = beta * ghT_land / (beta * ghT_land + ghN_land)
                        if rnd <= Tam_p:
                            final_S[idx] = InvasiveUtility.Tam
                        else:
                            final_S[idx] = InvasiveUtility.Nat
                else:
                    final_S[idx] = si

        #new_S=final_S
        final_S = np.squeeze(np.asarray(final_S))
        if nbr_samples == 1:
            return final_S
        else:
            result[sampling_idx, :] = final_S#.conj().transpose()
    return result
