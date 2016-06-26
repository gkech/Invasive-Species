/*
Authors: Majid Alkaee Taleghan, Mark Crowley, Thomas Dietterich
Invasive Species Project
2012 Oregon State University
Send code issues to: alkaee@gmail.com
Date: 3/17/13:7:48 PM

I used some of Brian Tanner's Sarsa agent code for the demo version of invasive agent.
*/

import java.util.*;

import org.apache.commons.lang3.ArrayUtils;
import org.rlcommunity.rlglue.codec.AgentInterface;
import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.util.AgentLoader;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec;


public class InvasiveAgent implements AgentInterface {

    private Random randGenerator = new Random();
    private Action lastAction;
    private Observation lastObservation;
    private double sarsa_stepsize = 0.1;
    private double sarsa_epsilon = 0.1;
    private double sarsa_gamma = 1.0;
    private boolean policyFrozen = false;
    private boolean exploringFrozen = false;
    private String edges;
    private double budget;
    private int habitatSize;
    private double Bad_Action_Penalty;
    private HashMap<Integer, Vector<Double>> Q_value_function;
    private int nbrReaches;
    private double rewardRangeMin;
    private double rewardRangeMax;
    private HashMap<Integer, List<List<Integer>>> all_allowed_actions;

    /**
     * Parse the task spec, make sure it is only 1 integer observation and
     * action, and then allocate the valueFunction.
     *
     * @param taskSpecification
     */
    public void agent_init(String taskSpecification) {
        TaskSpec theTaskSpec = new TaskSpec(taskSpecification);
        //Assume the message is valid
        nbrReaches = theTaskSpec.getNumDiscreteActionDims();
        Bad_Action_Penalty = theTaskSpec.getRewardRange().getMin();
        rewardRangeMin = theTaskSpec.getRewardRange().getMin();
        rewardRangeMax = theTaskSpec.getRewardRange().getMax();
        habitatSize = theTaskSpec.getNumDiscreteObsDims() / this.nbrReaches;
        sarsa_gamma = theTaskSpec.getDiscountFactor();

        String[] theExtra=theTaskSpec.getExtraString().split("BUDGET");
        edges=theExtra[0];
        budget= Double.parseDouble(theExtra[1].split("by")[0]);
        sarsa_gamma = theTaskSpec.getDiscountFactor();

        Q_value_function = new HashMap<Integer, Vector<Double>>();
        all_allowed_actions = new HashMap<Integer, List<List<Integer>>>();
    }

    /**
     * Choose an action e-greedily from the value function and store the action
     * and observation.
     *
     * @param observation
     * @return
     */
    public Action agent_start(Observation observation) {
        int[] theState = observation.intArray;
        int[] thisIntAction = this.egreedy(theState);
        Action returnAction = new Action();
        if (thisIntAction.length > 0) {
            returnAction.intArray = thisIntAction;

            this.lastAction = returnAction;
            this.lastObservation = observation;
        }
        return returnAction;
    }

    /**
     * Choose an action e-greedily from the value function and store the action
     * and observation.  Update the valueFunction entry for the last
     * state,action pair.
     *
     * @param reward
     * @param observation
     * @return
     */
    public Action agent_step(double reward, Observation observation) {
        int[] lastState = this.lastObservation.intArray;
        int[] lastAction_ = this.lastAction.intArray;
        int lastStateId = Utilities.getStateId(lastState);
        List<Integer> lastActionList = Arrays.asList(ArrayUtils.toObject(lastAction_));
        int lastActionIdx = this.all_allowed_actions.get(lastStateId).indexOf(lastActionList);
        if (reward == this.Bad_Action_Penalty) {
            this.all_allowed_actions.get(lastStateId).remove(lastActionIdx);
            this.Q_value_function.get(lastStateId).remove(lastActionIdx);
            int[] newAction = this.egreedy(this.lastObservation.intArray);
            Action returnAction = new Action();
            returnAction.intArray = newAction;
            this.lastAction = returnAction;
            return returnAction;
        }
        int[] newState = observation.intArray;
        int[] newAction = this.egreedy(newState);
        List<Integer> newActionList = Arrays.asList(ArrayUtils.toObject(newAction));
        double Q_sa = this.Q_value_function.get(lastStateId).get(lastActionIdx);
        double Q_sprime_aprime = this.Q_value_function.get(Utilities.getStateId(newState)).get(this.all_allowed_actions.get(Utilities.getStateId(newState)).indexOf(newActionList));
        double new_Q_sa = Q_sa + this.sarsa_stepsize * (reward + this.sarsa_gamma * Q_sprime_aprime - Q_sa);
        if (!this.policyFrozen) {
            this.Q_value_function.get(Utilities.getStateId(lastState)).set(this.all_allowed_actions.get(Utilities.getStateId(lastState)).indexOf(lastActionList), new_Q_sa);
        }
        Action returnAction = new Action();
        returnAction.intArray = newAction;

        this.lastAction = returnAction;
        this.lastObservation = observation;
        return returnAction;
    }

    /**
     * The episode is over, learn from the last reward that was received.
     *
     * @param reward
     */
    public void agent_end(double reward) {
        int[] lastState = this.lastObservation.intArray;
        int[] lastAction_ = this.lastAction.intArray;
        List<Integer> newActionList = Arrays.asList(ArrayUtils.toObject(lastAction_));
        double Q_sa = this.Q_value_function.get(Utilities.getStateId(lastState)).get(
                this.all_allowed_actions.get(Utilities.getStateId(lastState)).indexOf(newActionList));
        double new_Q_sa = (Q_sa + this.sarsa_stepsize * (reward - Q_sa));
        if (!this.policyFrozen) {
            this.Q_value_function.get(Utilities.getStateId(lastState)).set(
                    this.all_allowed_actions.get(Utilities.getStateId(lastState)).indexOf(newActionList), new_Q_sa);
        }
    }

    /**
     * Release memory that is no longer required/used.
     */
    public void agent_cleanup() {
        lastAction = null;
        lastObservation = null;
        Q_value_function = null;
        all_allowed_actions = null;
    }

    /**
     * This agent responds to some simple messages for freezing learning and
     * saving/loading the value function to a file.
     *
     * @param inMessage
     * @return
     */
    public String agent_message(String inMessage) {
        //	Message Description
        // 'freeze learning'
        // Action{ Set flag to stop updating policy
        //
        if (inMessage.startsWith("freeze learning")) {
            this.policyFrozen = true;
            return "message understood, policy frozen";
        }
        //	Message Description
        // unfreeze learning
        // Action{ Set flag to resume updating policy
        //
        if (inMessage.startsWith("unfreeze learning")) {
            this.policyFrozen = false;
            return "message understood, policy unfrozen";
        }
        //Message Description
        // freeze exploring
        // Action{ Set flag to stop exploring (greedy actions only)
        //
        if (inMessage.startsWith("freeze exploring")) {
            this.exploringFrozen = true;
            return "message understood, exploring frozen";
        }
        //Message Description
        // unfreeze exploring
        // Action{ Set flag to resume exploring (e-greedy actions)
        //
        if (inMessage.startsWith("unfreeze exploring")) {
            this.exploringFrozen = false;
            return "message understood, exploring frozen";
        }
        return "Invasive agent does not understand your message.";

    }

    /**
     * Selects a random action with probability 1-sarsa_epsilon,
     * and the action with the highest value otherwise.  This is a
     * quick'n'dirty implementation, it does not do tie-breaking.
     *
     * @param theState
     * @return
     */
    private int[] egreedy(int[] theState) {
        int stateId = Utilities.getStateId(theState);
        if (this.Q_value_function.isEmpty() || !this.Q_value_function.containsKey(stateId)) {
            this.all_allowed_actions.put(stateId, Utilities.getActions(theState, this.nbrReaches, this.habitatSize));
            Vector<Double> vector = new Vector<Double>(Collections.nCopies(this.all_allowed_actions.get(stateId).size(), 0.0));
            this.Q_value_function.put(stateId, vector);
        }
        int index;
        if (!this.exploringFrozen && this.randGenerator.nextDouble() < this.sarsa_epsilon) {
            index = this.randGenerator.nextInt(this.all_allowed_actions.get(stateId).size());

        } else {
            Double max = Collections.max(this.Q_value_function.get(stateId));
            index = this.Q_value_function.get(stateId).indexOf(max);
        }
        final int[] result = new int[this.all_allowed_actions.get(stateId).get(index).size()];
        for (int i = 0; i < this.all_allowed_actions.get(stateId).get(index).size(); i++) {
            result[i] = this.all_allowed_actions.get(stateId).get(index).get(i).intValue();
        }
        return result;
    }

    /**
     * This is a trick we can use to make the agent easily loadable.  Using this
     * trick you can directly execute the class and it will load itself through
     * AgentLoader and connect to the rl_glue server.
     *
     * @param args
     */
    public static void main(String[] args) {
        AgentLoader theLoader = new AgentLoader(new InvasiveAgent());
        theLoader.run();
    }


}
