/*Authors: Majid Alkaee Taleghan, Mark Crowley, Thomas Dietterich
* Invasive Species Project
* 2012-2013 Oregon State University
* Send code issues to: alkaee@gmail.com
* Date: 3/18/12:9:24 AM
*
*/

import org.apache.commons.lang3.ArrayUtils;

import java.util.*;

/**
 This class has useful methods for calculation of possible actions in invasive species project.
 **/

public class Utilities {
    //tamarisk
    static final int Tam = 1;
    //native
    static final int Nat = 2;
    //empty
    static final int Emp = 3;
    //a character that represents each slot occupancy
    //empty
    static final char Emp_Sym = 'E';
    //tamarisk
    static final char Tam_Sym = 'T';
    //native
    static final char Nat_Sym = 'N';

    //different actions
    //nothing
    static final int Not = 1;
    //eradication
    static final int Erad = 2;
    //restoration
    static final int Res = 3;
    //eradication+restoration
    static final int EradRes = 4;

    //a character that represents the action
    //nothing
    static final char Not_Sym = 'N';
    //eradication
    static final char Erad_Sym = 'E';
    //restoration
    static final char Res_Sym = 'R';
    //eradication+restoration
    static final char EradRes_Sym = 'S';

    //A map to translate the array state representation to a number.
    static Map<List<Integer>, Integer> sMap = new HashMap<List<Integer>, Integer>();

    /**
     * assigns a unique number to a state represented by array
     * @param state
     * @return
     */
    public static int getStateId(int[] state) {
        int sid = 0;

        List<Integer> s = Arrays.asList(ArrayUtils.toObject(state));
        if (sMap.containsKey(s)) {
            sid = sMap.get(s);
        } else {
            sid = sMap.keySet().size();
            sMap.put(s, sid);
        }
        return sid;
    }

    /**
     * Does the reverse operation of getStateId
     * @param sid
     * @return
     */
    public static List<Integer> getStateValue(int sid) {
        int index = Arrays.asList(sMap.values().toArray()).indexOf(sid);
        return (List<Integer>) sMap.keySet().toArray()[index];

    }

    /**
     * Returns the possible actions that could be allowable on a given state, regardless of budget consideration
     * @param state
     * @param nbrReaches
     * @param habitatSize
     * @return
     */

    public static List<List<Integer>> getActions(int[] state, int nbrReaches, int habitatSize) {
        List<Integer[]> action = new ArrayList<Integer[]>(nbrReaches);
        for (int r = 0; r < nbrReaches; r++) {
            int[] S_reach = Arrays.copyOfRange(state, r * habitatSize, (r + 1) * habitatSize);

            if (sum(equals(S_reach, Utilities.Nat)) == habitatSize) {
                action.add(r, new Integer[]{Utilities.Not});
            } else if (sum(equals(S_reach, Utilities.Tam)) == 0) {
                action.add(r, new Integer[]{Utilities.Not, Utilities.Res});
            } else if (sum(equals(S_reach, Utilities.Tam)) == habitatSize) {
                action.add(r, new Integer[]{Utilities.Not, Utilities.Erad, Utilities.EradRes});
            } else if (sum(equals(S_reach, Utilities.Emp)) == habitatSize) {
                action.add(r, new Integer[]{Utilities.Not, Utilities.Res});
            } else if (sum(equals(S_reach, Utilities.Emp)) == 0) {//N or T
                action.add(r, new Integer[]{Utilities.Not, Utilities.Erad, Utilities.EradRes});
            } else {
                action.add(r, new Integer[]{Utilities.Not, Utilities.Erad, Utilities.Res, Utilities.EradRes});
            }
        }
        List<List<Integer>> actions = new ArrayList<List<Integer>>();
        ArrayList<Integer> action_ = new ArrayList<Integer>(Collections.nCopies(nbrReaches, 0));
        Permute(action, 0, 0, action_, actions, nbrReaches);
        return actions;

    }

    private static int[] equals(int[] arr, int val) {
        int[] out = new int[arr.length];
        for (int i = 0; i < arr.length; i++)
            if (arr[i] == val) {
                out[i] = 1;
            } else {
                out[i] = 0;
            }
        return out;
    }

    /**
     * returns the sum of array
     * @param myArray
     * @return
     */
    public static int sum(int[] myArray) {
        int sum = 0;
        for (int i = 0; i < myArray.length; i++)
            sum += myArray[i];
        return sum;
    }

    /**
     * permute based on the possibilities and output the results as realization in out variable
     * @param possibilities
     * @param pos
     * @param selector
     * @param realization
     * @param out
     * @param length
     */
    public static void Permute(List<Integer[]> possibilities, int pos, int selector, List<Integer> realization, List<List<Integer>> out, int length) {
        if (pos == length) {
            out.add(new ArrayList<Integer>(realization));
            return;
        }
        for (int j = 0; j < possibilities.get(selector).length; j++) {
            realization.set(pos, possibilities.get(selector)[j]);
            Permute(possibilities, pos + 1, selector + 1, realization, out, length);
        }
    }

    public static void main(String[] args) {
        Utilities.getActions(new int[]{1, 1, 1, 2, 2}, 5, 1);
    }
}




