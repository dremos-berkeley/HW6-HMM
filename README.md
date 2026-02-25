# HW6-HMM

## Methods

**Forward Algorithm**
The forward algorithm computes the probability of an observation sequence given the HMM model. It uses dynamic programming to avoid redundant calculations by storing intermediate alpha values in a table. For each time step, it calculates the probability of being in each hidden state given all observations up to that point. The final probability is the sum of all alpha values at the last time step.

**Viterbi Algorithm**
The viterbi algorithm finds the most likely sequence of hidden states that generated the observed sequence. It uses dynamic programming but takes the maximum probability path instead of summing over all paths. We maintain a backpointer table to track which previous state gave us the max probability at each step, then traceback from the best final state to reconstruct the optimal path.