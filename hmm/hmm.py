import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables

        # handle empty sequence - can't compute probability of nothing
        if len(input_observation_states) == 0:
            return 0.0

        num_obs = len(input_observation_states)
        num_states = len(self.hidden_states)

        # alpha table: rows = time steps, cols = hidden states
        # stores P(being in state j after seeing observations 0 to t)
        alpha = [[0.0 for _ in range(num_states)] for _ in range(num_obs)]

        # Step 2. Calculate probabilities

        # initialization for t=0
        first_obs = input_observation_states[0]
        if first_obs not in self.observation_states_dict:
            raise ValueError(f"Observation '{first_obs}' not in known observations")
        first_obs_idx = self.observation_states_dict[first_obs]

        for s in range(num_states):
            # P(state) * P(observation | state)
            alpha[0][s] = self.prior_p[s] * self.emission_p[s][first_obs_idx]

        # recursion for t = 1 to T-1
        for t in range(1, num_obs):
            obs = input_observation_states[t]
            if obs not in self.observation_states_dict:
                raise ValueError(f"Observation '{obs}' not in known observations")
            obs_idx = self.observation_states_dict[obs]

            for s in range(num_states):
                # sum over all prev states: alpha[t-1][prev] * P(s|prev)
                trans_sum = 0.0
                for prev_s in range(num_states):
                    trans_sum += alpha[t-1][prev_s] * self.transition_p[prev_s][s]

                alpha[t][s] = self.emission_p[s][obs_idx] * trans_sum

        # Step 3. Return final probability
        # total prob = sum of alpha values at final time step
        forward_prob = sum(alpha[num_obs - 1])
        return forward_prob 
        


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables

        # handle edge case of empty sequence
        if len(decode_observation_states) == 0:
            return []

        num_obs = len(decode_observation_states)
        num_states = len(self.hidden_states)

        #store probabilities of hidden state at each step
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states))
        backpointer = [[0 for _ in range(num_states)] for _ in range(num_obs)]

        # convert observation names to indices for matrix lookups
        obs_indices = []
        for obs in decode_observation_states:
            if obs not in self.observation_states_dict:
                raise ValueError(f"Observation '{obs}' not in known observations")
            obs_indices.append(self.observation_states_dict[obs])

       # Step 2. Calculate Probabilities
        for t in range(num_obs):
            for s in range(num_states):
                if t == 0:
                    # base case: prior prob * emission prob
                    viterbi_table[t][s] = self.prior_p[s] * self.emission_p[s][obs_indices[t]]
                else:
                    # find max probability from all previous states
                    max_prob = max(viterbi_table[t-1][prev_s] * self.transition_p[prev_s][s] for prev_s in range(num_states))
                    viterbi_table[t][s] = max_prob * self.emission_p[s][obs_indices[t]]
                    # store which previous state gave us the max
                    backpointer[t][s] = max(range(num_states), key=lambda prev_s: viterbi_table[t-1][prev_s] * self.transition_p[prev_s][s])

        # Step 3. Traceback
        # find the best final state
        best_path_pointer = max(range(num_states), key=lambda s: viterbi_table[-1][s])
        best_path[num_obs - 1] = best_path_pointer

        # work backwards through backpointers
        for t in range(num_obs - 1, 0, -1):
            best_path[t - 1] = backpointer[t][int(best_path[t])]

        # Step 4. Return best hidden state sequence
        # convert indices back to state names
        best_hidden_state_sequence = [self.hidden_states_dict[int(idx)] for idx in best_path]
        return best_hidden_state_sequence