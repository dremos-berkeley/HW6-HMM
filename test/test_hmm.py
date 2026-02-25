import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO:
    Create an instance of your HMM class using the "small_weather_hmm.npz" file.
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct.

    Ensure that the output of your Viterbi algorithm correct.
    Assert that the state sequence returned is in the right order, has the right number of states, etc.

    In addition, check for at least 2 edge cases using this toy model.
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    # create HMM instance
    model = HiddenMarkovModel(
        observation_states=mini_hmm['observation_states'],
        hidden_states=mini_hmm['hidden_states'],
        prior_p=mini_hmm['prior_p'],
        transition_p=mini_hmm['transition_p'],
        emission_p=mini_hmm['emission_p']
    )

    obs_seq = mini_input['observation_state_sequence']
    expected_hidden = list(mini_input['best_hidden_state_sequence'])

    # test forward algorithm - check that it returns a reasonable probability
    forward_prob = model.forward(obs_seq)
    assert forward_prob > 0, "Forward probability should be positive"
    assert forward_prob <= 1, "Forward probability should be <= 1"

    # test viterbi algorithm
    viterbi_result = model.viterbi(obs_seq)

    # check correct length
    assert len(viterbi_result) == len(obs_seq), "Viterbi output length should match input length"

    # check correct sequence
    assert viterbi_result == expected_hidden, "Viterbi sequence does not match expected"

    # Edge case 1: empty observation sequence
    empty_result = model.viterbi([])
    assert empty_result == [], "Empty input should return empty list"
    empty_forward = model.forward([])
    assert empty_forward == 0.0, "Empty input should return 0 probability"

    # Edge case 2: single observation
    single_obs = ['sunny']
    single_result = model.viterbi(single_obs)
    assert len(single_result) == 1, "Single obs should return single state"
    single_forward = model.forward(single_obs)
    assert single_forward > 0, "Single obs should have positive probability"



def test_full_weather():

    """
    TODO:
    Create an instance of your HMM class using the "full_weather_hmm.npz" file.
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file

    Ensure that the output of your Viterbi algorithm correct.
    Assert that the state sequence returned is in the right order, has the right number of states, etc.

    """

    full_hmm = np.load('./data/full_weather_hmm.npz')
    full_input = np.load('./data/full_weather_sequences.npz')

    # create HMM instance
    model = HiddenMarkovModel(
        observation_states=full_hmm['observation_states'],
        hidden_states=full_hmm['hidden_states'],
        prior_p=full_hmm['prior_p'],
        transition_p=full_hmm['transition_p'],
        emission_p=full_hmm['emission_p']
    )

    obs_seq = full_input['observation_state_sequence']
    expected_hidden = list(full_input['best_hidden_state_sequence'])

    # test forward algorithm
    forward_prob = model.forward(obs_seq)
    assert forward_prob > 0, "Forward probability should be positive"
    assert forward_prob <= 1, "Forward probability should be <= 1"

    # test viterbi algorithm
    viterbi_result = model.viterbi(obs_seq)

    # check correct length
    assert len(viterbi_result) == len(obs_seq), "Viterbi output length should match input length"

    # check correct sequence
    assert viterbi_result == expected_hidden, "Viterbi sequence does not match expected"
