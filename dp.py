from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

class DeterministicPolicy(Policy):
    def __init__(self, p):
        self.p = p # matrix of probabilities nSxnA size; 

    def action_prob(self, state: int, action: int) -> float:
        return self.p[state, action]

    def action(self, state: int) -> int:
        return np.argmax(self.p[state])


def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Value Prediction Algorithm (Hint: Sutton Book p.75)
    #####################

    Q = np.zeros((env.spec.nS, env.spec.nA))
    V = initV

    # Value prediction for v_pi
    terminated = False
    while not terminated:
        delta = 0.0
        for state in range(env.spec.nS):
            curr_v = V[state]
            val = 0
            for action in range(env.spec.nA):
                possible_next_states_prob = env.TD[state, action, :]
                for next_state, prob in enumerate(possible_next_states_prob):
                    val += pi.action_prob(state, action)*prob*(env.R[state, action, next_state] + env.spec.gamma*V[next_state])
            #print(abs(best_val - curr_v))
            V[state] = val
            delta = max(delta, abs(curr_v - val))
            #print(delta)
            #print(delta)
        if delta < theta:
            terminated = True

    return V, Q

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################

    V = initV #starting values
    terminated = False
    while not terminated:
        delta = 0.0
        for state in range(env.spec.nS):
            curr_v = V[state]
            best_val = -1*np.inf
            for action in range(env.spec.nA):
                val = 0
                possible_next_states_prob = env.TD[state, action, :]
                for next_state, prob in enumerate(possible_next_states_prob):
                    val += prob*(env.R[state, action, next_state] + env.spec.gamma*V[next_state])
                if val > best_val:
                    best_val = val
            #print(abs(best_val - curr_v))
            V[state] = best_val
            delta = max(delta, abs(curr_v - best_val))
            
        if delta < theta:
            terminated = True

    # Form the policy probabilities nSxnA array, with prob = 1 for maximal action
    # This is the one step ahead planning from v_pi
    p = np.zeros((env.spec.nS, env.spec.nA))
    for state in range(env.spec.nS): # TODO: Create helper function for prediction iteration
        best_val = -1*np.inf
        best_action = -1
        for action in range(env.spec.nA):
            val = 0
            possible_next_states_prob = env.TD[state, action, :]
            for next_state, prob in enumerate(possible_next_states_prob):
                val += prob*(env.R[state, action, next_state] + env.spec.gamma*V[next_state])
            if val > best_val:
                best_val = val
                best_action = action
        p[state, best_action] = 1.0 # put all prob on this action for this state

    pi = DeterministicPolicy(p)

    return V, pi
