from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using ordinary importance
    # sampling (Hint: Sutton Book p. 109, every-visit implementation is fine)
    #####################

    Q = initQ

    counts = np.zeros((env_spec.nS, env_spec.nA)) # for holding counts in ordinary i.s.
    for episode in trajs:
        G = 0 # return of rewards init to 0
        W = 1 # importance sampling weight init to 1
        n_steps = len(episode)
        for t in range(n_steps-1, -1,-1):
            cur_state, cur_action, next_reward, next_state = episode[t]
            G = env_spec.gamma*G + next_reward
            counts[cur_state, cur_action] += 1 # in theory this is the weight but for ordinary i.s. this is 1
            Q[cur_state, cur_action] += (W/counts[cur_state, cur_action])*(G - Q[cur_state, cur_action]) # incremental update
            W *= (pi.action_prob(cur_state, cur_action)/bpi.action_prob(cur_state, cur_action))
            if W == 0:
                break

    return Q

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using weighted importance
    # sampling (Hint: Sutton Book p. 110, every-visit implementation is fine)
    #####################

    Q = initQ

    cum_weight = np.zeros((env_spec.nS, env_spec.nA)) # for holding cumulative weight
    for episode in trajs:
        G = 0 # return of rewards init to 0
        W = 1 # importance sampling weight init to 1
        n_steps = len(episode)
        for t in range(n_steps-1, -1,-1):
            cur_state, cur_action, next_reward, next_state = episode[t]
            G = env_spec.gamma*G + next_reward
            cum_weight[cur_state, cur_action] += W # in theory this is the weight but for ordinary i.s. this is 1
            Q[cur_state, cur_action] += (W/cum_weight[cur_state, cur_action])*(G - Q[cur_state, cur_action]) # incremental update
            W *= (pi.action_prob(cur_state, cur_action)/bpi.action_prob(cur_state, cur_action))
            if W == 0:
                break

    return Q
