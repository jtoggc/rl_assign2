from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy


class GreedyPolicy(Policy):
    def __init__(self, Q, nA, nS):
        self.p = np.array([[1/nA]*nA]*nS)
        self.Q = Q
        # for i in range(len(self.Q)):
        #     action_idx = np.argmax(self.Q[i,:]) # action with largest Q value
        #     self.p[i, action_idx] = 1.0 # full probability for largest Q value

    def update_policy(self, Q: np.array) -> None:
        for i in range(len(Q)):
            action_idx = np.argmax(Q[i,:]) # action with largest Q value
            self.p[i, action_idx] = 1.0 # full probability for largest Q value, 0 else
            self.p[i, [y for y in range(self.p.shape[1]) if y != action_idx]] = 0.0

    def action_prob(self, state: int, action: int) -> float:
        return self.p[state, action]

    def action(self, state: int) -> int:
        return np.argmax(self.p[state])


def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################

    V = initV
    for episode in trajs:
        T = len(episode)
        for t in range(T):
            tau = t-n+1
            if tau >= 0:
                G = 0
                for i in range(tau+1, min(tau+n, T-1)+1):
                    G += pow(env_spec.gamma, i-tau-1)*episode[i][2] # reward is at second tuple index in each step
                if tau+n < T:
                    G += pow(env_spec.gamma, n)*V[episode[tau+n][0]] # could be indexing error here for V(s_tau+n)
                V[episode[tau][0]] += alpha*(G - V[episode[tau][0]])
            if tau == T-1:
                break
    return V

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################

    Q = initQ
    pi = GreedyPolicy(Q, env_spec.nA, env_spec.nS)

    for episode in trajs:
        T = len(episode)
        for t in range(T):
            tau = t-n+1
            if tau >= 0:
                rho = 1
                G = 0
                for k in range(tau+1, min(tau+n, T-2) + 1):
                    rho *= pi.action_prob(episode[k][0], episode[k][1])/bpi.action_prob(episode[k][0], episode[k][1])
                for j in range(tau+1, min(tau+n, T-1) + 1):
                    G += pow(env_spec.gamma, j-tau-1)*episode[j][2]
                if tau+n < T:
                    G += pow(env_spec.gamma, n)*Q[episode[tau+n][0], episode[tau+n][1]]
                Q[episode[tau][0], episode[tau][1]] += alpha*rho*(G - Q[episode[tau][0], episode[tau][1]])
                pi.update_policy(Q)
            if tau == T-1:
                break

    return Q, pi
