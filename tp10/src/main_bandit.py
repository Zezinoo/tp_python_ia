#!/usr/bin/env python3

# main_bandit.py
import numpy as np
import matplotlib.pyplot as plt


from bandit_env import KArmedBandit
from agent import EpsilonGreedyAgent
from plot_utils import (
    plot_cumulative_mean,
    plot_multiple_cumulative,
    plot_true_probs,
    plot_chosen_arm_quality,
)
from dataclasses import dataclass

@dataclass
class SimulationResult:
    cum_mean_rewards  : np.ndarray
    mean_reward  : float
    times_each_arm : dict
    actions: list
    regrets: np.ndarray
    qualities: list
    gaps: list
    optimal_choice_rate : float



def run_episode_with_metric(env, epsilon: float, alpha: float, T: int, rng=None):
    """
    Simule un épisode de T pas de temps avec un agent epsilon-glouton.
    Renvoie la récompense moyenne cumulée (array de taille T).
    """
    if rng is None:
        rng = np.random.default_rng()
    agent = EpsilonGreedyAgent(env.K, epsilon=epsilon, alpha=alpha, rng=rng)

    reward_dict = { i : [] for i in range(env.K)}
    actions = []
    regrets = []
    regret_cum = 0
    qualities = []
    gaps = []
    count_optimal_choice = 0
    delta = 0.001

    for e in range(T):
        p_t = env.true_probs(e)
        p_opt_t = np.max(p_t)
        action = agent.select_action()
        if p_t[action] > p_opt_t - delta:
            count_optimal_choice += 1
        reward = env.step(action , e)
        agent.update(action , reward)
        reward_dict[action].append((e , reward))
        actions.append(action)
        regret_cum += (p_opt_t - reward)
        regrets.append(regret_cum)
        qualities.append(p_t[action])
        gaps.append(p_opt_t - p_t[action])

    regrets = np.array(regrets)/np.arange(1 , T+1)

    cum_mean_rewards , mean_reward , times_each_arm = treat_reward_dict(reward_dict , T)
    optimal_choice_rate = count_optimal_choice / T

    return SimulationResult(
        cum_mean_rewards=cum_mean_rewards,
        mean_reward=mean_reward,
        times_each_arm=times_each_arm,
        actions=actions,
        regrets=regrets,
        qualities=qualities,
        gaps=gaps,
        optimal_choice_rate = optimal_choice_rate
    )




def run_episode(env, epsilon: float, alpha: float, T: int, rng=None):
    """
    Simule un épisode de T pas de temps avec un agent epsilon-glouton.
    Renvoie la récompense moyenne cumulée (array de taille T).
    """
    if rng is None:
        rng = np.random.default_rng()
    agent = EpsilonGreedyAgent(env.K, epsilon=epsilon, alpha=alpha, rng=rng)
    # TODO :
    # pour t de 0 à T-1 :
    # - choisir une action avec agent.select_action()
    # - jouer cette action dans l'environnement : env.step(action, t)
    # - appeler agent.update(action, reward)
    # - stocker la récompense

    reward_dict = { i : [] for i in range(env.K)}
    actions = []


    for e in range(T):
        action = agent.select_action()
        reward = env.step(action , e)
        agent.update(action , reward)
        reward_dict[action].append((e , reward))
        actions.append(action)



    return *treat_reward_dict(reward_dict , T) , actions 


def treat_reward_dict(reward_dict , epochs):
    rewards = [i for k in list(reward_dict.values()) for i in k]
    mean_reward = [x[1] for x in rewards]
    
    mean_reward = sum(mean_reward) / len(mean_reward)

    times_each_arm = {f"arm {i+1}" : len(reward_dict[i]) for i in reward_dict.keys()}

    rewards = sorted(rewards , key = lambda x : x[0])
    rewards = np.array([x[1] for x in rewards])
    cum_mean_rewards = np.cumsum(np.array(rewards)) / np.arange(1 , epochs + 1)

    return cum_mean_rewards , mean_reward , times_each_arm



def main():
    print(f"Apprentissage par renforcement")
    env = KArmedBandit( K=4)
    epochs = 2000
    reward_dict = { i : [] for i in range(env.K)}
    for e in range(epochs):
        arm_number = np.random.choice(range(env.K))
        reward = env.step(arm_number , e)
        reward_dict[arm_number].append((e , reward))

    cum_mean_rewards , mean_reward, times_each_arm = treat_reward_dict(reward_dict , epochs)
    eps_cum_mean_reward , eps_mean_reward , eps_times_each_arm  = run_episode(env , epsilon=0.1 , alpha=0.05 , T = epochs)



    plot_multiple_cumulative([cum_mean_rewards , eps_cum_mean_reward]
                              , labels = ["Random "+str(times_each_arm) + f" mean {mean_reward}" , 
                                         "Greedy " + str(eps_times_each_arm) + f" mean {eps_mean_reward}"] ,
                                title= "Random vs Greedy")

def run_multiple_episodes(env_K, epsilon, alpha, T, n_episodes=20):
    """
    Lance n_episodes indépendants pour un couple (epsilon, alpha),
    et renvoie la courbe de récompense moyenne cumulée moyenne
    sur ces épisodes.
    """ 
    all_cum_means = []
    for seed in range(n_episodes):
        rng = np.random.default_rng(seed=seed)
        env = KArmedBandit(env_K.K, rng=rng)
        cum_mean , _ , _ = run_episode(env, epsilon=epsilon, alpha=alpha, T=T, rng=rng)
        all_cum_means.append(cum_mean)
    all_cum_means = np.stack(all_cum_means, axis=0) # (n_episodes, T)
    mean_curve = all_cum_means.mean(axis=0)
    return mean_curve

def compare_multiple_eps(alpha = 0.01):
    eps = [0 , 0.05 ,0.1 ,0.3]
    epochs = 2000

    env = KArmedBandit(K=4)
    eps_dict = {e : None for e in eps}
    for e in eps:
        eps_dict[e] = run_multiple_episodes(env , epsilon = e , alpha = alpha , T = epochs )
    
    plot_multiple_cumulative(list(eps_dict.values()) , labels = map(lambda x : rf"$\epsilon={x}$", list(eps_dict.keys())) )

def compare_multiple_alpha(eps = 0):
    alphas = [0.01,0.05,0.15]
    epochs = 2000

    env = KArmedBandit(K=4)
    alpha_dict = {a : None for a in alphas}
    for a in alphas:
        alpha_dict[a] = run_multiple_episodes(env , epsilon = eps , alpha = a , T = epochs )
    
    plot_multiple_cumulative(list(alpha_dict.values()) , labels = map(lambda x : rf"$\alpha={x}$", list(alpha_dict.keys())) )


def compute_arm_quality(eps  , alpha , epochs):
    env = KArmedBandit( K=4)
    epochs = 2000
    eps_cum_mean_reward , eps_mean_reward , eps_times_each_arm , action_list  = run_episode(env , eps , alpha , T = epochs)
    plot_chosen_arm_quality(env, action_list,epochs)
    plot_true_probs(env , epochs)

def compute_regrets(eps,alpha,epochs):
    env = KArmedBandit(K=4)
    epochs = 2000
    s = run_episode_with_metric(env , eps , alpha , epochs)
    plot_multiple_cumulative([s.cum_mean_rewards , s.regrets , s.cum_mean_rewards + s.regrets]  , 
                             labels = ["mean reward " , "regrets" , "sum"])
    plt.plot(np.arange(1 , epochs + 1) ,  s.qualities , label = "qualities")
    cumsum_gaps = np.insert(np.cumsum(s.gaps), 0,0)
    running_average = (cumsum_gaps[50:] - cumsum_gaps[:-50])/50 # s_50 - 0 , s_51 - s_1
    print(s.optimal_choice_rate)
    plt.plot(np.arange(1 , len(running_average) + 1) , running_average , label = "running average" )
    plt.show()
    

if __name__ == "__main__":
    #compare_multiple_eps()
    #compare_multiple_alpha()
    #compute_arm_quality(eps = 0.15 , alpha=0.3 , epochs = 2000)
    compute_regrets(eps = 0.1 , alpha=0.3 , epochs = 2000)