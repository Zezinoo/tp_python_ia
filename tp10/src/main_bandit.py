#!/usr/bin/env python3

# main_bandit.py
import numpy as np


from bandit_env import KArmedBandit
from agent import EpsilonGreedyAgent
from plot_utils import (
    plot_cumulative_mean,
    plot_multiple_cumulative,
    plot_true_probs,
    plot_chosen_arm_quality,
)

def run_episode(env, epsilon: float, alpha: float, T: int, rng=None):
    """
    Simule un épisode de T pas de temps avec un agent epsilon-glouton.
    Renvoie la récompense moyenne cumulée (array de taille T).
    """
    if rng is None:
        rng = np.random.default_rng()
    agent = EpsilonGreedyAgent(env.K, epsilon=epsilon, alpha=alpha, rng=rng)
    rewards = []
    # TODO :
    # pour t de 0 à T-1 :
    # - choisir une action avec agent.select_action()
    # - jouer cette action dans l'environnement : env.step(action, t)
    # - appeler agent.update(action, reward)
    # - stocker la récompense

    reward_dict = { i : [] for i in range(env.K)}

    for e in range(T):
        action = agent.select_action()
        reward = env.step(action , e)
        agent.update(action , reward)
        reward_dict[action].append((e , reward))



    return treat_reward_dict(reward_dict , T)


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
                                         "Greedy " + str(eps_times_each_arm) + f" mean {eps_mean_reward} eps= {0.1} alpha=  {0.05} "] ,
                                title= "Random vs Greedy")





if __name__ == "__main__":
    main()
