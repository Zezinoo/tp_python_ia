# plot_utils.py
import numpy as np
import matplotlib.pyplot as plt


def plot_cumulative_mean(cumulative_mean, label=None, title=None):
    """
    Trace une courbe de récompense moyenne cumulée en fonction du temps.
    cumulative_mean : array de taille T
    """
    T = len(cumulative_mean)
    fig, ax = plt.subplots()
    ax.plot(np.arange(T), cumulative_mean, label=label)
    ax.set_xlabel("Temps t")
    ax.set_ylabel("Récompense moyenne cumulée")
    if title:
        ax.set_title(title)
    if label:
        ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_multiple_cumulative(curves, labels, title=None):
    """
    Trace plusieurs courbes de récompense moyenne cumulée sur le même graphique.
    curves : liste de arrays (chacun de taille T)
    labels : liste de labels (même taille que curves)
    """
    fig, ax = plt.subplots()
    T = len(curves[0])
    ts = np.arange(T)

    for cm, lab in zip(curves, labels):
        ax.plot(ts, cm, label=lab)

    ax.set_xlabel("Temps t")
    ax.set_ylabel("Récompense moyenne cumulée")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_true_probs(env, T):
    """
    Visualise l'évolution des probabilités réelles de chaque bras sur [0, T).
    """
    ts = np.arange(T)
    probs = np.stack([env.true_probs(int(t)) for t in ts], axis=0)  # (T, K)

    fig, ax = plt.subplots()
    for k in range(env.K):
        ax.plot(ts, probs[:, k], label=f"Bras {k}")

    ax.set_xlabel("Temps t")
    ax.set_ylabel("Probabilité réelle de gain")
    ax.set_title("Probabilités réelles des bras (non stationnaires)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_chosen_arm_quality(env, actions, T):
    """
    Trace la probabilité réelle du bras choisi à chaque pas de temps :
    pour chaque t, on regarde p_k(t) où k = action(t).
    """
    ts = np.arange(T)
    true_p_chosen = []

    for t, a_t in zip(ts, actions):
        p_t = env.true_probs(int(t))[a_t]
        true_p_chosen.append(p_t)

    true_p_chosen = np.array(true_p_chosen)

    fig, ax = plt.subplots()
    ax.plot(ts, true_p_chosen, label="Probabilité réelle du bras choisi")
    ax.set_xlabel("Temps t")
    ax.set_ylabel("Probabilité réelle de gain")
    ax.set_title("Qualité du bras choisi au cours du temps")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()
