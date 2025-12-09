# agent.py
import numpy as np

class EpsilonGreedyAgent:
    """
    Agent epsilon-glouton pour un bandit à K bras.
    Il maintient une estimation de la récompense moyenne Q_k pour chaque bras k.
    """
    def __init__(self, K: int, epsilon: float, alpha: float, rng=None):
        """
        K       : nombre de bras
        epsilon : probabilité d'exploration
        alpha   : taux d'apprentissage (0 < alpha <= 1)
        """
        self.K = K
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        # Estimations de la récompense moyenne Q_k
        self.Q = np.zeros(K, dtype=float)
        self.rng = np.random.default_rng() if rng is None else rng

    def select_action(self) -> int:
        """
        Choisit une action selon la stratégie epsilon-glouton :
        - avec probabilité epsilon : action aléatoire (exploration)
        - sinon                   : action argmax_k Q_k (exploitation)
        """
        # TODO :
        u = np.random.rand()
        if u < self.epsilon:
            action = np.random.choice(range(self.K))
        else:
            action = np.argmax(self.Q)
        return action
        # - tirer un nombre u dans [0 1[
        # - si u < epsilon : renvoyer une action aléatoire
        # - sinon : renvoyer l'indice du max de Q

    def update(self, action: int, reward: float) -> None:
        """
        Met à jour l'estimation Q[action] à partir de la récompense observée.
        Cette mise à jour est une moyenne mobile exponentielle où :
        - alpha contrôle le poids des nouvelles observations
        - Si alpha = 1 : seule la dernière récompense compte
        - Si alpha petit : l'historique a plus de poids
        - Contrairement à la moyenne arithmétique, cela permet de suivre
        un environnement non-stationnaire.
        Q[action] <- Q[action] + alpha * (reward - Q[action])
        """
        self.Q[action] = self.Q[action] + self.alpha * (reward  - self.Q[action])
        # TODO : implémenter la mise à jour de Q[action]
